import os
import httpx
from urllib.parse import urlparse
from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI, AuthenticationError, PermissionDeniedError, NotFoundError, RateLimitError, APIError, APIConnectionError, BadRequestError
import json

from bolna.constants import DEFAULT_LANGUAGE_CODE, ROLE_SYSTEM, ROLE_ASSISTANT, GPT5_MODEL_PREFIX
from bolna.enums import ReasoningEffort, Verbosity, ResponseStreamEvent, ResponseItemType
from bolna.helpers.utils import convert_to_request_log, compute_function_pre_call_message, now_ms
from .llm import BaseLLM
from .tool_call_accumulator import ToolCallAccumulator
from .format_adapter import MessageFormatAdapter
from .types import LLMStreamChunk, LatencyData, FunctionCallPayload
from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)
load_dotenv()


class OpenAiLLM(BaseLLM):
    def __init__(self, max_tokens=100, buffer_size=40, model="gpt-3.5-turbo-16k", temperature=0.1, language=DEFAULT_LANGUAGE_CODE, **kwargs):
        super().__init__(max_tokens, buffer_size)
        self.model = model

        self.custom_tools = kwargs.get("api_tools", None)
        self.language = language
        logger.info(f"API Tools {self.custom_tools}")
        if self.custom_tools is not None:
            self.trigger_function_call = True
            self.api_params = self.custom_tools['tools_params']
            logger.info(f"Function dict {self.api_params}")
            self.tools = self.custom_tools['tools']
        else:
            self.trigger_function_call = False

        self.started_streaming = False
        logger.info(f"Initializing OpenAI LLM with model: {self.model} and maxc tokens {max_tokens}")
        self.max_tokens = max_tokens
        self.temperature = temperature

        max_tokens_key = "max_tokens"
        self.model_args = {}
        if model.startswith(GPT5_MODEL_PREFIX):
            max_tokens_key = "max_completion_tokens"
            self.model_args["reasoning_effort"] = kwargs.get("reasoning_effort", None) or ReasoningEffort.LOW.value
            self.model_args["verbosity"] = kwargs.get("verbosity", None) or Verbosity.LOW.value

        self.model_args.update({max_tokens_key: self.max_tokens, "temperature": self.temperature, "model": self.model})

        self.model_args["service_tier"] = kwargs.get("service_tier", "default")

        limits = httpx.Limits(
            max_connections=50,
            max_keepalive_connections=50,
            keepalive_expiry=30
        )
        http_client = httpx.AsyncClient(
            limits=limits,
            timeout=httpx.Timeout(600.0, connect=10.0),
            http2=True
        )

        if kwargs.get("provider", "openai") == "custom":
            base_url = kwargs.get("base_url")
            api_key = kwargs.get('llm_key', None)
            self.async_client = AsyncOpenAI(base_url=base_url, api_key=api_key, http_client=http_client)
        else:
            llm_key = kwargs.get('llm_key', os.getenv('OPENAI_API_KEY'))
            base_url = kwargs.get('base_url')
            if base_url:
                self.async_client = AsyncOpenAI(base_url=base_url, api_key=llm_key, http_client=http_client)
            else:
                self.async_client = AsyncOpenAI(api_key=llm_key, http_client=http_client)
            api_key = llm_key
        self.llm_host = urlparse(base_url).netloc if base_url else None
        self.assistant_id = kwargs.get("assistant_id", None)
        if self.assistant_id:
            logger.info(f"Initializing OpenAI assistant with assistant id {self.assistant_id}")
            self.openai = OpenAI(api_key=api_key)
            #self.thread_id = self.openai.beta.threads.create().id
            self.model_args = {"max_completion_tokens": self.max_tokens, "temperature": self.temperature}
            my_assistant = self.openai.beta.assistants.retrieve(self.assistant_id)
            if my_assistant.tools is not None:
                self.tools = [i for i in my_assistant.tools if i.type == "function"]
            #logger.info(f'thread id : {self.thread_id}')
        self.run_id = kwargs.get("run_id", None)

        # Responses API state
        self.use_responses_api = kwargs.get("use_responses_api", False)
        self.previous_response_id = None

    async def generate_stream(self, messages, synthesize=True, request_json=False, meta_info=None, tool_choice=None):
        if self.use_responses_api:
            async for chunk in self._generate_stream_responses(messages, synthesize, request_json, meta_info, tool_choice):
                yield chunk
        else:
            async for chunk in self._generate_stream_chat(messages, synthesize, request_json, meta_info, tool_choice):
                yield chunk

    async def _generate_stream_chat(self, messages, synthesize=True, request_json=False, meta_info=None, tool_choice=None):
        if not messages or len(messages) == 0:
            raise Exception("No messages provided")

        response_format = self.get_response_format(request_json)
        model_args = {
            **self.model_args,
            "response_format": response_format,
            "messages": messages,
            "stream": True,
            "user": f"{self.run_id}#{meta_info['turn_id']}"
        }

        if not self.model.startswith(GPT5_MODEL_PREFIX):
            model_args["stop"] = ["User:"]


        if self.trigger_function_call:
            model_args["tools"] = json.loads(self.tools) if isinstance(self.tools, str) else self.tools
            model_args["tool_choice"] = tool_choice or "auto"
            model_args["parallel_tool_calls"] = False


        answer, buffer = "", ""
        tools = model_args.get("tools", [])
        accumulator = None
        if self.trigger_function_call:
            accumulator = ToolCallAccumulator(self.api_params, tools, self.language, self.model, self.run_id)

        start_time = now_ms()
        first_token_time = None
        latency_data = None
        service_tier = None

        try:
            completion_stream = await self.async_client.chat.completions.create(**model_args)
        except AuthenticationError as e:
            logger.error(f"OpenAI authentication failed: Invalid or expired API key - {e}")
            raise
        except PermissionDeniedError as e:
            logger.error(f"OpenAI permission denied (403): {e}")
            raise
        except NotFoundError as e:
            logger.error(f"OpenAI resource not found (404): Check model name or endpoint - {e}")
            raise
        except RateLimitError as e:
            logger.error(f"OpenAI rate limit exceeded: {e}")
            raise
        except APIConnectionError as e:
            logger.error(f"OpenAI connection error: {e}")
            raise
        except APIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise
        except Exception as e:
            logger.error(f"OpenAI unexpected error: {e}")
            raise

        async for chunk in completion_stream:
            now = now_ms()
            if hasattr(chunk, 'service_tier') and chunk.service_tier:
                service_tier = chunk.service_tier
                if latency_data:
                    latency_data.service_tier = service_tier

            if not first_token_time:
                first_token_time = now
                self.started_streaming = True

                latency_data = LatencyData(
                    sequence_id=meta_info.get("sequence_id"),
                    first_token_latency_ms=first_token_time - start_time,
                    total_stream_duration_ms=None,
                    service_tier=service_tier,
                    llm_host=self.llm_host,
                )

            delta = chunk.choices[0].delta

            if hasattr(delta, 'tool_calls') and delta.tool_calls and accumulator:
                if buffer:
                    yield LLMStreamChunk(data=buffer, end_of_stream=True, latency=latency_data)
                    buffer = ""

                accumulator.process_delta(delta.tool_calls)

                pre_call = accumulator.get_pre_call_message(meta_info)
                if pre_call:
                    yield LLMStreamChunk(data=pre_call[0], end_of_stream=True, latency=latency_data, function_name=pre_call[1], function_message=pre_call[2])

            elif hasattr(delta, 'content') and delta.content is not None:
                if accumulator:
                    accumulator.received_textual = True
                answer += delta.content
                buffer += delta.content
                if synthesize and len(buffer) >= self.buffer_size:
                    split = buffer.rsplit(" ", 1)
                    yield LLMStreamChunk(data=split[0], end_of_stream=False, latency=latency_data)
                    buffer = split[1] if len(split) > 1 else ""

        if latency_data:
            latency_data.total_stream_duration_ms = now_ms() - start_time

        if accumulator and accumulator.final_tool_calls:
            api_call_payload = accumulator.build_api_payload(model_args, meta_info, answer)
            if api_call_payload:
                yield LLMStreamChunk(data=api_call_payload, end_of_stream=False, latency=latency_data, is_function_call=True)

        if synthesize:
            yield LLMStreamChunk(data=buffer, end_of_stream=True, latency=latency_data)
        else:
            yield LLMStreamChunk(data=answer, end_of_stream=True, latency=latency_data)

        self.started_streaming = False

    def _build_responses_input(self, messages):
        """Build (instructions, input_items) for Responses API.

        When previous_response_id is set, only sends new items since last server response.
        Otherwise sends the full conversation.
        """
        if self.previous_response_id:
            return self._extract_new_input(messages)
        return MessageFormatAdapter.chat_to_responses_input(messages)

    def _extract_new_input(self, messages):
        """Extract only new items since last server response.

        The server already has all context up to its last output. We only need to send:
        - New user messages (after the last assistant message)
        - Function call outputs (tool results)
        """
        instructions = ""
        if messages and messages[0].get("role") == ROLE_SYSTEM:
            instructions = messages[0].get("content", "")

        last_assistant_idx = -1
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == ROLE_ASSISTANT:
                last_assistant_idx = i
                break

        if last_assistant_idx < 0:
            _, input_items = MessageFormatAdapter.chat_to_responses_input(messages)
            return instructions, input_items

        new_messages = messages[last_assistant_idx + 1:]
        _, input_items = MessageFormatAdapter.chat_to_responses_input(new_messages)
        return instructions, input_items

    @staticmethod
    def _is_stale_response_error(error):
        """Check if an API error is likely due to a stale previous_response_id.

        All known stale-response errors from OpenAI are BadRequestError (HTTP 400).
        Since this is only called when previous_response_id is set, a BadRequestError
        is most likely caused by it. The retry is self-limiting: it clears the ID,
        so a second failure will raise the real error.
        """
        return isinstance(error, BadRequestError)

    def _parse_tools(self):
        """Parse tools from string or list format."""
        if not self.trigger_function_call:
            return []
        return json.loads(self.tools) if isinstance(self.tools, str) else self.tools

    async def _generate_stream_responses(self, messages, synthesize=True, request_json=False, meta_info=None, tool_choice=None):
        if not messages:
            raise ValueError("No messages provided")

        instructions, input_items = self._build_responses_input(messages)
        responses_tools = MessageFormatAdapter.chat_tools_to_responses_tools(self._parse_tools())

        create_kwargs = {
            "model": self.model,
            "instructions": instructions or None,
            "input": input_items,
            "stream": True,
            "store": True,
            "max_output_tokens": self.max_tokens,
            "temperature": self.temperature,
            "user": f"{self.run_id}#{meta_info['turn_id']}" if meta_info else None,
        }

        service_tier = self.model_args.get("service_tier")
        if service_tier:
            create_kwargs["service_tier"] = service_tier

        if self.model.startswith(GPT5_MODEL_PREFIX):
            reasoning_effort = self.model_args.get("reasoning_effort")
            if reasoning_effort:
                create_kwargs["reasoning"] = {"effort": reasoning_effort}

        if self.previous_response_id:
            create_kwargs["previous_response_id"] = self.previous_response_id

        if responses_tools:
            create_kwargs["tools"] = responses_tools
            create_kwargs["tool_choice"] = tool_choice or "auto"
            create_kwargs["parallel_tool_calls"] = False

        if request_json:
            create_kwargs["text"] = {"format": {"type": "json_object"}}

        if not self.model.startswith(GPT5_MODEL_PREFIX):
            text_config = create_kwargs.get("text", {})
            text_config["stop"] = ["User:"]
            create_kwargs["text"] = text_config

        answer, buffer = "", ""
        func_call_args = {}  # item_id -> accumulated arguments
        func_call_names = {}  # item_id -> function name
        func_call_ids = {}  # item_id -> call_id
        gave_pre_call_msg = False
        received_textual = False

        start_time = now_ms()
        first_token_time = None
        latency_data = None
        service_tier = None

        try:
            stream = await self.async_client.responses.create(**create_kwargs)
        except Exception as e:
            if self.previous_response_id and self._is_stale_response_error(e):
                logger.warning(f"Stale previous_response_id, retrying with full history: {e}")
                self.previous_response_id = None
                async for chunk in self._generate_stream_responses(messages, synthesize, request_json, meta_info, tool_choice):
                    yield chunk
                return
            logger.error(f"OpenAI Responses API error: {e}")
            raise

        async for event in stream:
            now = now_ms()

            if event.type == ResponseStreamEvent.CREATED:
                self.previous_response_id = event.response.id
                service_tier = getattr(event.response, 'service_tier', None)
                continue

            if event.type == ResponseStreamEvent.FAILED:
                error_info = getattr(event.response, 'error', None) or getattr(event.response, 'last_error', None)
                logger.error(f"Responses API stream failed: {error_info}")
                self.previous_response_id = None
                raise APIError(
                    message=f"Response failed: {error_info}",
                    request=None, body=None
                )

            if event.type == ResponseStreamEvent.INCOMPLETE:
                logger.warning("Responses API stream incomplete, partial response returned")
                self.previous_response_id = None
                break

            if not first_token_time and event.type in (ResponseStreamEvent.OUTPUT_TEXT_DELTA, ResponseStreamEvent.FUNCTION_CALL_ARGS_DELTA):
                first_token_time = now
                self.started_streaming = True
                latency_data = LatencyData(
                    sequence_id=meta_info.get("sequence_id"),
                    first_token_latency_ms=first_token_time - start_time,
                    total_stream_duration_ms=None,
                    service_tier=service_tier,
                    llm_host=self.llm_host,
                )

            if event.type == ResponseStreamEvent.OUTPUT_TEXT_DELTA:
                received_textual = True
                answer += event.delta
                buffer += event.delta
                if synthesize and len(buffer) >= self.buffer_size:
                    split = buffer.rsplit(" ", 1)
                    yield LLMStreamChunk(data=split[0], end_of_stream=False, latency=latency_data)
                    buffer = split[1] if len(split) > 1 else ""

            elif event.type == ResponseStreamEvent.OUTPUT_ITEM_ADDED:
                item = event.item
                if item.type == ResponseItemType.FUNCTION_CALL:
                    if buffer:
                        yield LLMStreamChunk(data=buffer, end_of_stream=True, latency=latency_data)
                        buffer = ""
                    func_call_args[item.id] = ""
                    func_call_names[item.id] = item.name
                    func_call_ids[item.id] = item.call_id

                    if not gave_pre_call_msg and not received_textual and self.trigger_function_call:
                        gave_pre_call_msg = True
                        api_tool_pre_call_message = self.api_params.get(item.name, {}).get('pre_call_message', None)
                        detected_lang = meta_info.get('detected_language') if meta_info else None
                        active_language = detected_lang or self.language
                        pre_msg = compute_function_pre_call_message(active_language, item.name, api_tool_pre_call_message)
                        if pre_msg:
                            yield LLMStreamChunk(data=pre_msg, end_of_stream=True, latency=latency_data, function_name=item.name, function_message=api_tool_pre_call_message)

            elif event.type == ResponseStreamEvent.FUNCTION_CALL_ARGS_DELTA:
                func_call_args[event.item_id] = func_call_args.get(event.item_id, "") + event.delta

            elif event.type == ResponseStreamEvent.COMPLETED:
                if hasattr(event.response, 'id'):
                    self.previous_response_id = event.response.id
                service_tier = service_tier or getattr(event.response, 'service_tier', None)
                break

        if latency_data:
            latency_data.total_stream_duration_ms = now_ms() - start_time
            if service_tier:
                latency_data.service_tier = service_tier

        if func_call_args and self.trigger_function_call:
            first_item_id = next(iter(func_call_args))
            func_name = func_call_names[first_item_id]
            call_id = func_call_ids[first_item_id]
            arguments_str = func_call_args[first_item_id]

            if func_name in self.api_params:
                func_conf = self.api_params[func_name]
                logger.info(f"Payload to send {arguments_str} func_dict {func_conf}")

                method = func_conf.get('method')
                api_call_payload = FunctionCallPayload(
                    url=func_conf.get('url'),
                    method=method.lower() if method else None,
                    param=func_conf.get('param'),
                    api_token=func_conf.get('api_token'),
                    headers=func_conf.get('headers'),
                    model_args=create_kwargs,
                    meta_info=meta_info,
                    called_fun=func_name,
                    model_response=[{
                        "index": 0,
                        "id": call_id,
                        "function": {"name": func_name, "arguments": arguments_str},
                        "type": "function",
                    }],
                    tool_call_id=call_id,
                    textual_response=answer.strip() if received_textual else None,
                )

                tool_spec = next((t for t in responses_tools if t["name"] == func_name), None)
                if tool_spec:
                    try:
                        parsed_args = json.loads(arguments_str)
                        required_keys = tool_spec.get("parameters", {}).get("required", [])
                        if tool_spec.get("parameters") is not None and all(k in parsed_args for k in required_keys):
                            convert_to_request_log(arguments_str, meta_info, self.model, "llm",
                                                   direction="response", is_cached=False, run_id=self.run_id)
                            for k, v in parsed_args.items():
                                setattr(api_call_payload, k, v)
                        else:
                            api_call_payload.resp = None
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.error(f"Error parsing function arguments: {e}")
                        api_call_payload.resp = None
                else:
                    api_call_payload.resp = None

                yield LLMStreamChunk(data=api_call_payload, end_of_stream=False, latency=latency_data, is_function_call=True)

        if synthesize:
            yield LLMStreamChunk(data=buffer, end_of_stream=True, latency=latency_data)
        else:
            yield LLMStreamChunk(data=answer, end_of_stream=True, latency=latency_data)

        self.started_streaming = False

    async def generate(self, messages, request_json=False, ret_metadata=False):
        if self.use_responses_api:
            return await self._generate_responses(messages, request_json, ret_metadata)
        return await self._generate_chat(messages, request_json, ret_metadata)

    async def _generate_chat(self, messages, request_json=False, ret_metadata=False):
        response_format = self.get_response_format(request_json)

        try:
            completion = await self.async_client.chat.completions.create(model=self.model, temperature=0.0, messages=messages,
                                                                         stream=False, response_format=response_format)
            res = completion.choices[0].message.content
            if ret_metadata:
                metadata = {
                    "llm_host": self.llm_host,
                    "service_tier": completion.service_tier if hasattr(completion, 'service_tier') else None
                }
                return res, metadata
            else:
                return res
        except AuthenticationError as e:
            logger.error(f"OpenAI authentication failed: Invalid or expired API key - {e}")
            raise
        except PermissionDeniedError as e:
            logger.error(f"OpenAI permission denied (403): {e}")
            raise
        except NotFoundError as e:
            logger.error(f"OpenAI resource not found (404): Check model name or endpoint - {e}")
            raise
        except RateLimitError as e:
            logger.error(f"OpenAI rate limit exceeded: {e}")
            raise
        except APIConnectionError as e:
            logger.error(f"OpenAI connection error: {e}")
            raise
        except APIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise
        except Exception as e:
            logger.error(f"OpenAI unexpected error: {e}")
            raise

    async def _generate_responses(self, messages, request_json=False, ret_metadata=False):
        instructions, input_items = self._build_responses_input(messages)

        create_kwargs = {
            "model": self.model,
            "instructions": instructions or None,
            "input": input_items,
            "store": True,
            "max_output_tokens": self.max_tokens,
            "temperature": 0.0,
        }

        service_tier = self.model_args.get("service_tier")
        if service_tier:
            create_kwargs["service_tier"] = service_tier

        if self.model.startswith(GPT5_MODEL_PREFIX):
            reasoning_effort = self.model_args.get("reasoning_effort")
            if reasoning_effort:
                create_kwargs["reasoning"] = {"effort": reasoning_effort}

        if self.previous_response_id:
            create_kwargs["previous_response_id"] = self.previous_response_id

        if request_json:
            create_kwargs["text"] = {"format": {"type": "json_object"}}

        try:
            response = await self.async_client.responses.create(**create_kwargs)
            self.previous_response_id = response.id
            res = response.output_text

            if ret_metadata:
                metadata = {
                    "llm_host": self.llm_host,
                    "service_tier": getattr(response, 'service_tier', None),
                }
                return res, metadata
            return res
        except Exception as e:
            if self.previous_response_id and self._is_stale_response_error(e):
                logger.warning(f"Stale previous_response_id, retrying with full history: {e}")
                self.previous_response_id = None
                return await self._generate_responses(messages, request_json, ret_metadata)
            logger.error(f"OpenAI Responses API error: {e}")
            raise

    def invalidate_response_chain(self):
        self.previous_response_id = None

    def get_response_format(self, is_json_format: bool):
        if is_json_format and self.model in ('gpt-4-1106-preview', 'gpt-3.5-turbo-1106', 'gpt-4o-mini', 'gpt-4.1-mini'):
            return {"type": "json_object"}
        else:
            return {"type": "text"}
