import json
import re
from typing import Optional

from openai import BadRequestError, APIError

from bolna.constants import GPT5_MODEL_PREFIX
from bolna.enums import ChatRole, ResponseStreamEvent, ResponseItemType, LogComponent, LogDirection
from bolna.helpers.utils import convert_to_request_log, compute_function_pre_call_message, now_ms
from .llm import BaseLLM
from .message_models import MessageFormatAdapter
from .types import APIParams, LLMStreamChunk, LatencyData, FunctionCallPayload
from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)


def _clean_rescue_answer(answer: str) -> str | None:
    """Strip leftover 'functions' / 'functions.xxx' tokens from a rescue textual_response."""
    cleaned = re.sub(r"\bfunctions(\.\w+)?\b", "", answer).strip()
    return cleaned if cleaned else None


class OpenAICompatibleLLM(BaseLLM):
    """Base class for OpenAI-API-compatible LLM providers.

    Subclasses must:
    - Call _init_responses_api() during __init__
    - Override _responses_client property if they need a different client
    """

    @staticmethod
    def _find_tool_call_end(text):
        """Return the index after the closing brace/paren of a text-based tool call, or -1 if incomplete."""
        m = re.search(r"functions\.\w+\s*[({]", text)
        if not m:
            return -1
        start = m.end() - 1
        depth = 0
        for i in range(start, len(text)):
            if text[i] in "({":
                depth += 1
            elif text[i] in ")}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    if end < len(text) and text[end] == ";":
                        end += 1
                    return end
        return -1

    @staticmethod
    def _parse_text_tool_call(text):
        """Parse functions.name({...}) or functions.name { ... } from text.

        Returns (func_name, args_json_str) or None if unparseable.
        Handles both strict JSON and JS-style unquoted keys.
        """
        m = re.search(r"functions\.(\w+)\s*([({])", text)
        if not m:
            return None

        func_name = m.group(1)
        start = m.start(2)
        depth = 0
        end = -1
        for i in range(start, len(text)):
            if text[i] in "({":
                depth += 1
            elif text[i] in ")}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break

        if end == -1:
            return None

        raw = text[start:end]
        if raw.startswith("("):
            raw = raw[1:-1].strip()

        try:
            json.loads(raw)
            return func_name, raw
        except json.JSONDecodeError:
            pass

        # Fix JS-style unquoted keys: notes: "..." → "notes": "..."
        fixed = re.sub(r'(?<!["\w])([a-zA-Z_]\w*)\s*:', r'"\1":', raw)
        try:
            json.loads(fixed)
            return func_name, fixed
        except json.JSONDecodeError:
            logger.warning(f"Could not parse text tool call args for {func_name}: {raw[:100]}")
            return None

    def _try_rescue_text_tool_call(self, text, model_args, meta_info, answer, latency_data):
        """Attempt to execute a tool call that was output as plain text instead of delta.tool_calls.

        Parses the text, builds FunctionCallPayload directly from api_params, and returns an
        LLMStreamChunk with is_function_call=True. Returns None if rescue is not possible
        (no tools registered, unknown function, or unparseable args) so the caller can fall
        back to sending the raw text to TTS.
        """
        if not self.trigger_function_call:
            return None

        parsed = self._parse_text_tool_call(text)
        if not parsed:
            return None

        func_name, args_str = parsed

        if func_name not in self.api_params:
            logger.warning(f"Text tool call rescue: '{func_name}' not in api_params, falling back to TTS")
            return None

        func_conf = self.api_params[func_name]
        method = func_conf.get("method")

        api_call_payload = FunctionCallPayload(
            url=func_conf.get("url"),
            method=method.lower() if method else None,
            param=func_conf.get("param"),
            api_token=func_conf.get("api_token"),
            headers=func_conf.get("headers"),
            model_args=model_args,
            meta_info=meta_info,
            called_fun=func_name,
            model_response=[
                {
                    "index": 0,
                    "id": f"rescued_{func_name}",
                    "function": {"name": func_name, "arguments": args_str},
                    "type": "function",
                }
            ],
            tool_call_id=f"rescued_{func_name}",
            textual_response=_clean_rescue_answer(answer),
        )

        # Mirror ToolCallAccumulator.build_api_payload: validate required keys against the tool spec
        tools_list = json.loads(self.tools) if isinstance(self.tools, str) else (self.tools or [])
        tool_spec = next((t for t in tools_list if t.get("function", {}).get("name") == func_name), None)

        try:
            parsed_args = json.loads(args_str)
            if tool_spec and tool_spec["function"].get("parameters") is not None:
                required_keys = tool_spec["function"]["parameters"].get("required", [])
                missing = [k for k in required_keys if k not in parsed_args]
                if missing:
                    logger.warning(
                        f"Text tool call rescue: '{func_name}' missing required args {missing}, zeroing resp"
                    )
                    api_call_payload.resp = None
                    return LLMStreamChunk(
                        data=api_call_payload, end_of_stream=False, latency=latency_data, is_function_call=True
                    )
            for k, v in parsed_args.items():
                setattr(api_call_payload, k, v)
            logger.info(f"Text tool call rescue succeeded: {func_name}")
        except Exception as e:
            logger.error(f"Text tool call rescue: failed to apply args for {func_name}: {e}")
            api_call_payload.resp = None

        return LLMStreamChunk(data=api_call_payload, end_of_stream=False, latency=latency_data, is_function_call=True)

    def _init_responses_api(self, use_responses_api: bool = False, compact_threshold: Optional[int] = None):
        self.use_responses_api = use_responses_api
        self.previous_response_id = None
        self._pending_call_ids: set[str] = set()
        self.compact_threshold = compact_threshold

    @property
    def _responses_client(self):
        """Return the async client for Responses API calls.

        Default: ``self.async_client``. Override in subclasses that need a
        different client (e.g. Azure v1 endpoint).
        """
        return self.async_client

    def _build_responses_input(self, messages):
        """Build (instructions, input_items) for Responses API.

        When previous_response_id is set, only sends new items since last
        server response. Otherwise sends the full conversation.

        Tool call guard: if the previous response made function calls whose
        outputs are not yet in the history, fall back to full context to
        avoid 400 errors from the server.
        """
        if self.previous_response_id:
            if self._pending_call_ids:
                completed = {m.get("tool_call_id") for m in messages if m.get("role") == ChatRole.TOOL}
                if not self._pending_call_ids.issubset(completed):
                    logger.info("Pending tool call outputs missing, sending full context")
                    self.previous_response_id = None
                    return MessageFormatAdapter.chat_to_responses_input(messages)
            return self._extract_new_input(messages)
        return MessageFormatAdapter.chat_to_responses_input(messages)

    def _extract_new_input(self, messages):
        """Extract only new items since last server response.

        The server already has all context up to its last output. We only
        need to send:
        - New user messages (after the last assistant message)
        - Function call outputs (tool results)
        """
        instructions = ""
        if messages and messages[0].get("role") == ChatRole.SYSTEM:
            instructions = messages[0].get("content", "")

        last_assistant_idx = -1
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == ChatRole.ASSISTANT:
                last_assistant_idx = i
                break

        if last_assistant_idx < 0:
            _, input_items = MessageFormatAdapter.chat_to_responses_input(messages)
            return instructions, input_items

        new_messages = messages[last_assistant_idx + 1 :]
        _, input_items = MessageFormatAdapter.chat_to_responses_input(new_messages)
        return instructions, input_items

    @staticmethod
    def _is_stale_response_error(error):
        """Check if an API error is likely due to a stale previous_response_id.

        All known stale-response errors from OpenAI are BadRequestError
        (HTTP 400). Since this is only called when previous_response_id is
        set, a BadRequestError is most likely caused by it. The retry is
        self-limiting: it clears the ID, so a second failure will raise the
        real error.
        """
        return isinstance(error, BadRequestError)

    def _parse_tools(self):
        """Parse tools from string or list format."""
        if not self.trigger_function_call:
            return []
        return json.loads(self.tools) if isinstance(self.tools, str) else self.tools

    def invalidate_response_chain(self):
        self.previous_response_id = None
        self._pending_call_ids = set()

    def _build_function_call_chunk(
        self,
        func_call_args,
        func_call_names,
        func_call_ids,
        responses_tools,
        create_kwargs,
        meta_info,
        answer,
        received_textual,
        latency_data,
    ):
        """Build LLMStreamChunk with FunctionCallPayload from accumulated function call data, or None."""
        if not (func_call_args and self.trigger_function_call):
            return None

        first_item_id = next(iter(func_call_args))
        func_name = func_call_names[first_item_id]
        call_id = func_call_ids[first_item_id]
        arguments_str = func_call_args[first_item_id]

        if func_name not in self.api_params:
            return None

        func_conf = APIParams.model_validate(self.api_params[func_name])
        logger.info(f"Payload to send {arguments_str} func_dict {func_conf}")

        api_call_payload = FunctionCallPayload(
            url=func_conf.url,
            method=func_conf.method.lower() if func_conf.method else None,
            param=func_conf.param,
            api_token=func_conf.api_token,
            headers=func_conf.headers,
            model_args=create_kwargs,
            meta_info=meta_info,
            called_fun=func_name,
            model_response=[
                {
                    "index": 0,
                    "id": call_id,
                    "function": {"name": func_name, "arguments": arguments_str},
                    "type": "function",
                }
            ],
            tool_call_id=call_id,
            textual_response=answer.strip() if received_textual else None,
        )

        tool_spec = next((t for t in responses_tools if t["name"] == func_name), None)
        if tool_spec:
            try:
                parsed_args = json.loads(arguments_str)
                required_keys = tool_spec.get("parameters", {}).get("required", [])
                if tool_spec.get("parameters") is not None and all(k in parsed_args for k in required_keys):
                    convert_to_request_log(
                        arguments_str,
                        meta_info,
                        self.model,
                        LogComponent.LLM,
                        direction=LogDirection.RESPONSE,
                        is_cached=False,
                        run_id=self.run_id,
                    )
                    for k, v in parsed_args.items():
                        setattr(api_call_payload, k, v)
                else:
                    api_call_payload.resp = None
            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"Error parsing function arguments: {e}")
                api_call_payload.resp = None
        else:
            api_call_payload.resp = None

        return LLMStreamChunk(data=api_call_payload, end_of_stream=False, latency=latency_data, is_function_call=True)

    def _build_responses_create_kwargs(
        self, messages, meta_info, request_json, tool_choice, *, store=True, stream=None
    ):
        """Build create kwargs common to both HTTP SSE and WebSocket streaming paths."""
        instructions, input_items = self._build_responses_input(messages)
        responses_tools = MessageFormatAdapter.chat_tools_to_responses_tools(self._parse_tools())

        create_kwargs = {
            "model": self.model,
            "instructions": instructions or None,
            "input": input_items,
            "store": store,
            "truncation": "auto",
            "max_output_tokens": self.max_tokens,
            "temperature": self.temperature,
            "user": f"{self.run_id}#{meta_info['turn_id']}" if meta_info else None,
        }

        if stream is not None:
            create_kwargs["stream"] = stream

        if self.compact_threshold:
            create_kwargs["context_management"] = [{"type": "compaction", "compact_threshold": self.compact_threshold}]

        service_tier = self.model_args.get("service_tier")
        if service_tier:
            create_kwargs["service_tier"] = service_tier

        if self.model.startswith(GPT5_MODEL_PREFIX):
            create_kwargs["temperature"] = 1
            reasoning_effort = self.model_args.get("reasoning_effort")
            reasoning_config = {}
            if reasoning_effort:
                reasoning_config["effort"] = reasoning_effort
            reasoning_config["summary"] = "auto"
            create_kwargs["reasoning"] = reasoning_config
            verbosity = self.model_args.get("verbosity")
            if verbosity:
                create_kwargs.setdefault("text", {})["verbosity"] = verbosity

        if self.previous_response_id:
            create_kwargs["previous_response_id"] = self.previous_response_id

        if responses_tools:
            create_kwargs["tools"] = responses_tools
            create_kwargs["tool_choice"] = MessageFormatAdapter.chat_tool_choice_to_responses(tool_choice or "auto")
            create_kwargs["parallel_tool_calls"] = False

        if request_json:
            create_kwargs.setdefault("text", {})["format"] = {"type": "json_object"}

        return create_kwargs, responses_tools

    async def _generate_stream_responses(
        self, messages, synthesize=True, request_json=False, meta_info=None, tool_choice=None
    ):
        if not messages:
            raise ValueError("No messages provided")

        create_kwargs, responses_tools = self._build_responses_create_kwargs(
            messages, meta_info, request_json, tool_choice, store=True, stream=True
        )

        answer, buffer = "", ""
        func_call_args = {}
        func_call_names = {}
        func_call_ids = {}
        gave_pre_call_msg = False
        received_textual = False
        reasoning_summary_parts = []

        start_time = now_ms()
        first_token_time = None
        latency_data = None
        service_tier = None
        llm_host = getattr(self, "llm_host", None)
        response_usage = None

        try:
            stream = await self._responses_client.responses.create(**create_kwargs)
        except Exception as e:
            if self.previous_response_id and self._is_stale_response_error(e):
                logger.warning(f"Stale previous_response_id, retrying with full history: {e}")
                if isinstance(meta_info, dict):
                    meta_info.setdefault("_non_fatal_errors", []).append(
                        {"error_type": "stale_response_id", "error": str(e), "model": self.model}
                    )
                self.previous_response_id = None
                async for chunk in self._generate_stream_responses(
                    messages, synthesize, request_json, meta_info, tool_choice
                ):
                    yield chunk
                return
            logger.error(f"Responses API error: {e}")
            raise

        async for event in stream:
            now = now_ms()

            if event.type == ResponseStreamEvent.CREATED:
                self.previous_response_id = event.response.id
                service_tier = getattr(event.response, "service_tier", None)
                if latency_data is None:
                    latency_data = LatencyData(
                        sequence_id=meta_info.get("sequence_id") if meta_info else None,
                        connection_latency_ms=round(now - start_time, 2),
                    )
                else:
                    latency_data.connection_latency_ms = round(now - start_time, 2)
                continue

            if event.type == ResponseStreamEvent.FAILED:
                error_info = getattr(event.response, "error", None) or getattr(event.response, "last_error", None)
                logger.error(f"Responses API stream failed: {error_info}")
                self.invalidate_response_chain()
                raise APIError(message=f"Response failed: {error_info}", request=None, body=None)

            if event.type == ResponseStreamEvent.INCOMPLETE:
                logger.warning("Responses API stream incomplete, partial response returned")
                self.invalidate_response_chain()
                break

            if not first_token_time and event.type in (
                ResponseStreamEvent.OUTPUT_TEXT_DELTA,
                ResponseStreamEvent.FUNCTION_CALL_ARGS_DELTA,
            ):
                first_token_time = now
                self.started_streaming = True
                latency_data = LatencyData(
                    sequence_id=meta_info.get("sequence_id") if meta_info else None,
                    first_token_latency_ms=first_token_time - start_time,
                    total_stream_duration_ms=None,
                    service_tier=service_tier,
                    llm_host=llm_host,
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
                        func_params = self.api_params.get(item.name)
                        api_tool_pre_call_message = (
                            APIParams.model_validate(func_params).pre_call_message if func_params else None
                        )
                        detected_lang = meta_info.get("detected_language") if meta_info else None
                        active_language = detected_lang or self.language
                        pre_msg = compute_function_pre_call_message(
                            active_language, item.name, api_tool_pre_call_message
                        )
                        if pre_msg:
                            yield LLMStreamChunk(
                                data=pre_msg,
                                end_of_stream=True,
                                latency=latency_data,
                                function_name=item.name,
                                function_message=api_tool_pre_call_message,
                            )

            elif event.type == ResponseStreamEvent.FUNCTION_CALL_ARGS_DELTA:
                func_call_args[event.item_id] = func_call_args.get(event.item_id, "") + event.delta

            elif event.type == ResponseStreamEvent.REASONING_SUMMARY_TEXT_DELTA:
                reasoning_summary_parts.append(event.delta)

            elif event.type == ResponseStreamEvent.COMPLETED:
                if hasattr(event.response, "id"):
                    self.previous_response_id = event.response.id
                self._pending_call_ids = set(func_call_ids.values())
                service_tier = service_tier or getattr(event.response, "service_tier", None)
                if hasattr(event.response, "usage") and event.response.usage:
                    response_usage = event.response.usage
                break

        if latency_data:
            latency_data.total_stream_duration_ms = now_ms() - start_time
            if service_tier:
                latency_data.service_tier = service_tier

        fc_chunk = self._build_function_call_chunk(
            func_call_args,
            func_call_names,
            func_call_ids,
            responses_tools,
            create_kwargs,
            meta_info,
            answer,
            received_textual,
            latency_data,
        )
        if fc_chunk:
            if response_usage:
                fc_chunk.input_tokens = getattr(response_usage, "input_tokens", None)
                fc_chunk.output_tokens = getattr(response_usage, "output_tokens", None)
                _od = getattr(response_usage, "output_tokens_details", None)
                if _od:
                    fc_chunk.reasoning_tokens = getattr(_od, "reasoning_tokens", None)
                _id = getattr(response_usage, "input_tokens_details", None)
                if _id:
                    fc_chunk.cached_tokens = getattr(_id, "cached_tokens", None)
            yield fc_chunk

        usage_kwargs = {}
        if response_usage:
            usage_kwargs["input_tokens"] = getattr(response_usage, "input_tokens", None)
            usage_kwargs["output_tokens"] = getattr(response_usage, "output_tokens", None)
            output_details = getattr(response_usage, "output_tokens_details", None)
            if output_details:
                usage_kwargs["reasoning_tokens"] = getattr(output_details, "reasoning_tokens", None)
            input_details = getattr(response_usage, "input_tokens_details", None)
            if input_details:
                usage_kwargs["cached_tokens"] = getattr(input_details, "cached_tokens", None)

        reasoning_content = "".join(reasoning_summary_parts) if reasoning_summary_parts else None
        if reasoning_content:
            usage_kwargs["reasoning_content"] = reasoning_content

        if synthesize:
            yield LLMStreamChunk(data=buffer, end_of_stream=True, latency=latency_data, **usage_kwargs)
        else:
            yield LLMStreamChunk(data=answer, end_of_stream=True, latency=latency_data, **usage_kwargs)

        self.started_streaming = False

    async def _generate_responses(self, messages, request_json=False, ret_metadata=False, meta_info=None):
        instructions, input_items = self._build_responses_input(messages)

        create_kwargs = {
            "model": self.model,
            "instructions": instructions or None,
            "input": input_items,
            "store": True,
            "truncation": "auto",
            "max_output_tokens": self.max_tokens,
            "temperature": 0.0,  # Intentional: non-streaming uses deterministic output
        }

        if self.compact_threshold:
            create_kwargs["context_management"] = [{"type": "compaction", "compact_threshold": self.compact_threshold}]

        service_tier = self.model_args.get("service_tier")
        if service_tier:
            create_kwargs["service_tier"] = service_tier

        if self.model.startswith(GPT5_MODEL_PREFIX):
            create_kwargs["temperature"] = 1
            reasoning_config = {}
            reasoning_effort = self.model_args.get("reasoning_effort")
            if reasoning_effort:
                reasoning_config["effort"] = reasoning_effort
            reasoning_config["summary"] = "auto"
            create_kwargs["reasoning"] = reasoning_config
            verbosity = self.model_args.get("verbosity")
            if verbosity:
                create_kwargs.setdefault("text", {})["verbosity"] = verbosity

        if self.previous_response_id:
            create_kwargs["previous_response_id"] = self.previous_response_id

        if request_json:
            create_kwargs.setdefault("text", {})["format"] = {"type": "json_object"}

        llm_host = getattr(self, "llm_host", None)

        try:
            response = await self._responses_client.responses.create(**create_kwargs)
            self.previous_response_id = response.id
            res = response.output_text

            if ret_metadata:
                metadata = {
                    "llm_host": llm_host,
                    "service_tier": getattr(response, "service_tier", None),
                }
                if hasattr(response, "usage") and response.usage:
                    metadata["input_tokens"] = getattr(response.usage, "input_tokens", None)
                    metadata["output_tokens"] = getattr(response.usage, "output_tokens", None)
                    output_details = getattr(response.usage, "output_tokens_details", None)
                    if output_details:
                        metadata["reasoning_tokens"] = getattr(output_details, "reasoning_tokens", None)
                    input_details = getattr(response.usage, "input_tokens_details", None)
                    if input_details:
                        metadata["cached_tokens"] = getattr(input_details, "cached_tokens", None)

                reasoning_texts = []
                if hasattr(response, "output") and response.output:
                    for item in response.output:
                        if getattr(item, "type", None) == "reasoning" and hasattr(item, "summary"):
                            for part in item.summary or []:
                                if getattr(part, "text", None):
                                    reasoning_texts.append(part.text)
                if reasoning_texts:
                    metadata["reasoning_content"] = "\n".join(reasoning_texts)

                return res, metadata
            return res
        except Exception as e:
            if self.previous_response_id and self._is_stale_response_error(e):
                logger.warning(f"Stale previous_response_id, retrying with full history: {e}")
                if isinstance(meta_info, dict):
                    meta_info.setdefault("_non_fatal_errors", []).append(
                        {"error_type": "stale_response_id", "error": str(e), "model": self.model}
                    )
                self.previous_response_id = None
                return await self._generate_responses(messages, request_json, ret_metadata, meta_info)
            logger.error(f"Responses API error: {e}")
            raise
