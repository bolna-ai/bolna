import os
import asyncio
import json
import time
from typing import Optional
from urllib.parse import urlparse
from bolna.llms.http_client_pool import get_shared_http_client
from dotenv import load_dotenv
from openai import (
    AsyncOpenAI,
    OpenAI,
    AuthenticationError,
    PermissionDeniedError,
    NotFoundError,
    RateLimitError,
    APIError,
    APIConnectionError,
)

import websockets
from websockets.protocol import State as WSState

from bolna.constants import DEFAULT_LANGUAGE_CODE, GPT5_MODEL_PREFIX
from bolna.enums import ReasoningEffort, ResponseStreamEvent, ResponseItemType, Verbosity
from bolna.helpers.utils import compute_function_pre_call_message, now_ms
from .openai_base import OpenAICompatibleLLM
from .tool_call_accumulator import ToolCallAccumulator
from .types import APIParams, LLMStreamChunk, LatencyData
from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)
load_dotenv()


class OpenAIWSConnection:
    """Persistent WebSocket connection to wss://api.openai.com/v1/responses.

    - Eager connect at init, ready before first LLM call
    - Auto-reconnect on connection drop or before 60-min server limit
    - Sequential: one in-flight response at a time (API constraint)
    """

    WS_URL = "wss://api.openai.com/v1/responses"
    RECONNECT_BEFORE_SECS = 55 * 60
    TERMINAL_EVENTS = ResponseStreamEvent.terminal_events()

    def __init__(self, api_key: str):
        self._api_key = api_key
        self._ws = None
        self._connected_at: float = 0
        self._lock = asyncio.Lock()
        self._connect_task: Optional[asyncio.Task] = None

    def start_connect(self):
        """Kick off WS connection eagerly. Must be called from a running event loop."""
        self._connect_task = asyncio.create_task(self._do_connect())

    async def _do_connect(self):
        """Background connect with error handling — exceptions surface in ensure_connected."""
        try:
            await self._connect()
        except Exception as e:
            logger.warning(f"Eager WS connect failed, will retry on first use: {e}")

    async def ensure_connected(self):
        if self._connect_task and not self._connect_task.done():
            await self._connect_task
            self._connect_task = None

        if self._ws is not None:
            if self._ws.state is not WSState.OPEN:
                logger.info("WebSocket closed unexpectedly, reconnecting")
                await self._close_ws()
            elif time.monotonic() - self._connected_at >= self.RECONNECT_BEFORE_SECS:
                logger.info("WebSocket approaching 60-min limit, reconnecting")
                await self._close_ws()
            else:
                return

        await self._connect()

    async def _connect(self):
        self._ws = await websockets.connect(
            self.WS_URL,
            additional_headers={"Authorization": f"Bearer {self._api_key}"},
            max_size=None,
            close_timeout=5,
        )
        self._connected_at = time.monotonic()
        logger.info("WebSocket connected to OpenAI Responses API")

    async def stream_response(self, create_params: dict):
        """Send response.create and yield raw event dicts until terminal event."""
        async with self._lock:
            await self.ensure_connected()
            await self._ws.send(json.dumps({"type": ResponseStreamEvent.CREATE, **create_params}))

            async for raw_msg in self._ws:
                evt = json.loads(raw_msg)
                evt_type = evt.get("type", "")
                yield evt
                if evt_type in self.TERMINAL_EVENTS:
                    return

    async def cancel_response(self, response_id: str):
        """Cancel an in-flight response. Best-effort, errors are non-critical."""
        if self._ws is None:
            return
        try:
            await self._ws.send(
                json.dumps(
                    {
                        "type": ResponseStreamEvent.CANCEL,
                        "response_id": response_id,
                    }
                )
            )
            async for raw_msg in self._ws:
                if json.loads(raw_msg).get("type") in self.TERMINAL_EVENTS:
                    break
        except Exception:
            pass

    async def disconnect(self):
        await self._close_ws()

    async def _close_ws(self):
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None


class OpenAiLLM(OpenAICompatibleLLM):
    def __init__(
        self,
        max_tokens=100,
        buffer_size=40,
        model="gpt-3.5-turbo-16k",
        temperature=0.1,
        language=DEFAULT_LANGUAGE_CODE,
        **kwargs,
    ):
        super().__init__(max_tokens, buffer_size)
        self.model = model

        self.custom_tools = kwargs.get("api_tools", None)
        self.language = language
        logger.info(f"API Tools {self.custom_tools}")
        if self.custom_tools is not None:
            self.trigger_function_call = True
            self.api_params = self.custom_tools["tools_params"]
            logger.info(f"Function dict {self.api_params}")
            self.tools = self.custom_tools["tools"]
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
            self.model_args["reasoning_effort"] = kwargs.get("reasoning_effort", None) or ReasoningEffort.MINIMAL.value
            self.model_args["verbosity"] = kwargs.get("verbosity", None) or Verbosity.LOW.value

        self.model_args.update({max_tokens_key: self.max_tokens, "temperature": self.temperature, "model": self.model})

        self.model_args["service_tier"] = kwargs.get("service_tier", "default")

        http_client = get_shared_http_client(base_url=kwargs.get("base_url"), http2=True)

        if kwargs.get("provider", "openai") == "custom":
            base_url = kwargs.get("base_url")
            api_key = kwargs.get("llm_key", None)
            self.async_client = AsyncOpenAI(base_url=base_url, api_key=api_key, http_client=http_client)
        else:
            llm_key = kwargs.get("llm_key", os.getenv("OPENAI_API_KEY"))
            base_url = kwargs.get("base_url")
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
            # self.thread_id = self.openai.beta.threads.create().id
            self.model_args = {"max_completion_tokens": self.max_tokens, "temperature": self.temperature}
            my_assistant = self.openai.beta.assistants.retrieve(self.assistant_id)
            if my_assistant.tools is not None:
                self.tools = [i for i in my_assistant.tools if i.type == "function"]
            # logger.info(f'thread id : {self.thread_id}')
        self.run_id = kwargs.get("run_id", None)

        self._init_responses_api(
            kwargs.get("use_responses_api", False), compact_threshold=kwargs.get("compact_threshold")
        )

        self._ws_transport = None
        if self.use_responses_api and kwargs.get("provider", "openai") != "custom" and not base_url:
            self._ws_transport = OpenAIWSConnection(api_key=api_key)
            self._ws_transport.start_connect()

    async def generate_stream(self, messages, synthesize=True, request_json=False, meta_info=None, tool_choice=None):
        if self.use_responses_api:
            if self._ws_transport:
                async for chunk in self._generate_stream_ws_responses(
                    messages, synthesize, request_json, meta_info, tool_choice
                ):
                    yield chunk
            else:
                async for chunk in self._generate_stream_responses(
                    messages, synthesize, request_json, meta_info, tool_choice
                ):
                    yield chunk
        else:
            async for chunk in self._generate_stream_chat(messages, synthesize, request_json, meta_info, tool_choice):
                yield chunk

    async def _generate_stream_chat(
        self, messages, synthesize=True, request_json=False, meta_info=None, tool_choice=None
    ):
        if not messages or len(messages) == 0:
            raise Exception("No messages provided")

        response_format = self.get_response_format(request_json)
        model_args = {
            **self.model_args,
            "response_format": response_format,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            "user": f"{self.run_id}#{meta_info['turn_id']}",
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
        stream_usage = None

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
            if hasattr(chunk, "service_tier") and chunk.service_tier:
                service_tier = chunk.service_tier
                if latency_data:
                    latency_data.service_tier = service_tier

            # Final usage-only chunk (choices=[]) from stream_options
            if hasattr(chunk, "usage") and chunk.usage and (not chunk.choices or len(chunk.choices) == 0):
                stream_usage = chunk.usage
                continue

            if not first_token_time:
                first_token_time = now
                self.started_streaming = True

                latency_data = LatencyData(
                    sequence_id=meta_info.get("sequence_id") if meta_info else None,
                    first_token_latency_ms=first_token_time - start_time,
                    total_stream_duration_ms=None,
                    service_tier=service_tier,
                    llm_host=self.llm_host,
                )

            delta = chunk.choices[0].delta

            if hasattr(delta, "tool_calls") and delta.tool_calls and accumulator:
                if buffer:
                    yield LLMStreamChunk(data=buffer, end_of_stream=True, latency=latency_data)
                    buffer = ""

                accumulator.process_delta(delta.tool_calls)

                pre_call = accumulator.get_pre_call_message(meta_info)
                if pre_call:
                    yield LLMStreamChunk(
                        data=pre_call[0],
                        end_of_stream=True,
                        latency=latency_data,
                        function_name=pre_call[1],
                        function_message=pre_call[2],
                    )

            elif hasattr(delta, "content") and delta.content is not None:
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
                yield LLMStreamChunk(
                    data=api_call_payload, end_of_stream=False, latency=latency_data, is_function_call=True
                )

        usage_kwargs = {}
        if stream_usage:
            usage_kwargs["input_tokens"] = getattr(stream_usage, "prompt_tokens", None)
            usage_kwargs["output_tokens"] = getattr(stream_usage, "completion_tokens", None)
            details = getattr(stream_usage, "completion_tokens_details", None)
            if details:
                usage_kwargs["reasoning_tokens"] = getattr(details, "reasoning_tokens", None)
            prompt_details = getattr(stream_usage, "prompt_tokens_details", None)
            if prompt_details:
                usage_kwargs["cached_tokens"] = getattr(prompt_details, "cached_tokens", None)

        if synthesize:
            yield LLMStreamChunk(data=buffer, end_of_stream=True, latency=latency_data, **usage_kwargs)
        else:
            yield LLMStreamChunk(data=answer, end_of_stream=True, latency=latency_data, **usage_kwargs)

        self.started_streaming = False

    async def generate(self, messages, request_json=False, ret_metadata=False):
        if self.use_responses_api:
            return await self._generate_responses(messages, request_json, ret_metadata)
        return await self._generate_chat(messages, request_json, ret_metadata)

    async def _generate_chat(self, messages, request_json=False, ret_metadata=False):
        response_format = self.get_response_format(request_json)

        try:
            completion = await self.async_client.chat.completions.create(
                model=self.model, temperature=0.0, messages=messages, stream=False, response_format=response_format
            )
            res = completion.choices[0].message.content
            if ret_metadata:
                metadata = {
                    "llm_host": self.llm_host,
                    "service_tier": completion.service_tier if hasattr(completion, "service_tier") else None,
                }
                if completion.usage:
                    metadata["input_tokens"] = completion.usage.prompt_tokens
                    metadata["output_tokens"] = completion.usage.completion_tokens
                    details = getattr(completion.usage, "completion_tokens_details", None)
                    if details:
                        metadata["reasoning_tokens"] = getattr(details, "reasoning_tokens", None)
                    prompt_details = getattr(completion.usage, "prompt_tokens_details", None)
                    if prompt_details:
                        metadata["cached_tokens"] = getattr(prompt_details, "cached_tokens", None)
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

    def get_response_format(self, is_json_format: bool):
        if is_json_format and self.model in ("gpt-4-1106-preview", "gpt-3.5-turbo-1106", "gpt-4o-mini", "gpt-4.1-mini"):
            return {"type": "json_object"}
        else:
            return {"type": "text"}

    async def _generate_stream_ws_responses(
        self, messages, synthesize=True, request_json=False, meta_info=None, tool_choice=None
    ):
        """Stream via persistent WebSocket — same interface as _generate_stream_responses."""
        if not messages:
            raise ValueError("No messages provided")

        # store=False: WS previous_response_id uses connection-local cache, not server storage
        create_params, responses_tools = self._build_responses_create_kwargs(
            messages, meta_info, request_json, tool_choice, store=False
        )

        # WS endpoint silently closes on float temperature — coerce to int
        temp = create_params.get("temperature")
        if temp is not None:
            create_params["temperature"] = int(round(temp))

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
        ws_service_tier = None
        llm_host = self.llm_host
        response_usage = None

        try:
            async for evt in self._ws_transport.stream_response(create_params):
                now = now_ms()
                evt_type = evt.get("type", "")

                if evt_type == ResponseStreamEvent.ERROR:
                    error_info = evt.get("error", {})
                    error_code = error_info.get("code", "")
                    if error_code == "previous_response_not_found" and self.previous_response_id:
                        logger.warning(f"WS previous_response_id not found, retrying with full history")
                        self.previous_response_id = None
                        async for chunk in self._generate_stream_ws_responses(
                            messages, synthesize, request_json, meta_info, tool_choice
                        ):
                            yield chunk
                        return
                    logger.error(f"WebSocket Responses API error: {error_info}")
                    raise APIError(message=f"WS response error: {error_info}", request=None, body=None)

                if evt_type == ResponseStreamEvent.CREATED:
                    resp = evt.get("response", {})
                    self.previous_response_id = resp.get("id")
                    ws_service_tier = resp.get("service_tier")
                    continue

                if evt_type == ResponseStreamEvent.FAILED:
                    resp = evt.get("response", {})
                    error_info = resp.get("error") or resp.get("last_error")
                    logger.error(f"WS Responses API stream failed: {error_info}")
                    self.invalidate_response_chain()
                    raise APIError(message=f"Response failed: {error_info}", request=None, body=None)

                if evt_type == ResponseStreamEvent.INCOMPLETE:
                    logger.warning("WS Responses API stream incomplete")
                    self.invalidate_response_chain()
                    break

                if not first_token_time and evt_type in (
                    ResponseStreamEvent.OUTPUT_TEXT_DELTA,
                    ResponseStreamEvent.FUNCTION_CALL_ARGS_DELTA,
                ):
                    first_token_time = now
                    self.started_streaming = True
                    latency_data = LatencyData(
                        sequence_id=meta_info.get("sequence_id") if meta_info else None,
                        first_token_latency_ms=first_token_time - start_time,
                        total_stream_duration_ms=None,
                        service_tier=ws_service_tier,
                        llm_host=llm_host,
                    )

                if evt_type == ResponseStreamEvent.OUTPUT_TEXT_DELTA:
                    received_textual = True
                    delta = evt.get("delta", "")
                    answer += delta
                    buffer += delta
                    if synthesize and len(buffer) >= self.buffer_size:
                        split = buffer.rsplit(" ", 1)
                        yield LLMStreamChunk(data=split[0], end_of_stream=False, latency=latency_data)
                        buffer = split[1] if len(split) > 1 else ""

                elif evt_type == ResponseStreamEvent.REASONING_SUMMARY_TEXT_DELTA:
                    reasoning_summary_parts.append(evt.get("delta", ""))

                elif evt_type == ResponseStreamEvent.OUTPUT_ITEM_ADDED:
                    item = evt.get("item", {})
                    if item.get("type") == ResponseItemType.FUNCTION_CALL:
                        if buffer:
                            yield LLMStreamChunk(data=buffer, end_of_stream=True, latency=latency_data)
                            buffer = ""
                        item_id = item.get("id", "")
                        func_call_args[item_id] = ""
                        func_call_names[item_id] = item.get("name", "")
                        func_call_ids[item_id] = item.get("call_id", "")

                        if not gave_pre_call_msg and not received_textual and self.trigger_function_call:
                            gave_pre_call_msg = True
                            func_name = item.get("name", "")
                            func_params = self.api_params.get(func_name)
                            api_tool_pre_call_message = (
                                APIParams.model_validate(func_params).pre_call_message if func_params else None
                            )
                            detected_lang = meta_info.get("detected_language") if meta_info else None
                            active_language = detected_lang or self.language
                            pre_msg = compute_function_pre_call_message(
                                active_language, func_name, api_tool_pre_call_message
                            )
                            if pre_msg:
                                yield LLMStreamChunk(
                                    data=pre_msg,
                                    end_of_stream=True,
                                    latency=latency_data,
                                    function_name=func_name,
                                    function_message=api_tool_pre_call_message,
                                )

                elif evt_type == ResponseStreamEvent.FUNCTION_CALL_ARGS_DELTA:
                    item_id = evt.get("item_id", "")
                    func_call_args[item_id] = func_call_args.get(item_id, "") + evt.get("delta", "")

                elif evt_type == ResponseStreamEvent.COMPLETED:
                    resp = evt.get("response", {})
                    self.previous_response_id = resp.get("id", self.previous_response_id)
                    self._pending_call_ids = set(func_call_ids.values())
                    ws_service_tier = ws_service_tier or resp.get("service_tier")
                    response_usage = resp.get("usage")
                    break

        except APIError:
            raise
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"WS streaming error: {e}, falling back to HTTP SSE")
            self.invalidate_response_chain()
            async for chunk in self._generate_stream_responses(
                messages, synthesize, request_json, meta_info, tool_choice
            ):
                yield chunk
            return

        if latency_data:
            latency_data.total_stream_duration_ms = now_ms() - start_time
            if ws_service_tier:
                latency_data.service_tier = ws_service_tier

        fc_chunk = self._build_function_call_chunk(
            func_call_args,
            func_call_names,
            func_call_ids,
            responses_tools,
            create_params,
            meta_info,
            answer,
            received_textual,
            latency_data,
        )
        if fc_chunk:
            yield fc_chunk

        usage_kwargs = {}
        if response_usage and isinstance(response_usage, dict):
            usage_kwargs["input_tokens"] = response_usage.get("input_tokens")
            usage_kwargs["output_tokens"] = response_usage.get("output_tokens")
            output_details = response_usage.get("output_tokens_details", {}) or {}
            if output_details.get("reasoning_tokens"):
                usage_kwargs["reasoning_tokens"] = output_details["reasoning_tokens"]
            input_details = response_usage.get("input_tokens_details", {}) or {}
            if input_details.get("cached_tokens"):
                usage_kwargs["cached_tokens"] = input_details["cached_tokens"]

        reasoning_content = "".join(reasoning_summary_parts) if reasoning_summary_parts else None
        if reasoning_content:
            usage_kwargs["reasoning_content"] = reasoning_content

        if synthesize:
            yield LLMStreamChunk(data=buffer, end_of_stream=True, latency=latency_data, **usage_kwargs)
        else:
            yield LLMStreamChunk(data=answer, end_of_stream=True, latency=latency_data, **usage_kwargs)

        self.started_streaming = False

    def invalidate_response_chain(self):
        response_id = self.previous_response_id
        super().invalidate_response_chain()
        if self._ws_transport and response_id:
            asyncio.ensure_future(self._ws_transport.cancel_response(response_id))

    async def close(self):
        if self._ws_transport:
            await self._ws_transport.disconnect()
