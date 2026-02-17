import os
import json
import time
import logging
from litellm import acompletion, ContentPolicyViolationError
from litellm.exceptions import AuthenticationError, RateLimitError, APIError, APIConnectionError
from dotenv import load_dotenv

from bolna.constants import DEFAULT_LANGUAGE_CODE
from bolna.helpers.utils import convert_to_request_log, compute_function_pre_call_message, now_ms
from .llm import BaseLLM
from .tool_call_accumulator import ToolCallAccumulator
from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)
load_dotenv()

logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("LiteLLM Router").setLevel(logging.WARNING)
logging.getLogger("LiteLLM Proxy").setLevel(logging.WARNING)


class LiteLLM(BaseLLM):
    def __init__(self, model, max_tokens=30, buffer_size=40, temperature=0.0, language=DEFAULT_LANGUAGE_CODE, **kwargs):
        super().__init__(max_tokens, buffer_size)
        self.model = model
        self.started_streaming = False

        self.language = language
        self.model_args = {"max_tokens": max_tokens, "temperature": temperature, "model": self.model}
        self.api_key = kwargs.get("llm_key", os.getenv('LITELLM_MODEL_API_KEY'))
        self.api_base = kwargs.get("base_url", os.getenv('LITELLM_MODEL_API_BASE'))
        self.api_version = kwargs.get("api_version", os.getenv('LITELLM_MODEL_API_VERSION'))
        if self.api_key:
            self.model_args["api_key"] = self.api_key
        if self.api_base:
            self.model_args["api_base"] = self.api_base
        if self.api_version:
            self.model_args["api_version"] = self.api_version

        if len(kwargs) != 0:
            if kwargs.get("base_url", None):
                self.model_args["api_base"] = kwargs["base_url"]
            if kwargs.get("llm_key", None):
                self.model_args["api_key"] = kwargs["llm_key"]
            if kwargs.get("api_version", None):
                self.model_args["api_version"] = kwargs["api_version"]

        self.custom_tools = kwargs.get("api_tools", None)
        logger.info(f"API Tools {self.custom_tools}")
        if self.custom_tools is not None:
            self.trigger_function_call = True
            self.api_params = self.custom_tools['tools_params']
            logger.info(f"Function dict {self.api_params}")
            self.tools = self.custom_tools['tools']
        else:
            self.trigger_function_call = False
        self.run_id = kwargs.get("run_id", None)

    async def generate_stream(self, messages, synthesize=True, meta_info=None, tool_choice=None):
        if not messages or len(messages) == 0:
            raise Exception("No messages provided")

        answer, buffer = "", ""
        first_token_time = None

        model_args = self.model_args.copy()
        model_args["messages"] = messages
        model_args["stream"] = True
        model_args["stop"] = ["User:"]

        if self.trigger_function_call:
            model_args["tools"] = json.loads(self.tools) if isinstance(self.tools, str) else self.tools
            model_args["tool_choice"] = tool_choice or "auto"
            model_args["parallel_tool_calls"] = False

        tools = model_args.get("tools", [])
        accumulator = None
        if self.trigger_function_call:
            accumulator = ToolCallAccumulator(self.api_params, tools, self.language, self.model, self.run_id)

        start_time = now_ms()
        latency_data = {
            "sequence_id": meta_info.get("sequence_id") if meta_info else None,
            "first_token_latency_ms": None,
            "total_stream_duration_ms": None,
        }

        try:
            completion_stream = await acompletion(**model_args)
        except ContentPolicyViolationError as e:
            error_message = str(e)
            logger.error(f'Content policy violation in stream: {error_message}')
            if meta_info and self.run_id:
                convert_to_request_log(
                    f"Content Policy Violation: {error_message}",
                    meta_info, self.model, component="llm",
                    direction="error", is_cached=False, run_id=self.run_id
                )
            return
        except AuthenticationError as e:
            logger.error(f"LiteLLM authentication failed: Invalid or expired API key - {e}")
            raise
        except RateLimitError as e:
            logger.error(f"LiteLLM rate limit exceeded: {e}")
            raise
        except APIConnectionError as e:
            logger.error(f"LiteLLM connection error: {e}")
            raise
        except APIError as e:
            logger.error(f"LiteLLM API error: {e}")
            raise
        except Exception as e:
            logger.error(f"LiteLLM unexpected error: {e}")
            raise

        async for chunk in completion_stream:
            now = now_ms()
            if not first_token_time:
                first_token_time = now
                self.started_streaming = True
                latency_data = {
                    "sequence_id": meta_info.get("sequence_id"),
                    "first_token_latency_ms": first_token_time - start_time,
                    "total_stream_duration_ms": None
                }

            choice = chunk["choices"][0]
            delta = choice.get("delta", {})

            if hasattr(delta, "tool_calls") and delta.tool_calls and accumulator:
                if buffer:
                    yield buffer, True, latency_data, False, None, None
                    buffer = ""

                accumulator.process_delta(delta.tool_calls)

                pre_call = accumulator.get_pre_call_message(meta_info)
                if pre_call:
                    yield pre_call[0], True, latency_data, False, pre_call[1], pre_call[2]

            elif hasattr(delta, "content") and delta.content:
                if accumulator:
                    accumulator.received_textual = True
                answer += delta.content
                buffer += delta.content
                if synthesize and len(buffer) >= self.buffer_size:
                    split = buffer.rsplit(" ", 1)
                    yield split[0], False, latency_data, False, None, None
                    buffer = split[1] if len(split) > 1 else ""

        if latency_data:
            latency_data["total_stream_duration_ms"] = now_ms() - start_time

        if accumulator and accumulator.final_tool_calls:
            api_call_payload = accumulator.build_api_payload(model_args, meta_info, answer)
            if api_call_payload:
                yield api_call_payload, False, latency_data, True, None, None

        if synthesize and buffer.strip():
            yield buffer, True, latency_data, False, None, None
        elif not synthesize:
            yield answer, True, latency_data, False, None, None

        self.started_streaming = False

    async def generate(self, messages, stream=False, request_json=False, meta_info = None, ret_metadata=False):
        text = ""
        model_args = self.model_args.copy()
        model_args["model"] = self.model
        model_args["messages"] = messages
        model_args["stream"] = stream

        if request_json:
            model_args['response_format'] = {
                "type": "json_object"
            }
        logger.info(f'Request to litellm {model_args}')
        try:
            completion = await acompletion(**model_args)
            text = completion.choices[0].message.content
        except ContentPolicyViolationError as e:
            error_message = str(e)
            logger.error(f'Content policy violation: {error_message}')

            # Log to CSV trace
            if meta_info and self.run_id:
                convert_to_request_log(
                    f"Content Policy Violation: {error_message}",
                    meta_info,
                    self.model,
                    component="llm",
                    direction="error",
                    is_cached=False,
                    run_id=self.run_id
                )
            # Don't re-raise - allow graceful degradation for content policy violations
        except AuthenticationError as e:
            logger.error(f"LiteLLM authentication failed: Invalid or expired API key - {e}")
            raise
        except RateLimitError as e:
            logger.error(f"LiteLLM rate limit exceeded: {e}")
            raise
        except APIConnectionError as e:
            logger.error(f"LiteLLM connection error: {e}")
            raise
        except APIError as e:
            logger.error(f"LiteLLM API error: {e}")
            raise
        except Exception as e:
            error_message = str(e)
            logger.error(f'LiteLLM unexpected error generating response: {error_message}')
            raise
        if ret_metadata:
            return text, {}
        else:
            return text
