import os
import json
import time
import logging
import aiohttp
import litellm
from litellm import acompletion, ContentPolicyViolationError
from litellm.exceptions import AuthenticationError, RateLimitError, APIError, APIConnectionError
from litellm.llms.custom_httpx.aiohttp_handler import BaseLLMAIOHTTPHandler
from dotenv import load_dotenv

from bolna.constants import DEFAULT_LANGUAGE_CODE
from bolna.helpers.utils import convert_to_request_log, compute_function_pre_call_message, now_ms
from .llm import BaseLLM
from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)
load_dotenv()

logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("LiteLLM Router").setLevel(logging.WARNING)
logging.getLogger("LiteLLM Proxy").setLevel(logging.WARNING)

# Global optimized aiohttp session for connection pooling
_global_aiohttp_session = None

def get_or_create_aiohttp_session():
    """Get or create a global aiohttp session with optimized connection pooling"""
    global _global_aiohttp_session
    if _global_aiohttp_session is None or _global_aiohttp_session.closed:
        # Create optimized session with connection pooling
        connector = aiohttp.TCPConnector(
            limit=100,  # Total connection pool size
            limit_per_host=30,  # Per-host connection limit
            ttl_dns_cache=300,  # DNS cache TTL
            keepalive_timeout=30,  # Keep connections alive
            force_close=False  # Reuse connections
        )
        _global_aiohttp_session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=180),
            connector=connector
        )
        # Configure LiteLLM to use our optimized session
        litellm.base_llm_aiohttp_handler = BaseLLMAIOHTTPHandler(client_session=_global_aiohttp_session)
        logger.info("Created optimized aiohttp session for LiteLLM with connection pooling")
    return _global_aiohttp_session

async def cleanup_aiohttp_session():
    """Cleanup the global aiohttp session"""
    global _global_aiohttp_session
    if _global_aiohttp_session and not _global_aiohttp_session.closed:
        await _global_aiohttp_session.close()
        logger.info("Closed global aiohttp session for LiteLLM")
        _global_aiohttp_session = None


class LiteLLM(BaseLLM):
    def __init__(self, model, max_tokens=30, buffer_size=40, temperature=0.0, language=DEFAULT_LANGUAGE_CODE, **kwargs):
        super().__init__(max_tokens, buffer_size)
        self.model = model
        self.started_streaming = False

        # Ensure the global session is initialized
        get_or_create_aiohttp_session()

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
            if "base_url" in kwargs:
                self.model_args["api_base"] = kwargs["base_url"]
            if "llm_key" in kwargs:
                self.model_args["api_key"] = kwargs["llm_key"]
            if "api_version" in kwargs:
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
        self.gave_out_prefunction_call_message = False

    async def generate_stream(self, messages, synthesize=True, meta_info=None):
        if not messages or len(messages) == 0:
            raise Exception("No messages provided")

        answer, buffer = "", ""
        final_tool_calls_data = {}
        received_textual_response = False
        first_token_time = None
        called_fun = None

        model_args = self.model_args.copy()
        model_args["messages"] = messages
        model_args["stream"] = True
        model_args["stop"] = ["User:"]

        if self.trigger_function_call:
            model_args["tools"] = json.loads(self.tools) if isinstance(self.tools, str) else self.tools
            model_args["tool_choice"] = "auto"
            model_args["parallel_tool_calls"] = False

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
                    "total_stream_duration_ms": None  # Will be filled at end
                }

            choice = chunk["choices"][0]
            delta = choice.get("delta", {})

            # Handle tool_calls
            if hasattr(delta, "tool_calls") and delta.tool_calls:
                if buffer:
                    yield buffer, True, latency_data, False, None, None
                    buffer = ""

                for tool_call in delta.tool_calls:
                    idx = tool_call.index
                    called_fun = tool_call.function.name

                    if idx not in final_tool_calls_data:
                        logger.info(f"Tool function triggered: {called_fun}")
                        final_tool_calls_data[idx] = {
                            "index": tool_call.index,
                            "id": tool_call.id,
                            "function": {
                                "name": called_fun,
                                "arguments": tool_call.function.arguments or ""
                            },
                            "type": "function"
                        }
                    else:
                        final_tool_calls_data[idx]["function"]["arguments"] += tool_call.function.arguments or ""

                # Pre-function call message (if any)
                if not self.gave_out_prefunction_call_message and not received_textual_response:
                    api_tool_pre_call_message = self.api_params.get(called_fun, {}).get("pre_call_message", None)
                    pre_msg = compute_function_pre_call_message(self.language, called_fun, api_tool_pre_call_message)
                    yield pre_msg, True, latency_data, False, called_fun, api_tool_pre_call_message
                    self.gave_out_prefunction_call_message = True

            # Normal streamed tokens
            elif hasattr(delta, "content") and delta.content:
                received_textual_response = True
                answer += delta.content
                buffer += delta.content

                if synthesize and len(buffer) >= self.buffer_size:
                    split = buffer.rsplit(" ", 1)
                    yield split[0], False, latency_data, False, None, None
                    buffer = split[1] if len(split) > 1 else ""

        # Set final duration
        if latency_data:
            latency_data["total_stream_duration_ms"] = now_ms() - start_time

        # Handle final function call logic
        if self.trigger_function_call and final_tool_calls_data:
            # Safely get the first tool call
            first_tool_call = final_tool_calls_data[0]["function"]
            func_name = first_tool_call["name"]
            args_str = first_tool_call["arguments"]

            tool_spec = next((t for t in model_args["tools"] if t["function"]["name"] == func_name), None)
            func_conf = self.api_params.get(func_name)

            if not func_conf:
                logger.warning(f"No API config found for tool: {func_name}")
                return

            try:
                parsed_args = json.loads(args_str)
            except json.JSONDecodeError:
                parsed_args = args_str

            logger.info(f"Tool payload: {parsed_args} | Config: {func_conf}")
            api_call_payload = {
                "url": func_conf.get("url"),
                "method": (func_conf.get("method") or "").lower(),
                "param": func_conf.get("param"),
                "api_token": func_conf.get("api_token"),
                "headers": func_conf.get('headers', None),
                "model_args": model_args,
                "meta_info": meta_info,
                "called_fun": func_name,
                "model_response": list(final_tool_calls_data.values()),
                "tool_call_id": final_tool_calls_data[0].get("id")
            }

            # Merge function arguments into payload if all required keys exist
            if tool_spec:
                required_keys = tool_spec["function"].get("parameters", {}).get("required", [])
                if all(k in parsed_args for k in required_keys):
                    convert_to_request_log(parsed_args, meta_info, self.model, "llm", direction="response",
                                           is_cached=False, run_id=self.run_id)
                    api_call_payload.update(parsed_args)
                else:
                    api_call_payload["resp"] = None

            yield api_call_payload, False, latency_data, True, None, None

        # Final buffer flush
        if synthesize and buffer.strip():
            yield buffer, True, latency_data, False, None, None
        elif not synthesize:
            yield answer, True, latency_data, False, None, None

        self.started_streaming = False

    async def generate(self, messages, stream=False, request_json=False, meta_info = None):
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
        return text
