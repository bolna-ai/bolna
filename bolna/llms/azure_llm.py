import os
import json
import httpx
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI, AuthenticationError, PermissionDeniedError, NotFoundError, RateLimitError, APIError, APIConnectionError, BadRequestError

from bolna.constants import DEFAULT_LANGUAGE_CODE
from bolna.helpers.utils import convert_to_request_log, compute_function_pre_call_message, now_ms
from .llm import BaseLLM
from .tool_call_accumulator import ToolCallAccumulator
from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)
load_dotenv()


class AzureLLM(BaseLLM):
    def __init__(self, max_tokens=100, buffer_size=40, model="gpt-4.1-mini", temperature=0.1, language=DEFAULT_LANGUAGE_CODE, **kwargs):
        super().__init__(max_tokens, buffer_size)

        if model.startswith("azure/"):
            self.model = model.replace("azure/", "", 1)
        else:
            self.model = model

        self.custom_tools = kwargs.get("api_tools", None)
        self.language = language
        if self.custom_tools is not None:
            self.trigger_function_call = True
            self.api_params = self.custom_tools['tools_params']
            self.tools = self.custom_tools['tools']
        else:
            self.trigger_function_call = False

        self.started_streaming = False
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.model_args = {"max_tokens": self.max_tokens, "temperature": self.temperature, "model": self.model}

        azure_endpoint = kwargs.get("base_url", os.getenv('AZURE_OPENAI_ENDPOINT'))
        api_key = kwargs.get('llm_key', os.getenv('AZURE_OPENAI_API_KEY'))
        api_version = kwargs.get("api_version", os.getenv('AZURE_OPENAI_API_VERSION', '2024-12-01-preview'))

        limits = httpx.Limits(
            max_connections=50,
            max_keepalive_connections=50,
            keepalive_expiry=30
        )
        http_client = httpx.AsyncClient(limits=limits, timeout=httpx.Timeout(600.0, connect=10.0))

        self.async_client = AsyncAzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version,
            http_client=http_client
        )

        self.run_id = kwargs.get("run_id", None)
        self.gave_out_prefunction_call_message = False

    async def generate_stream(self, messages, synthesize=True, request_json=False, meta_info=None, tool_choice=None):
        if not messages or len(messages) == 0:
            raise Exception("No messages provided")

        response_format = self.get_response_format(request_json)
        model_args = {
            **self.model_args,
            "response_format": response_format,
            "messages": messages,
            "stream": True,
            "stop": ["User:"],
            "user": f"{self.run_id}#{meta_info.get('turn_id', '')}" if meta_info else self.run_id
        }

        if self.trigger_function_call:
            model_args["tools"] = json.loads(self.tools) if isinstance(self.tools, str) else self.tools
            model_args["tool_choice"] = tool_choice or "auto"
            model_args["parallel_tool_calls"] = False

        self.gave_out_prefunction_call_message = False

        answer, buffer = "", ""
        tools = model_args.get("tools", [])
        accumulator = None
        if self.trigger_function_call:
            accumulator = ToolCallAccumulator(self.api_params, tools, self.language, self.model, self.run_id)

        start_time = now_ms()
        first_token_time = None
        latency_data = None

        try:
            completion_stream = await self.async_client.chat.completions.create(**model_args)
        except BadRequestError as e:
            logger.error(f"Azure OpenAI bad request: {e}")
            raise
        except AuthenticationError as e:
            logger.error(f"Azure OpenAI authentication failed: {e}")
            raise
        except PermissionDeniedError as e:
            logger.error(f"Azure OpenAI permission denied: {e}")
            raise
        except NotFoundError as e:
            logger.error(f"Azure OpenAI resource not found: {e}")
            raise
        except RateLimitError as e:
            logger.error(f"Azure OpenAI rate limit exceeded: {e}")
            raise
        except APIConnectionError as e:
            logger.error(f"Azure OpenAI connection error: {e}")
            raise
        except APIError as e:
            logger.error(f"Azure OpenAI API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Azure OpenAI unexpected error: {e}")
            raise

        async for chunk in completion_stream:
            if not chunk.choices or len(chunk.choices) == 0:
                continue

            choice = chunk.choices[0]
            now = now_ms()
            if not first_token_time:
                first_token_time = now
                self.started_streaming = True
                latency_data = {
                    "sequence_id": meta_info.get("sequence_id"),
                    "first_token_latency_ms": first_token_time - start_time,
                    "total_stream_duration_ms": None
                }

            delta = choice.delta

            if hasattr(delta, 'tool_calls') and delta.tool_calls and accumulator:
                if buffer:
                    yield buffer, True, latency_data, False, None, None
                    buffer = ""

                accumulator.process_delta(delta.tool_calls)

                pre_call = accumulator.get_pre_call_message(meta_info)
                if pre_call:
                    yield pre_call[0], True, latency_data, False, pre_call[1], pre_call[2]

            elif hasattr(delta, 'content') and delta.content is not None:
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

        if synthesize:
            yield buffer, True, latency_data, False, None, None
        else:
            yield answer, True, latency_data, False, None, None

        self.started_streaming = False

    async def generate(self, messages, request_json=False, ret_metadata=False):
        response_format = self.get_response_format(request_json)

        try:
            completion = await self.async_client.chat.completions.create(
                model=self.model,
                temperature=0.0,
                messages=messages,
                stream=False,
                response_format=response_format
            )

            res = completion.choices[0].message.content
            if ret_metadata:  
                return res, {}
            else:
                return res
        except BadRequestError as e:
            logger.error(f"Azure OpenAI bad request: {e}")
            raise
        except AuthenticationError as e:
            logger.error(f"Azure OpenAI authentication failed: {e}")
            raise
        except PermissionDeniedError as e:
            logger.error(f"Azure OpenAI permission denied: {e}")
            raise
        except NotFoundError as e:
            logger.error(f"Azure OpenAI resource not found: {e}")
            raise
        except RateLimitError as e:
            logger.error(f"Azure OpenAI rate limit exceeded: {e}")
            raise
        except APIConnectionError as e:
            logger.error(f"Azure OpenAI connection error: {e}")
            raise
        except APIError as e:
            logger.error(f"Azure OpenAI API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Azure OpenAI unexpected error: {e}")
            raise

    def get_response_format(self, is_json_format: bool):
        if is_json_format and self.model in ('gpt-4-1106-preview', 'gpt-3.5-turbo-1106', 'gpt-4o-mini'):
            return {"type": "json_object"}
        else:
            return {"type": "text"}
