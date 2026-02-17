import os
import httpx
from urllib.parse import urlparse
from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI, AuthenticationError, PermissionDeniedError, NotFoundError, RateLimitError, APIError, APIConnectionError
import json

from bolna.constants import DEFAULT_LANGUAGE_CODE
from bolna.helpers.utils import convert_to_request_log, compute_function_pre_call_message, now_ms
from .llm import BaseLLM
from .tool_call_accumulator import ToolCallAccumulator
from bolna.enums import ReasoningEffort, Verbosity
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
        if model.startswith("gpt-5"):
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

    async def generate_stream(self, messages, synthesize=True, request_json=False, meta_info=None, tool_choice=None):
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

        if not self.model.startswith("gpt-5"):
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
                    latency_data["service_tier"] = service_tier

            if not first_token_time:
                first_token_time = now
                self.started_streaming = True

                latency_data = {
                    "sequence_id": meta_info.get("sequence_id"),
                    "first_token_latency_ms": first_token_time - start_time,
                    "total_stream_duration_ms": None,
                    "service_tier": service_tier,
                    "llm_host": self.llm_host
                }

            delta = chunk.choices[0].delta

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

    def get_response_format(self, is_json_format: bool):
        if is_json_format and self.model in ('gpt-4-1106-preview', 'gpt-3.5-turbo-1106', 'gpt-4o-mini', 'gpt-4.1-mini'):
            return {"type": "json_object"}
        else:
            return {"type": "text"}