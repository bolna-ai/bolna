import os
import json
import httpx
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI, AuthenticationError, PermissionDeniedError, NotFoundError, RateLimitError, APIError, APIConnectionError, BadRequestError

from bolna.constants import DEFAULT_LANGUAGE_CODE
from bolna.helpers.utils import convert_to_request_log, compute_function_pre_call_message, now_ms
from .llm import BaseLLM
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
        final_tool_calls_data = {}
        received_textual_response = False
        called_fun = None

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

            if hasattr(delta, 'tool_calls') and delta.tool_calls:
                if buffer:
                    yield buffer, True, latency_data, False, None, None
                    buffer = ""

                for tool_call in delta.tool_calls or []:
                    idx = tool_call.index
                    if idx not in final_tool_calls_data:
                        called_fun = tool_call.function.name
                        final_tool_calls_data[idx] = {
                            "index": tool_call.index,
                            "id": tool_call.id,
                            "function": {
                                "name": called_fun,
                                "arguments": tool_call.function.arguments
                            },
                            "type": "function"
                        }
                    else:
                        final_tool_calls_data[idx]["function"]["arguments"] += tool_call.function.arguments

                if not self.gave_out_prefunction_call_message and not received_textual_response:
                    api_tool_pre_call_message = self.api_params[called_fun].get('pre_call_message', None)
                    detected_lang = meta_info.get('detected_language') if meta_info else None
                    active_language = detected_lang or self.language
                    pre_msg = compute_function_pre_call_message(active_language, called_fun, api_tool_pre_call_message)
                    yield pre_msg, True, latency_data, False, called_fun, api_tool_pre_call_message
                    self.gave_out_prefunction_call_message = True

            elif hasattr(delta, 'content') and delta.content is not None:
                received_textual_response = True
                answer += delta.content
                buffer += delta.content
                if synthesize and len(buffer) >= self.buffer_size:
                    split = buffer.rsplit(" ", 1)
                    yield split[0], False, latency_data, False, None, None
                    buffer = split[1] if len(split) > 1 else ""

        if latency_data:
            latency_data["total_stream_duration_ms"] = now_ms() - start_time

        if self.trigger_function_call and final_tool_calls_data and final_tool_calls_data[0]["function"]["name"] in self.api_params:
            i = [i for i in range(len(tools)) if called_fun == tools[i]["function"]["name"]][0]
            func_conf = self.api_params[called_fun]
            arguments_received = final_tool_calls_data[0]["function"]["arguments"]

            self.gave_out_prefunction_call_message = False

            api_call_payload = {
                "url": func_conf['url'],
                "method": None if func_conf['method'] is None else func_conf['method'].lower(),
                "param": func_conf['param'],
                "api_token": func_conf['api_token'],
                "headers": func_conf.get('headers', None),
                "model_args": model_args,
                "meta_info": meta_info,
                "called_fun": called_fun,
                "model_response": list(final_tool_calls_data.values()),
                "tool_call_id": final_tool_calls_data[0].get("id", ""),
                "textual_response": answer.strip() if received_textual_response else None
            }

            try:
                parsed_arguments = json.loads(arguments_received)
                all_required_keys = tools[i]["function"]["parameters"].get("required", [])

                if tools[i]["function"].get("parameters", None) is not None and all(key in parsed_arguments for key in all_required_keys):
                    convert_to_request_log(arguments_received, meta_info, self.model, "llm", direction="response", is_cached=False, run_id=self.run_id)
                    api_call_payload.update(parsed_arguments)
                else:
                    api_call_payload['resp'] = None
            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"Error parsing function arguments: {e}")
                api_call_payload['resp'] = None
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
