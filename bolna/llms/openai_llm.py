import os
import httpx
from urllib.parse import urlparse
from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI, AuthenticationError, PermissionDeniedError, NotFoundError, RateLimitError, APIError, APIConnectionError
import json

from bolna.constants import DEFAULT_LANGUAGE_CODE
from bolna.helpers.utils import convert_to_request_log, compute_function_pre_call_message, now_ms
from .llm import BaseLLM
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
        if model.startswith("gpt-5"):
            max_tokens_key = "max_completion_tokens"
        self.model_args = {max_tokens_key: self.max_tokens, "temperature": self.temperature, "model": self.model}

        if kwargs.get("service_tier") == "priority":
            self.model_args["service_tier"] = "priority"

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
        self.gave_out_prefunction_call_message = False

    async def generate_stream(self, messages, synthesize=True, request_json=False, meta_info=None):
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
        else:
            logger.info(f"{self.model} starts with gpt-5")
            model_args["reasoning_effort"] = "low"


        if self.trigger_function_call:
            model_args["tools"] = json.loads(self.tools) if isinstance(self.tools, str) else self.tools
            model_args["tool_choice"] = "auto"
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
                    "total_stream_duration_ms": None,  # Will be filled at end
                    "service_tier": service_tier,
                    "llm_host": self.llm_host
                }

            delta = chunk.choices[0].delta

            # Function call chunk
            if hasattr(delta, 'tool_calls') and delta.tool_calls:
                if buffer:
                    yield buffer, True, latency_data, False, None, None
                    buffer = ""

                # This for loop is going to cover the case of multiple tool calls. Currently, we are not allowing parallel
                # tool calls but if enabled in the future then this code should take care of accumulating the tool call data
                for tool_call in delta.tool_calls or []:
                    idx = tool_call.index
                    if idx not in final_tool_calls_data:
                        called_fun = tool_call.function.name
                        logger.info(f"Function given by LLM to trigger is - {called_fun}")
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

            # Normal text delta
            elif hasattr(delta, 'content') and delta.content is not None:
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

        # Post-processing for function call payload
        if self.trigger_function_call and final_tool_calls_data and final_tool_calls_data[0]["function"]["name"] in self.api_params:
            i = [i for i in range(len(tools)) if called_fun == tools[i]["function"]["name"]][0]
            func_conf = self.api_params[called_fun]
            arguments_received = final_tool_calls_data[0]["function"]["arguments"]

            logger.info(f"Payload to send {arguments_received} func_dict {func_conf}")
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
                "tool_call_id": final_tool_calls_data[0].get("id", "")
            }

            all_required_keys = tools[i]["function"]["parameters"]["properties"].keys() and tools[i]["function"]["parameters"].get(
                "required", [])
            if tools[i]["function"].get("parameters", None) is not None and (all(key in arguments_received for key in all_required_keys)):
                convert_to_request_log(arguments_received, meta_info, self.model, "llm", direction="response", is_cached=False,
                                       run_id=self.run_id)
                api_call_payload.update(json.loads(arguments_received))
            else:
                api_call_payload['resp'] = None
            yield api_call_payload, False, latency_data, True, None, None

        if synthesize:  # This is used only in streaming sense
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