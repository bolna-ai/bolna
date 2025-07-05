import asyncio
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI
import json, requests, time

from bolna.constants import CHECKING_THE_DOCUMENTS_FILLER, DEFAULT_LANGUAGE_CODE
from bolna.helpers.utils import convert_to_request_log, compute_function_pre_call_message
from .llm import BaseLLM
from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)
load_dotenv()
    

class OpenAiLLM(BaseLLM):
    def __init__(self, max_tokens=100, buffer_size=40, model="gpt-3.5-turbo-16k", temperature=0.1, language=DEFAULT_LANGUAGE_CODE, **kwargs):
        super().__init__(max_tokens, buffer_size, language, **kwargs)
        self.model = model

        logger.info(f"API Tools {self.custom_tools}")
        logger.info(f"Initializing OpenAI LLM with model: {self.model} and maxc tokens {max_tokens}")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.model_args = {"max_tokens": self.max_tokens, "temperature": self.temperature, "model": self.model}

        if kwargs.get("provider", "openai") == "custom":
            base_url = kwargs.get("base_url")
            api_key = kwargs.get('llm_key', None)
            self.async_client = AsyncOpenAI(base_url=base_url, api_key= api_key)
        else:
            llm_key = kwargs.get('llm_key', os.getenv('OPENAI_API_KEY'))
            self.async_client = AsyncOpenAI(api_key=llm_key)
            api_key = llm_key
        self.assistant_id = kwargs.get("assistant_id", None)
        if self.assistant_id:
            logger.info(f"Initializing OpenAI assistant with assistant id {self.assistant_id}")
            self.openai = OpenAI(api_key=api_key)
            self.model_args = {"max_completion_tokens": self.max_tokens, "temperature": self.temperature}
            my_assistant = self.openai.beta.assistants.retrieve(self.assistant_id)
            if my_assistant.tools is not None:
                self.tools = [i for i in my_assistant.tools if i.type == "function"]

    async def generate_stream(self, messages, synthesize=True, request_json=False, meta_info=None):
        if not messages or len(messages) == 0:
            raise Exception("No messages provided")
        
        response_format = self.get_response_format(request_json)
        model_args = {
            **self.model_args,
            "response_format": response_format,
            "messages": messages,
            "stream": True,
            "stop": ["User:"]
        }
        
        if meta_info and self.run_id:
            model_args["user"] = f"{self.run_id}#{meta_info['turn_id']}"

        if self.trigger_function_call:
            model_args["tools"] = json.loads(self.tools) if isinstance(self.tools, str) else self.tools
            model_args["tool_choice"] = "auto"
            model_args["parallel_tool_calls"] = False

        self.gave_out_prefunction_call_message = False

        answer, buffer = "", ""
        final_tool_calls_data = {}
        received_textual_response = False
        called_fun = None

        start_time = time.time()
        first_token_time = None
        latency_data = self.create_latency_data(meta_info)

        async for chunk in await self.async_client.chat.completions.create(**model_args):
            if not first_token_time:
                first_token_time = time.time()
                self.update_first_token_latency(latency_data, start_time)

            delta = chunk.choices[0].delta

            if hasattr(delta, 'tool_calls') and delta.tool_calls:
                if buffer:
                    yield buffer, True, latency_data, False, None, None
                    buffer = ""

                for tool_call in delta.tool_calls or []:
                    called_fun = self.accumulate_tool_call(final_tool_calls_data, tool_call)
                    logger.info(f"Function given by LLM to trigger is - {called_fun}")

                if self.should_send_prefunction_message(received_textual_response):
                    pre_msg, api_tool_pre_call_message = self.create_prefunction_message(called_fun)
                    yield pre_msg, True, latency_data, False, called_fun, api_tool_pre_call_message

            elif hasattr(delta, 'content') and delta.content is not None:
                received_textual_response = True
                answer += delta.content
                chunk_text, buffer, should_yield = self.handle_streaming_buffer(buffer, delta.content, synthesize, latency_data)
                if should_yield:
                    yield chunk_text, False, latency_data, False, None, None

        total_duration = time.time() - start_time
        if latency_data:
            latency_data["total_stream_duration_ms"] = round(total_duration * 1000)

        if self.trigger_function_call and final_tool_calls_data and final_tool_calls_data[0]["function"]["name"] in self.api_params:
            api_call_payload = self.build_api_call_payload(final_tool_calls_data, model_args, meta_info)
            if api_call_payload:
                logger.info(f"Payload to send {api_call_payload['parsed_args']} func_dict {self.api_params[called_fun]}")
                self.gave_out_prefunction_call_message = False
                self.validate_and_log_tool_call(api_call_payload, model_args)
                yield api_call_payload, False, latency_data, True, None, None

        if synthesize:
            yield buffer, True, latency_data, False, None, None
        else:
            yield answer, True, latency_data, False, None, None

        self.started_streaming = False
    
    async def generate(self, messages, request_json=False):
        response_format = self.get_response_format(request_json)

        completion = await self.async_client.chat.completions.create(model=self.model, temperature=0.0, messages=messages,
                                                                     stream=False, response_format=response_format)
        res = completion.choices[0].message.content
        return res

    def get_response_format(self, is_json_format: bool):
        if is_json_format and self.model in ('gpt-4-1106-preview', 'gpt-3.5-turbo-1106', 'gpt-4o-mini'):
            return {"type": "json_object"}
        else:
            return {"type": "text"}
