import os
import json
import time
from litellm import acompletion
from dotenv import load_dotenv

from bolna.constants import DEFAULT_LANGUAGE_CODE
from bolna.helpers.utils import json_to_pydantic_schema, convert_to_request_log, compute_function_pre_call_message
from .llm import BaseLLM
from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)
load_dotenv()


class LiteLLM(BaseLLM):
    def __init__(self, model, max_tokens=30, buffer_size=40, temperature=0.0, language=DEFAULT_LANGUAGE_CODE, **kwargs):
        super().__init__(max_tokens, buffer_size, language, **kwargs)
        self.model = model

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

        logger.info(f"API Tools {self.custom_tools}")

    async def generate_stream(self, messages, synthesize=True, meta_info=None):
        answer, buffer = "", ""
        final_tool_calls_data = {}
        received_textual_response = False
        first_token_time = None
        called_fun = None

        model_args = self.model_args.copy()
        model_args["messages"] = messages
        model_args["stream"] = True

        if self.trigger_function_call:
            model_args["tools"] = json.loads(self.tools) if isinstance(self.tools, str) else self.tools
            model_args["tool_choice"] = "auto"
            model_args["parallel_tool_calls"] = False

        logger.info(f"Request to model {self.model}: {messages} with args: {model_args}")
        start_time = time.time()
        latency_data = self.create_latency_data(meta_info)

        async for chunk in await acompletion(**model_args):
            choice = chunk["choices"][0]
            delta = choice.get("delta", {})

            if not first_token_time:
                first_token_time = time.time()
                self.update_first_token_latency(latency_data, start_time)

            if hasattr(delta, "tool_calls") and delta.tool_calls:
                if buffer:
                    yield buffer, True, latency_data, False, None, None
                    buffer = ""

                for tool_call in delta.tool_calls:
                    called_fun = self.accumulate_tool_call(final_tool_calls_data, tool_call)
                    logger.info(f"Tool function triggered: {called_fun}")

                if self.should_send_prefunction_message(received_textual_response):
                    pre_msg, api_tool_pre_call_message = self.create_prefunction_message(called_fun)
                    yield pre_msg, True, latency_data, False, called_fun, api_tool_pre_call_message

            elif hasattr(delta, "content") and delta.content:
                received_textual_response = True
                answer += delta.content
                chunk_text, buffer, should_yield = self.handle_streaming_buffer(buffer, delta.content, synthesize, latency_data)
                if should_yield:
                    yield chunk_text, False, latency_data, False, None, None

        latency_data["total_stream_duration_ms"] = round((time.time() - start_time) * 1000)

        if self.trigger_function_call and final_tool_calls_data:
            api_call_payload = self.build_api_call_payload(final_tool_calls_data, model_args, meta_info)
            if api_call_payload:
                func_conf = self.api_params.get(api_call_payload["called_fun"])
                if not func_conf:
                    logger.warning(f"No API config found for tool: {api_call_payload['called_fun']}")
                    return

                logger.info(f"Tool payload: {api_call_payload['parsed_args']} | Config: {func_conf}")
                self.validate_and_log_tool_call(api_call_payload, model_args)
                yield api_call_payload, False, latency_data, True, None, None

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

        if request_json is True:
            model_args['response_format'] = {
                "type": "json_object",
                "schema": json_to_pydantic_schema('{"classification_label": "classification label goes here"}')
            }
        logger.info(f'Request to litellm {model_args}')
        try:
            completion = await acompletion(**model_args)
            text = completion.choices[0].message.content
        except Exception as e:
            logger.error(f'Error generating response {e}')
        return text
