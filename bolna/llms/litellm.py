import os
import json
import time
from litellm import acompletion
from dotenv import load_dotenv

from bolna.constants import DEFAULT_LANGUAGE_CODE
from bolna.helpers.utils import json_to_pydantic_schema
from .llm import BaseLLM
from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)
load_dotenv()


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

    async def generate_stream(self, messages, synthesize=True, meta_info=None):
        answer, buffer = "", ""
        latency = None
        function_call_detected = False

        model_args = self.model_args.copy()
        model_args["messages"] = messages
        model_args["stream"] = True

        # Add tool/function call support if enabled
        if self.trigger_function_call:
            model_args["tools"] = json.loads(self.tools) if isinstance(self.tools, str) else self.tools
            model_args["tool_choice"] = "auto"
            model_args["parallel_tool_calls"] = False  # Optional: adjust as needed

        logger.info(f"Request to model {self.model}: {messages} with args: {model_args}")
        start_time = time.time()

        async for chunk in await acompletion(**model_args):
            choice = chunk["choices"][0]

            # Track latency (first token time)
            if not self.started_streaming:
                latency = time.time() - start_time
                logger.info(f"LLM latency: {latency:.2f} s")
                self.started_streaming = True

            # Function call streaming (delta)
            if choice.get("delta", {}).get("function_call"):
                function_call_detected = True
                func_delta = choice["delta"]["function_call"]

                if not hasattr(self, "_function_buffer"):
                    self._function_buffer = {"name": "", "arguments": ""}

                self._function_buffer["name"] += func_delta.get("name", "")
                self._function_buffer["arguments"] += func_delta.get("arguments", "")

            # Regular token streaming
            elif (text_chunk := choice.get("delta", {}).get("content")) and not choice.get("finish_reason"):
                answer += text_chunk
                buffer += text_chunk

                if len(buffer) >= self.buffer_size and synthesize:
                    text = ' '.join(buffer.split(" ")[:-1])
                    if text.strip():
                        yield text, False, latency, False, None, None
                    buffer = buffer.split(" ")[-1]

        # Final chunk processing
        if function_call_detected and hasattr(self, "_function_buffer"):
            try:
                function_args = json.loads(self._function_buffer["arguments"])
            except Exception as e:
                logger.warning(f"Invalid function call args: {self._function_buffer['arguments']}")
                function_args = self._function_buffer["arguments"]

            yield "", True, latency, True, self._function_buffer["name"], function_args
            del self._function_buffer

        elif synthesize:
            if buffer.strip():
                yield buffer, True, latency, False, None, None
        else:
            yield answer, True, latency, False, None, None

        self.started_streaming = False
        logger.info(f"Total generation time: {time.time() - start_time:.2f}s, output: {answer}")

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
            completion = await litellm.acompletion(**model_args)
            text = completion.choices[0].message.content
        except Exception as e:
            logger.error(f'Error generating response {e}')
        return text
