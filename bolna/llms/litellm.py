import os
import litellm
from dotenv import load_dotenv
from .llm import BaseLLM
from bolna.constants import DEFAULT_LANGUAGE_CODE
from bolna.helpers.utils import json_to_pydantic_schema
from bolna.helpers.logger_config import configure_logger
import time

logger = configure_logger(__name__)
load_dotenv()


class LiteLLM(BaseLLM):
    def __init__(self, model, max_tokens=30, buffer_size=40, temperature=0.0, language=DEFAULT_LANGUAGE_CODE, **kwargs):
        super().__init__(max_tokens, buffer_size)
        self.model = model
        # self hosted azure
        if 'azure_model' in kwargs and kwargs['azure_model']:
            self.model = kwargs['azure_model']

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

    async def generate_stream(self, messages, synthesize=True, meta_info = None):
        answer, buffer = "", ""
        model_args = self.model_args.copy()
        model_args["messages"] = messages
        model_args["stream"] = True

        logger.info(f"request to model: {self.model}: {messages} and model args {model_args}")
        latency = False
        start_time = time.time()
        async for chunk in await litellm.acompletion(**model_args):
            if not self.started_streaming:
                first_chunk_time = time.time()
                latency = first_chunk_time - start_time
                logger.info(f"LLM Latency: {latency:.2f} s")
                self.started_streaming = True
            if (text_chunk := chunk['choices'][0]['delta'].content) and not chunk['choices'][0].finish_reason:
                answer += text_chunk
                buffer += text_chunk

                if len(buffer) >= self.buffer_size and synthesize:
                    text = ' '.join(buffer.split(" ")[:-1])

                    if synthesize:
                        if not self.started_streaming:
                            self.started_streaming = True
                        yield text, False, latency, False
                    buffer = buffer.split(" ")[-1]

        if synthesize:
            if buffer != "":
                yield buffer, True, latency, False
        else:
            yield answer, True, latency, False
        self.started_streaming = False
        logger.info(f"Time to generate response {time.time() - start_time} {answer}")

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
