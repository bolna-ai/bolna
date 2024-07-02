import os
import time
from dotenv import load_dotenv
import litellm

from bolna.helpers.logger_config import configure_logger
from bolna.helpers.utils import json_to_pydantic_schema
from bolna.llms.llm import BaseLLM


logger = configure_logger(__name__)
load_dotenv()

class GroqLLM(BaseLLM):
    def __init__(self, model="llama3-8b-8192", max_tokens=100, buffer_size=40, temperature=0.0, **kwargs):
        super().__init__(max_tokens, buffer_size)
        self.model = model
        self.started_streaming = False
        self.model_args = {"max_tokens": max_tokens, "temperature": temperature, "model": self.model}
        self.api_key = kwargs.get("llm_key", os.getenv('GROQ_API_KEY'))

        if self.api_key:
            self.model_args["api_key"] = self.api_key
        if "llm_key" in kwargs:
            self.model_args["api_key"] = kwargs["llm_key"]


    async def generate_stream(self, messages, synthesize=True, request_json=False):
        answer, buffer = "", ""
        model_args = self.model_args.copy()
        model_args["messages"] = messages
        model_args["stream"] = True

        logger.info(f"Request to model: {self.model}: {messages} and model args {model_args}")
        latency = False
        start_time = time.time()

        async for chunk in await litellm.acompletion(api_key=self.api_key, model=f"groq/{self.model}", messages=messages, stream=True, max_tokens=self.max_tokens, request_json=request_json):
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
                    yield text, False, latency, False
                    buffer = buffer.split(" ")[-1]

        if buffer:
            yield buffer, True, latency, False
        else:
            yield answer, True, latency, False
        self.started_streaming = False
        logger.info(f"Time to generate response {time.time() - start_time} {answer}")
        
        
    async def generate(self, messages, stream=False, request_json=False):
        text = ""
        model_args = self.model_args.copy()
        model_args["model"] = self.model
        model_args["messages"] = messages
        model_args["stream"] = stream

        if request_json:
            model_args['response_format'] = {
                "type": "json_object",
                "schema": json_to_pydantic_schema('{"classification_label": "classification label goes here"}')
            }
        logger.info(f'Request to Groq LLM {model_args}')
        try:
            completion = await litellm.acompletion(**model_args)
            text = completion.choices[0].message.content
            logger.debug(completion)  # Changed to debug for non-error logging
        except Exception as e:
            logger.error(f'Error generating response {e}')
        return text