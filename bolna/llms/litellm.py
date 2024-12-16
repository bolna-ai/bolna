import os
import litellm
from dotenv import load_dotenv
from .llm import BaseLLM
from bolna.constants import DEFAULT_LANGUAGE_CODE, PRE_FUNCTION_CALL_MESSAGE, TRANSFERING_CALL_FILLER
from bolna.helpers.utils import json_to_pydantic_schema, convert_to_request_log
from bolna.helpers.logger_config import configure_logger
import time
import json

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

        # Function calling setup
        self.custom_tools = kwargs.get("api_tools", None)
        if self.custom_tools is not None:
            self.trigger_function_call = True
            self.api_params = self.custom_tools['tools_params']
            logger.info(f"Function dict {self.api_params}")
            # Convert tools to LiteLLM format
            self.tools = [
                {
                    "type": "function",
                    "function": tool
                } for tool in self.custom_tools['tools']
            ]
        else:
            self.trigger_function_call = False
        
        self.gave_out_prefunction_call_message = False
        self.run_id = kwargs.get("run_id", None)

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

    async def generate_stream(self, messages, synthesize=True, request_json=False, meta_info=None):
        answer, buffer, resp, called_fun, i = "", "", "", "", 0
        model_args = self.model_args.copy()
        model_args["messages"] = messages
        model_args["stream"] = True
        model_args["stop"] = ["User:"]
        model_args["user"] = f"{self.run_id}#{meta_info['turn_id']}" if meta_info else None

        tools = []
        if self.trigger_function_call:
            if type(self.tools) is str:
                tools = json.loads(self.tools)
            else:
                tools = self.tools
            model_args["tools"] = tools
            model_args["tool_choice"] = "auto"

        logger.info(f"request to model: {self.model}: {messages} and model args {model_args}")
        latency = False
        start_time = time.time()
        textual_response = False

        async for chunk in await litellm.acompletion(**model_args):
            if not self.started_streaming:
                first_chunk_time = time.time()
                latency = first_chunk_time - start_time
                logger.info(f"LLM Latency: {latency:.2f} s")
                self.started_streaming = True

            delta = chunk['choices'][0]['delta']
            
            if self.trigger_function_call and hasattr(delta, 'tool_calls') and delta.tool_calls:
                tool_call = delta.tool_calls[0]
                
                if hasattr(tool_call, 'function'):
                    function_data = tool_call.function
                    logger.info(f"function_data: {function_data}")
                    
                    if hasattr(function_data, 'name') and function_data.name:
                        logger.info(f"Should do a function call {function_data.name}")
                        called_fun = str(function_data.name)
                        i = [i for i in range(len(self.tools)) if called_fun == self.tools[i]["function"]["name"]][0]

                    if not self.gave_out_prefunction_call_message and not textual_response:
                        filler = PRE_FUNCTION_CALL_MESSAGE if not called_fun.startswith("transfer_call") else TRANSFERING_CALL_FILLER.get(self.language, DEFAULT_LANGUAGE_CODE)
                        yield filler, True, latency, False, None, True
                        self.gave_out_prefunction_call_message = True

                    if len(buffer) > 0:
                        yield buffer, True, latency, False, None, True
                        buffer = ''
                    logger.info(f"Response from LLM {resp}")
                        
                    if buffer != '':
                        yield buffer, False, latency, False, None, True
                        buffer = ''    
                    if hasattr(function_data, 'arguments') and function_data.arguments:
                        resp += function_data.arguments

            elif hasattr(delta, 'content') and delta.content:
                text_chunk = delta.content
                textual_response = True
                answer += text_chunk
                buffer += text_chunk

                if len(buffer) >= self.buffer_size and synthesize:
                    buffer_words = buffer.split(" ")
                    text = ' '.join(buffer_words[:-1])

                    if not self.started_streaming:
                        self.started_streaming = True
                    yield text, False, latency, False, None, True
                    buffer = buffer_words[-1]

        if self.trigger_function_call and called_fun and called_fun in self.api_params:
            func_dict = self.api_params[called_fun]
            logger.info(f"Payload to send {resp} func_dict {func_dict}")
            self.gave_out_prefunction_call_message = False

            url = func_dict['url']
            method = func_dict['method']
            param = func_dict['param']
            api_token = func_dict['api_token']
            header = func_dict['header'] or None
            api_call_return = {
                "url": url,
                "method": None if method is None else method.lower(),
                "param": param,
                "api_token": api_token,
                "header": header,
                "model_args": model_args,
                "meta_info": meta_info,
                "called_fun": called_fun,
            }

            tool_params = tools[i]["function"]["parameters"]
            all_required_keys = tool_params["properties"].keys() and tool_params.get("required", [])
            
            if tool_params is not None and (all(key in resp for key in all_required_keys)):
                logger.info(f"Function call parameters: {resp}")
                convert_to_request_log(resp, meta_info, self.model, "llm", direction="response", is_cached=False, run_id=self.run_id)
                resp = json.loads(resp)
                api_call_return = {**api_call_return, **resp}
            else:
                api_call_return['resp'] = None
            logger.info(f"api call return: {api_call_return}")
            yield api_call_return, False, latency, True, tool_call.id, True

        if synthesize:
            yield buffer, True, latency, False, None, True
        else:
            yield answer, True, latency, False, None, True
        self.started_streaming = False
        logger.info(f"Time to generate response {time.time() - start_time} {answer}")

    async def generate(self, messages, stream=False, request_json=False, meta_info=None):
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
            
        if self.trigger_function_call:
            model_args["tools"] = self.tools
            model_args["tool_choice"] = "auto"
            
        logger.info(f'Request to litellm {model_args}')
        try:
            completion = await litellm.acompletion(**model_args)
            text = completion.choices[0].message.content
        except Exception as e:
            logger.error(f'Error generating response {e}')
        return text
