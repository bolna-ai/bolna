import asyncio
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI
import json, requests, time

from bolna.constants import CHECKING_THE_DOCUMENTS_FILLER, PRE_FUNCTION_CALL_MESSAGE, TRANSFERING_CALL_FILLER, DEFAULT_LANGUAGE_CODE
from bolna.helpers.utils import convert_to_request_log, format_messages
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
            #self.thread_id = self.openai.beta.threads.create().id
            self.model_args = {"max_completion_tokens": self.max_tokens, "temperature": self.temperature}
            my_assistant = self.openai.beta.assistants.retrieve(self.assistant_id)
            if my_assistant.tools is not None:
                self.tools = [i for i in my_assistant.tools if i.type == "function"]
            #logger.info(f'thread id : {self.thread_id}')
        self.run_id = kwargs.get("run_id", None)
        self.gave_out_prefunction_call_message = False

    async def generate_stream(self, messages, synthesize=True, request_json=False, meta_info=None):
        if len(messages) == 0:
            raise Exception("No messages provided")
        
        response_format = self.get_response_format(request_json)

        answer, buffer, arguments_received, called_fun, i = "", "", "", "", 0

        model_args = self.model_args
        model_args["response_format"] = response_format
        model_args["messages"] = messages
        model_args["stream"] = True
        model_args["stop"] = ["User:"]
        model_args["user"] = f"{self.run_id}#{meta_info['turn_id']}"
        
        latency = False
        start_time = time.time()
        tools = []
        if self.trigger_function_call:
            if type(self.tools) is str:
                tools = json.loads(self.tools)
            else:
                tools = self.tools
            model_args["tools"] = tools
            model_args["tool_choice"] = "auto"
            model_args["parallel_tool_calls"] = False
        received_textual_response = False
        self.gave_out_prefunction_call_message = False
        final_tool_calls_data = {}

        async for chunk in await self.async_client.chat.completions.create(**model_args):
            if not self.started_streaming:
                first_chunk_time = time.time()
                latency = first_chunk_time - start_time
                logger.info(f"LLM Latency: {latency:.2f} s")
                self.started_streaming = True

            delta = chunk.choices[0].delta
            if hasattr(delta, 'tool_calls') and delta.tool_calls:
                if buffer:
                    yield buffer, True, latency, False
                    buffer = ""

                # This for loop is going to cover the case of multiple tool calls. Currently, we are not allowing parallel
                # tool calls but if enabled in the future then this code should take care of accumulating the tool call data
                for tool_call in delta.tool_calls or []:
                    index = tool_call.index
                    if index not in final_tool_calls_data:
                        called_fun = tool_call.function.name
                        logger.info(f"Function given by LLM to trigger is - {called_fun}")
                        final_tool_calls_data[index] = {
                            "index": tool_call.index,
                            "id": tool_call.id,
                            "function": {
                                "name": called_fun,
                                "arguments": tool_call.function.arguments
                            },
                            "type": "function"
                        }
                    else:
                        final_tool_calls_data[index]["function"]["arguments"] += tool_call.function.arguments

                if not self.gave_out_prefunction_call_message and not received_textual_response:
                    filler = PRE_FUNCTION_CALL_MESSAGE.get(self.language, DEFAULT_LANGUAGE_CODE) if not called_fun.startswith("transfer_call") else TRANSFERING_CALL_FILLER.get(self.language, DEFAULT_LANGUAGE_CODE)
                    yield filler, True, latency, False
                    self.gave_out_prefunction_call_message = True

            elif hasattr(delta, 'content') and delta.content is not None:
                received_textual_response = True
                answer += delta.content
                buffer += delta.content
                if len(buffer) >= self.buffer_size and synthesize:
                    buffer_words = buffer.split(" ")
                    text = ' '.join(buffer_words[:-1])
                    yield text, False, latency, False
                    buffer = buffer_words[-1]

        if self.trigger_function_call and final_tool_calls_data and final_tool_calls_data[0]["function"]["name"] in self.api_params:
            i = [i for i in range(len(tools)) if called_fun == tools[i]["function"]["name"]][0]
            func_dict = self.api_params[called_fun]
            arguments_received = final_tool_calls_data[0]["function"]["arguments"]
            logger.info(f"Payload to send {arguments_received} func_dict {func_dict}")
            self.gave_out_prefunction_call_message = False

            api_call_return = {
                "url": func_dict['url'],
                "method": None if func_dict['method'] is None else func_dict['method'].lower(),
                "param": func_dict['param'],
                "api_token": func_dict['api_token'],
                "model_args": model_args,
                "meta_info": meta_info,
                "called_fun": called_fun,
                "model_response": list(final_tool_calls_data.values()),
                "tool_call_id": final_tool_calls_data[0].get("id", "")
            }

            all_required_keys = tools[i]["function"]["parameters"]["properties"].keys() and tools[i]["function"]["parameters"].get(
                "required", [])
            if tools[i]["function"].get("parameters", None) is not None and (all(key in arguments_received for key in all_required_keys)):
                logger.info(f"Function call parameters: {arguments_received}")
                convert_to_request_log(arguments_received, meta_info, self.model, "llm", direction="response", is_cached=False,
                                       run_id=self.run_id)
                arguments_received = json.loads(arguments_received)
                api_call_return = {**api_call_return, **arguments_received}
            else:
                api_call_return['resp'] = None
            yield api_call_return, False, latency, True

        if synthesize:  # This is used only in streaming sense
            yield buffer, True, latency, False
        else:
            yield answer, True, latency, False
        self.started_streaming = False
    
    async def generate(self, messages, request_json=False):
        response_format = self.get_response_format(request_json)

        completion = await self.async_client.chat.completions.create(model=self.model, temperature=0.0, messages=messages,
                                                                     stream=False, response_format=response_format)
        res = completion.choices[0].message.content
        return res

    async def generate_assistant_stream(self, message, synthesize=True, request_json=False, meta_info=None):
        if len(message) == 0:
            raise Exception("No messages provided")

        response_format = self.get_response_format(request_json)

        answer, buffer, resp, called_fun, api_params, i = "", "", "", "", "", 0

        latency = False
        start_time = time.time()
        textual_response = False

        tools = []
        if self.trigger_function_call:
            if type(self.tools) is str:
                tools = json.loads(self.tools)
            else:
                tools = self.tools
        
        thread_id = self.openai.beta.threads.create(messages= message[1:-2]).id

        model_args = self.model_args
        model_args["thread_id"] = thread_id
        model_args["assistant_id"] = self.assistant_id
        model_args["stream"] = True
        model_args["response_format"] = response_format

        await self.async_client.beta.threads.messages.create(thread_id=model_args["thread_id"], role="user", content=message[-1]['content'])

        async for chunk in await self.async_client.beta.threads.runs.create(**model_args):
            logger.info(f"chunk received : {chunk}")
            if self.trigger_function_call and chunk.event == "thread.run.step.delta":
                if chunk.data.delta.step_details.tool_calls[0].type == "file_search" or chunk.data.delta.step_details.tool_calls[0].type == "search_files":
                    yield CHECKING_THE_DOCUMENTS_FILLER, False, time.time() - start_time, False
                    continue
                textual_response = False
                if not self.started_streaming:
                    first_chunk_time = time.time()
                    latency = first_chunk_time - start_time
                    logger.info(f"LLM Latency: {latency:.2f} s")
                    self.started_streaming = True
                
                if chunk.data.delta.step_details.tool_calls[0].function.name and chunk.data.delta.step_details.tool_calls[0].function.arguments is not None:
                    logger.info(f"Should do a function call {chunk.data.delta.step_details.tool_calls[0].function.name}")
                    called_fun = str(chunk.data.delta.step_details.tool_calls[0].function.name)
                    i = [i for i in range(len(tools)) if called_fun == tools[i].function.name][0]
                    
                if not self.gave_out_prefunction_call_message and not textual_response:
                    filler = PRE_FUNCTION_CALL_MESSAGE.get(self.language, DEFAULT_LANGUAGE_CODE) if not called_fun.startswith("transfer_call_") else TRANSFERING_CALL_FILLER.get(self.language, DEFAULT_LANGUAGE_CODE)
                    yield filler, True, latency, False
                    self.gave_out_prefunction_call_message = True
                if len(buffer) > 0:
                    yield buffer, False, latency, False
                    buffer = ''
                yield buffer, False, latency, False
                buffer = ''
                
                if (text_chunk := chunk.data.delta.step_details.tool_calls[0].function.arguments):
                    resp += text_chunk
                    logger.info(f"Response from LLM {resp}")
            elif chunk.event == 'thread.message.delta':
                if not self.started_streaming:
                    first_chunk_time = time.time()
                    latency = first_chunk_time - start_time
                    logger.info(f"LLM Latency: {latency:.2f} s")
                    self.started_streaming = True
                textual_response = True
                text_chunk = chunk.data.delta.content[0].text.value
                answer += text_chunk
                buffer += text_chunk
                if len(buffer) >= self.buffer_size and synthesize:
                    buffer_words = buffer.split(" ")
                    text = ' '.join(buffer_words[:-1])

                    if not self.started_streaming:
                        self.started_streaming = True
                    yield text, False, latency, False
                    buffer = buffer_words[-1]
        
        if self.trigger_function_call and called_fun in self.api_params:
            func_dict = self.api_params[called_fun]
            logger.info(f"PAyload to send {resp} func_dict {func_dict} and tools {tools}")
            self.gave_out_prefunction_call_message = False

            url = func_dict['url']
            method = func_dict['method']
            param = func_dict['param']
            api_token = func_dict['api_token']
            model_args['messages'] = message
            api_call_return = {
                "url": url, 
                "method":None if method is None else method.lower(),
                "param": param, 
                "api_token":api_token, 
                "model_args": model_args,
                "meta_info": meta_info,
                "called_fun": called_fun,
                #**resp
            }
        
            if tools[i].function.parameters is not None and (all(key in resp for key in tools[i].function.parameters["properties"].keys())):
                logger.info(f"Function call paramaeters {resp}")
                convert_to_request_log(resp, meta_info, self.model, "llm", direction = "response", is_cached= False, run_id = self.run_id)
                resp  = json.loads(resp)
                api_call_return = {**api_call_return, **resp}
            else:
                logger.info(f"No parameters in function call")
                api_call_return['resp'] = None
            yield api_call_return, False, latency, True

        if synthesize:  # This is used only in streaming sense
            yield buffer, True, latency, False
        else:
            yield answer, True, latency, False
        self.started_streaming = False

    def get_response_format(self, is_json_format: bool):
        if is_json_format and self.model in ('gpt-4-1106-preview', 'gpt-3.5-turbo-1106', 'gpt-4o-mini'):
            return {"type": "json_object"}
        else:
            return {"type": "text"}