import time
import json
from bolna.constants import DEFAULT_LANGUAGE_CODE
from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)


class BaseLLM:
    def __init__(self, max_tokens=100, buffer_size=40, language=DEFAULT_LANGUAGE_CODE, **kwargs):
        self.buffer_size = buffer_size
        self.max_tokens = max_tokens
        self.language = language
        
        self.custom_tools = kwargs.get("api_tools", None)
        if self.custom_tools is not None:
            self.trigger_function_call = True
            self.api_params = self.custom_tools['tools_params']
            self.tools = self.custom_tools['tools']
        else:
            self.trigger_function_call = False
        
        self.run_id = kwargs.get("run_id", None)
        self.gave_out_prefunction_call_message = False
        self.started_streaming = False

    def handle_streaming_buffer(self, buffer, content, synthesize, latency_data):
        """Handle streaming buffer with word-boundary splitting"""
        buffer += content
        if synthesize and len(buffer) >= self.buffer_size:
            split = buffer.rsplit(" ", 1)
            return split[0], split[1] if len(split) > 1 else "", True
        return "", buffer, False

    def create_latency_data(self, meta_info):
        """Create initial latency tracking data"""
        return {
            "turn_id": meta_info.get("turn_id") if meta_info else None,
            "model": getattr(self, 'model', 'unknown'),
            "first_token_latency_ms": None,
            "total_stream_duration_ms": None,
        }

    def update_first_token_latency(self, latency_data, start_time):
        """Update latency data with first token timing"""
        latency_data["first_token_latency_ms"] = round((time.time() - start_time) * 1000)
        self.started_streaming = True

    def accumulate_tool_call(self, final_tool_calls_data, tool_call):
        """Accumulate tool call data from streaming chunks"""
        idx = tool_call.index
        if idx not in final_tool_calls_data:
            final_tool_calls_data[idx] = {
                "index": tool_call.index,
                "id": tool_call.id,
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments or ""
                },
                "type": "function"
            }
        else:
            final_tool_calls_data[idx]["function"]["arguments"] += tool_call.function.arguments or ""
        return tool_call.function.name

    def should_send_prefunction_message(self, received_textual_response):
        """Check if pre-function call message should be sent"""
        return not self.gave_out_prefunction_call_message and not received_textual_response

    def create_prefunction_message(self, called_fun):
        """Create pre-function call message"""
        from bolna.helpers.utils import compute_function_pre_call_message
        api_tool_pre_call_message = self.api_params.get(called_fun, {}).get('pre_call_message', None)
        pre_msg = compute_function_pre_call_message(self.language, called_fun, api_tool_pre_call_message)
        self.gave_out_prefunction_call_message = True
        return pre_msg, api_tool_pre_call_message

    def build_api_call_payload(self, final_tool_calls_data, model_args, meta_info):
        """Build API call payload from tool call data"""
        if not final_tool_calls_data:
            return None
            
        first_tool_call = final_tool_calls_data[0]["function"]
        func_name = first_tool_call["name"]
        args_str = first_tool_call["arguments"]
        
        func_conf = self.api_params.get(func_name)
        if not func_conf:
            return None
            
        try:
            parsed_args = json.loads(args_str) if isinstance(args_str, str) else args_str
        except json.JSONDecodeError:
            parsed_args = args_str
            
        return {
            "url": func_conf.get("url"),
            "method": (func_conf.get("method") or "").lower(),
            "param": func_conf.get("param"),
            "api_token": func_conf.get("api_token"),
            "model_args": model_args,
            "meta_info": meta_info,
            "called_fun": func_name,
            "model_response": list(final_tool_calls_data.values()),
            "tool_call_id": final_tool_calls_data[0].get("id"),
            "parsed_args": parsed_args
        }

    def validate_and_log_tool_call(self, api_call_payload, model_args):
        """Validate tool call arguments and log request"""
        from bolna.helpers.utils import convert_to_request_log
        
        func_name = api_call_payload["called_fun"]
        parsed_args = api_call_payload["parsed_args"]
        
        tools = model_args.get("tools", [])
        tool_spec = next((t for t in tools if t["function"]["name"] == func_name), None)
        
        if tool_spec:
            required_keys = tool_spec["function"].get("parameters", {}).get("required", [])
            if all(k in parsed_args for k in required_keys):
                convert_to_request_log(parsed_args, api_call_payload["meta_info"], 
                                     getattr(self, 'model', 'unknown'), "llm", 
                                     direction="response", is_cached=False, run_id=self.run_id)
                api_call_payload.update(parsed_args)
                return True
        
        api_call_payload["resp"] = None
        return False

    async def respond_back_with_filler(self, messages):
        pass

    async def generate(self, messages, stream=True):
        pass
