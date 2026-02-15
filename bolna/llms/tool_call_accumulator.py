import json
from bolna.helpers.utils import convert_to_request_log, compute_function_pre_call_message
from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)


class ToolCallAccumulator:
    def __init__(self, api_params: dict, tools: list, language: str, model: str, run_id: str):
        self.api_params = api_params
        self.tools = tools
        self.language = language
        self.model = model
        self.run_id = run_id
        self.final_tool_calls = {}
        self.called_fun = None
        self._gave_pre_call_msg = False
        self.received_textual = False

    def process_delta(self, tool_calls_delta) -> None:
        for tool_call in tool_calls_delta or []:
            idx = tool_call.index
            if idx not in self.final_tool_calls:
                self.called_fun = tool_call.function.name
                logger.info(f"Function given by LLM to trigger is - {self.called_fun}")
                self.final_tool_calls[idx] = {
                    "index": tool_call.index,
                    "id": tool_call.id,
                    "function": {
                        "name": self.called_fun,
                        "arguments": tool_call.function.arguments or ""
                    },
                    "type": "function"
                }
            else:
                self.final_tool_calls[idx]["function"]["arguments"] += tool_call.function.arguments or ""

    def get_pre_call_message(self, meta_info: dict) -> tuple[str, str | None, str | None] | None:
        if self._gave_pre_call_msg or self.received_textual or not self.called_fun:
            return None
        self._gave_pre_call_msg = True
        api_tool_pre_call_message = self.api_params.get(self.called_fun, {}).get('pre_call_message', None)
        detected_lang = meta_info.get('detected_language') if meta_info else None
        active_language = detected_lang or self.language
        pre_msg = compute_function_pre_call_message(active_language, self.called_fun, api_tool_pre_call_message)
        return pre_msg, self.called_fun, api_tool_pre_call_message

    def build_api_payload(self, model_args: dict, meta_info: dict, answer: str) -> dict | None:
        if not self.final_tool_calls:
            return None

        first_func_name = self.final_tool_calls[0]["function"]["name"]
        if first_func_name not in self.api_params:
            return None

        func_conf = self.api_params[first_func_name]
        arguments_received = self.final_tool_calls[0]["function"]["arguments"]

        logger.info(f"Payload to send {arguments_received} func_dict {func_conf}")
        self._gave_pre_call_msg = False

        api_call_payload = {
            "url": func_conf.get('url'),
            "method": (func_conf.get('method') or "").lower() or None,
            "param": func_conf.get('param'),
            "api_token": func_conf.get('api_token'),
            "headers": func_conf.get('headers', None),
            "model_args": model_args,
            "meta_info": meta_info,
            "called_fun": first_func_name,
            "model_response": list(self.final_tool_calls.values()),
            "tool_call_id": self.final_tool_calls[0].get("id", ""),
            "textual_response": answer.strip() if self.received_textual else None
        }

        # Find tool spec for validation
        tool_spec = next((t for t in self.tools if t["function"]["name"] == first_func_name), None)

        if tool_spec:
            try:
                parsed_args = json.loads(arguments_received)
                required_keys = tool_spec["function"].get("parameters", {}).get("required", [])
                if tool_spec["function"].get("parameters") is not None and all(k in parsed_args for k in required_keys):
                    convert_to_request_log(arguments_received, meta_info, self.model, "llm",
                                           direction="response", is_cached=False, run_id=self.run_id)
                    api_call_payload.update(parsed_args)
                else:
                    api_call_payload['resp'] = None
            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"Error parsing function arguments: {e}")
                api_call_payload['resp'] = None
        else:
            api_call_payload['resp'] = None

        return api_call_payload
