from bolna.constants import ROLE_SYSTEM, ROLE_USER, ROLE_ASSISTANT, ROLE_TOOL
from bolna.enums import ResponseItemType


class MessageFormatAdapter:
    @staticmethod
    def chat_to_responses_input(messages: list[dict]) -> tuple[str, list[dict]]:
        """Chat Completions messages -> (instructions, Responses API input items)."""
        instructions = ""
        input_items = []

        for msg in messages:
            role = msg.get("role")

            if role == ROLE_SYSTEM:
                instructions = msg.get("content", "")

            elif role == ROLE_USER:
                input_items.append({
                    "type": ResponseItemType.MESSAGE,
                    "role": ROLE_USER,
                    "content": msg.get("content", ""),
                })

            elif role == ROLE_ASSISTANT:
                tool_calls = msg.get("tool_calls")
                if tool_calls:
                    for tc in tool_calls:
                        func = tc.get("function", {})
                        input_items.append({
                            "type": ResponseItemType.FUNCTION_CALL,
                            "call_id": tc.get("id", ""),
                            "name": func.get("name", ""),
                            "arguments": func.get("arguments", ""),
                        })
                else:
                    content = msg.get("content")
                    if content is not None:
                        input_items.append({
                            "type": ResponseItemType.MESSAGE,
                            "role": ROLE_ASSISTANT,
                            "content": content,
                        })

            elif role == ROLE_TOOL:
                input_items.append({
                    "type": ResponseItemType.FUNCTION_CALL_OUTPUT,
                    "call_id": msg.get("tool_call_id", ""),
                    "output": msg.get("content", ""),
                })

        return instructions, input_items

    @staticmethod
    def chat_tools_to_responses_tools(chat_tools: list[dict]) -> list[dict]:
        """Flatten nested tool schema for Responses API.

        {"type":"function","function":{name,desc,params}}
        -> {"type":"function","name":...,"description":...,"parameters":...,"strict":true}
        """
        result = []
        for tool in chat_tools:
            func = tool.get("function", {})
            result.append({
                "type": ResponseItemType.FUNCTION,
                "name": func.get("name", ""),
                "description": func.get("description", ""),
                "parameters": func.get("parameters", {}),
                "strict": func.get("strict", False),
            })
        return result
