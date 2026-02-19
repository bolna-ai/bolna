from typing import Optional

from pydantic import BaseModel

from bolna.constants import ROLE_SYSTEM, ROLE_USER, ROLE_ASSISTANT, ROLE_TOOL
from bolna.enums import ResponseItemType


class ChatToolCallFunction(BaseModel):
    name: str = ""
    arguments: str = ""


class ChatToolCall(BaseModel):
    id: str = ""
    function: ChatToolCallFunction = ChatToolCallFunction()


class ChatMessage(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[list[ChatToolCall]] = None
    tool_call_id: Optional[str] = None


class ChatToolFunction(BaseModel):
    name: str = ""
    description: str = ""
    parameters: dict = {}
    strict: bool = False


class ChatToolDefinition(BaseModel):
    type: str = "function"
    function: ChatToolFunction = ChatToolFunction()


class MessageFormatAdapter:
    @staticmethod
    def chat_to_responses_input(messages: list[dict]) -> tuple[str, list[dict]]:
        """Chat Completions messages -> (instructions, Responses API input items)."""
        instructions = ""
        input_items = []

        parsed = [ChatMessage(**msg) for msg in messages]
        for msg in parsed:
            if msg.role == ROLE_SYSTEM:
                instructions = msg.content or ""

            elif msg.role == ROLE_USER:
                input_items.append({
                    "type": ResponseItemType.MESSAGE,
                    "role": ROLE_USER,
                    "content": msg.content or "",
                })

            elif msg.role == ROLE_ASSISTANT:
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        input_items.append({
                            "type": ResponseItemType.FUNCTION_CALL,
                            "call_id": tc.id,
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        })
                else:
                    if msg.content is not None:
                        input_items.append({
                            "type": ResponseItemType.MESSAGE,
                            "role": ROLE_ASSISTANT,
                            "content": msg.content,
                        })

            elif msg.role == ROLE_TOOL:
                input_items.append({
                    "type": ResponseItemType.FUNCTION_CALL_OUTPUT,
                    "call_id": msg.tool_call_id or "",
                    "output": msg.content or "",
                })

        return instructions, input_items

    @staticmethod
    def chat_tools_to_responses_tools(chat_tools: list[dict]) -> list[dict]:
        """Flatten nested tool schema for Responses API.

        {"type":"function","function":{name,desc,params}}
        -> {"type":"function","name":...,"description":...,"parameters":...,"strict":true}
        """
        result = []
        parsed = [ChatToolDefinition(**tool) for tool in chat_tools]
        for tool in parsed:
            result.append({
                "type": ResponseItemType.FUNCTION,
                "name": tool.function.name,
                "description": tool.function.description,
                "parameters": tool.function.parameters,
                "strict": tool.function.strict,
            })
        return result
