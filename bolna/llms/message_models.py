import json
from typing import Optional

from pydantic import BaseModel

from bolna.enums import ChatRole, ResponseItemType


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


# Bookkeeping bolna attaches to history messages for correlation and audio gating. Removed
# before the request is built: OpenAI-compatible providers forward unknown message keys
# verbatim, and nothing should depend on the server ignoring them. Denylist rather than an
# allowlist so a legitimate provider key (name, cache_control, multimodal parts) is never
# silently dropped.
INTERNAL_MESSAGE_KEYS = frozenset(
    {"turn_id", "response_uid", "asr_turn_id", "sequence_id", "message_category", "exclude_from_llm"}
)


def strip_internal_keys(messages: list[dict]) -> list[dict]:
    """Drop bolna's own bookkeeping keys, leaving every other key untouched."""
    return [
        {k: v for k, v in m.items() if k not in INTERNAL_MESSAGE_KEYS} if isinstance(m, dict) else m for m in messages
    ]


def first_tool_call_result(completion, overflowed: bool = False) -> Optional[dict]:
    """Normalize an OpenAI-shaped forced tool-call completion into a routing result.

    Shared by every provider whose SDK returns OpenAI-shaped tool_calls (OpenAI, Azure, LiteLLM).
    Returns None when the model emitted no tool call.
    """
    message = completion.choices[0].message
    if not getattr(message, "tool_calls", None):
        return None
    call = message.tool_calls[0].function
    usage = {}
    u = getattr(completion, "usage", None)
    if u:
        usage = {"input_tokens": u.prompt_tokens, "output_tokens": u.completion_tokens}
        details = getattr(u, "completion_tokens_details", None)
        if details:
            usage["reasoning_tokens"] = getattr(details, "reasoning_tokens", None)
        prompt_details = getattr(u, "prompt_tokens_details", None)
        if prompt_details:
            usage["cached_tokens"] = getattr(prompt_details, "cached_tokens", None)
    return {
        "function_name": call.name,
        "arguments": json.loads(call.arguments) if call.arguments else {},
        "usage": usage,
        "service_tier": getattr(completion, "service_tier", None),
        "overflowed": overflowed,
    }


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
        """Chat Completions messages -> (instructions, Responses API input items).

        System is emitted as a role=system input item, not as `instructions`,
        because the instructions field breaks prompt-cache hashing.
        """
        instructions = ""
        input_items = []

        parsed = [ChatMessage(**msg) for msg in messages]
        for msg in parsed:
            if msg.role == ChatRole.SYSTEM:
                input_items.append(
                    {
                        "type": ResponseItemType.MESSAGE,
                        "role": ChatRole.SYSTEM,
                        "content": msg.content or "",
                    }
                )

            elif msg.role == ChatRole.USER:
                input_items.append(
                    {
                        "type": ResponseItemType.MESSAGE,
                        "role": ChatRole.USER,
                        "content": msg.content or "",
                    }
                )

            elif msg.role == ChatRole.ASSISTANT:
                if msg.content is not None:
                    input_items.append(
                        {
                            "type": ResponseItemType.MESSAGE,
                            "role": ChatRole.ASSISTANT,
                            "content": msg.content,
                        }
                    )
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        input_items.append(
                            {
                                "type": ResponseItemType.FUNCTION_CALL,
                                "call_id": tc.id,
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            }
                        )

            elif msg.role == ChatRole.TOOL:
                input_items.append(
                    {
                        "type": ResponseItemType.FUNCTION_CALL_OUTPUT,
                        "call_id": msg.tool_call_id or "",
                        "output": msg.content or "",
                    }
                )

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
            result.append(
                {
                    "type": ResponseItemType.FUNCTION,
                    "name": tool.function.name,
                    "description": tool.function.description,
                    "parameters": tool.function.parameters,
                    "strict": tool.function.strict,
                }
            )
        return result

    @staticmethod
    def chat_tool_choice_to_responses(tool_choice):
        """Flatten Chat-Completions tool_choice to the Responses API shape; pass strings/None through."""
        if isinstance(tool_choice, dict) and tool_choice.get("type") == "function" and "function" in tool_choice:
            return {"type": "function", "name": tool_choice["function"].get("name")}
        return tool_choice
