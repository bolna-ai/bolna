"""A forced tool call made through a JSON schema, for backends that cannot force tool_choice.

The reply is constrained to the call's JSON and re-streamed as the tool-call deltas the forced
tool_choice would have produced, so everything downstream reads it as an ordinary tool call.
"""

import json
import re
import uuid
from dataclasses import dataclass
from typing import AsyncIterator, Optional

from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall, ChoiceDeltaToolCallFunction

from bolna.enums import LLMProvider
from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)

# Providers whose server accepts a forced tool_choice without enforcing it, so the model may answer instead.
SCHEMA_FORCED_TOOL_PROVIDERS = frozenset({LLMProvider.BOLNA.value})

_ENVELOPE_HEAD = re.compile(r'\s*\{\s*"name"\s*:\s*"((?:[^"\\]|\\.)*)"\s*,\s*"arguments"\s*:\s*')
_TRAILING_CLOSE = re.compile(r"[}\s]*$")


@dataclass
class ForcedToolCall:
    response_format: dict
    # Set when one named function is forced; None when the reply picks among several ("required").
    function_name: Optional[str]

    def stream(self, chunks) -> AsyncIterator:
        return _as_tool_call_deltas(chunks, self.function_name)


def _forced_tool_call(tools, tool_choice) -> Optional[ForcedToolCall]:
    """The json_schema stand-in for a forced tool_choice, or None when the request forces nothing."""
    functions = {t["function"]["name"]: t["function"] for t in tools or []}
    if isinstance(tool_choice, dict):
        name = (tool_choice.get("function") or {}).get("name")
        if name not in functions:
            return None
        return ForcedToolCall(_response_format(name, _parameters(functions[name])), name)
    if tool_choice == "required" and functions:
        options = [
            {
                "type": "object",
                "properties": {"name": {"const": name}, "arguments": _parameters(fn)},
                "required": ["name", "arguments"],
                "additionalProperties": False,
            }
            for name, fn in functions.items()
        ]
        return ForcedToolCall(_response_format("tool_call", {"anyOf": options}), None)
    return None


def force_tools_by_schema(model_args: dict) -> Optional[ForcedToolCall]:
    """Swap a forced tool_choice for the JSON schema it implies; the tools stay, as the schema never reaches the model."""
    forced = _forced_tool_call(model_args.get("tools"), model_args.get("tool_choice"))
    if forced:
        model_args["tool_choice"] = "auto"
        # A stop string inside an argument would cut the JSON short.
        model_args.pop("stop", None)
        model_args["response_format"] = forced.response_format
    return forced


def _parameters(function: dict) -> dict:
    return function.get("parameters") or {"type": "object", "properties": {}}


def _response_format(name: str, schema: dict) -> dict:
    return {"type": "json_schema", "json_schema": {"name": name, "schema": schema, "strict": True}}


class _Arguments:
    """The reply is the forced function's arguments."""

    def __init__(self, name: str):
        self.name = name

    def feed(self, text: str) -> tuple[Optional[str], str]:
        return self.name, text

    def finish(self) -> tuple[Optional[str], str]:
        return None, ""


class _Envelope:
    """The reply is {"name": ..., "arguments": {...}}; stream the inner object as the arguments."""

    def __init__(self):
        self.buffer = ""
        self.name: Optional[str] = None
        # The tail may hold the envelope's own closing brace, so it is released only once text follows it.
        self.held = ""

    def feed(self, text: str) -> tuple[Optional[str], str]:
        if self.name is None:
            self.buffer += text
            head = _ENVELOPE_HEAD.match(self.buffer)
            if not head:
                return None, ""
            self.name = json.loads(f'"{head.group(1)}"')
            return self.name, self._release(self.buffer[head.end() :])
        return None, self._release(text)

    def _release(self, text: str) -> str:
        text = self.held + text
        cut = _TRAILING_CLOSE.search(text).start()
        self.held = text[cut:]
        return text[:cut]

    def finish(self) -> tuple[Optional[str], str]:
        if self.name is not None:
            closing = self.held.rstrip()
            return None, closing[:-1] if closing.endswith("}") else closing
        # The head never matched (e.g. properties out of order): parse the whole reply instead.
        try:
            reply = json.loads(self.buffer)
            return reply["name"], json.dumps(reply.get("arguments") or {})
        except (json.JSONDecodeError, KeyError, TypeError):
            logger.error(f"Schema-forced tool call was not a valid envelope: {self.buffer[:200]!r}")
            return None, ""


async def _as_tool_call_deltas(chunks, function_name: Optional[str]):
    translator = _Arguments(function_name) if function_name else _Envelope()
    call_id = f"call_{uuid.uuid4().hex}"
    named = False

    def tool_delta(name, arguments):
        nonlocal named
        if name and not named:
            named = True
            return ChoiceDeltaToolCall(
                index=0,
                id=call_id,
                type="function",
                function=ChoiceDeltaToolCallFunction(name=name, arguments=arguments),
            )
        if named and arguments:
            return ChoiceDeltaToolCall(index=0, function=ChoiceDeltaToolCallFunction(arguments=arguments))
        return None

    try:
        async for chunk in chunks:
            if chunk.choices:
                choice = chunk.choices[0]
                deltas = []
                if choice.delta.content is not None:
                    deltas.append(tool_delta(*translator.feed(choice.delta.content)))
                    choice.delta.content = None
                if choice.finish_reason:
                    deltas.append(tool_delta(*translator.finish()))
                    # A cap-truncated reply keeps "length", so the cut stays visible downstream.
                    if named and choice.finish_reason == "stop":
                        choice.finish_reason = "tool_calls"
                choice.delta.tool_calls = [d for d in deltas if d] or None
            yield chunk
    finally:
        # A reader that stops early closes this generator; the response underneath must close with it.
        close = getattr(chunks, "close", None) or getattr(chunks, "aclose", None)
        if close is not None:
            await close()
