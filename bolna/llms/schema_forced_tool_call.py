"""A forced tool call made through a JSON schema, for backends that cannot force tool_choice.

The reply is constrained to the call's JSON and re-streamed as the tool-call deltas the forced
tool_choice would have produced, so everything downstream reads it as an ordinary tool call.
"""

import json
import re
import uuid
from dataclasses import dataclass
from typing import AsyncIterator, Callable, Optional

from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall, ChoiceDeltaToolCallFunction

from bolna.enums import LLMProvider
from bolna.helpers.logger_config import configure_logger
from bolna.llms.routing_stream import REASONING_KEY, without_trailing_parameters

logger = configure_logger(__name__)

# Providers whose server accepts a forced tool_choice without enforcing it, so the model may answer instead.
SCHEMA_FORCED_TOOL_PROVIDERS = frozenset({LLMProvider.BOLNA.value})

# Longest routing rationale guided decoding allows before the decision.
ROUTING_REASONING_MAX_CHARS = 600

_ENVELOPE_HEAD = re.compile(r'\s*\{\s*"name"\s*:\s*"((?:[^"\\]|\\.)*)"\s*,\s*"arguments"\s*:\s*')
_TRAILING_CLOSE = re.compile(r"[}\s]*$")


@dataclass
class ForcedToolCall:
    response_format: dict
    # Builds the reader that turns the constrained reply back into tool-call deltas.
    translator: Callable
    # The tools as the model should read them, when they differ from the request's.
    tools: Optional[list] = None
    # Output the reply writes ahead of the call, allowed on top of the request's cap so it cannot cut the call.
    preamble_tokens: int = 0

    def stream(self, chunks) -> AsyncIterator:
        return _as_tool_call_deltas(chunks, self.translator())


def _reason_first_choice(tools: list) -> ForcedToolCall:
    """Pick among the tools with the rationale written first: {"reasoning", "decision": {"name", "arguments"?}}."""
    # A model still offered the rationale as a parameter writes it into an argument's value.
    tools = [without_trailing_parameters(t) for t in tools]
    options = []
    for tool in tools:
        params = tool["function"]["parameters"]
        option = {"name": {"const": tool["function"]["name"]}}
        if params["properties"]:
            option["arguments"] = {**params, "additionalProperties": False}
        options.append(
            {"type": "object", "properties": option, "required": list(option), "additionalProperties": False}
        )
    schema = {
        "type": "object",
        "properties": {
            REASONING_KEY: {"type": "string", "maxLength": ROUTING_REASONING_MAX_CHARS},
            "decision": {"anyOf": options},
        },
        "required": [REASONING_KEY, "decision"],
        "additionalProperties": False,
    }
    # A character costs at most about a token, so the rationale's length bounds its tokens.
    return ForcedToolCall(
        _response_format("routing_decision", schema), _ReasonFirst, tools, preamble_tokens=ROUTING_REASONING_MAX_CHARS
    )


def _forced_tool_call(tools, tool_choice, reason_first: bool = False) -> Optional[ForcedToolCall]:
    """The json_schema stand-in for a forced tool_choice, or None when the request forces nothing."""
    functions = {t["function"]["name"]: t["function"] for t in tools or []}
    if isinstance(tool_choice, dict):
        name = (tool_choice.get("function") or {}).get("name")
        if name not in functions:
            return None
        return ForcedToolCall(_response_format(name, _parameters(functions[name])), lambda: _Arguments(name))
    if tool_choice != "required" or not functions:
        return None
    if reason_first:
        return _reason_first_choice(tools)
    options = [
        {
            "type": "object",
            "properties": {"name": {"const": name}, "arguments": _parameters(fn)},
            "required": ["name", "arguments"],
            "additionalProperties": False,
        }
        for name, fn in functions.items()
    ]
    return ForcedToolCall(_response_format("tool_call", {"anyOf": options}), _Envelope)


def force_tools_by_schema(model_args: dict, reason_first: bool = False) -> Optional[ForcedToolCall]:
    """Swap a forced tool_choice for the JSON schema it implies; the tools stay, as the schema never reaches the model."""
    forced = _forced_tool_call(model_args.get("tools"), model_args.get("tool_choice"), reason_first)
    if forced:
        model_args["tool_choice"] = "auto"
        # A stop string inside an argument would cut the JSON short.
        model_args.pop("stop", None)
        model_args["response_format"] = forced.response_format
        if forced.tools is not None:
            model_args["tools"] = forced.tools
        if forced.preamble_tokens and model_args.get("max_tokens"):
            model_args["max_tokens"] += forced.preamble_tokens
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


class _ReasonFirst:
    """The reply is {"reasoning", "decision": {...}}; the decision comes last, so the call is whole only at the end."""

    def __init__(self):
        self.buffer = ""

    def feed(self, text: str) -> tuple[Optional[str], str]:
        self.buffer += text
        return None, ""

    def finish(self) -> tuple[Optional[str], str]:
        try:
            reply = json.loads(self.buffer)
            decision = reply["decision"]
            arguments = {**(decision.get("arguments") or {}), REASONING_KEY: reply.get(REASONING_KEY)}
            return decision["name"], json.dumps(arguments)
        except (json.JSONDecodeError, KeyError, TypeError):
            logger.error(f"Reason-first routing reply was not a valid decision: {self.buffer[:200]!r}")
            return None, ""


async def _as_tool_call_deltas(chunks, translator):
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
                # A server whose parser reads the JSON as a call streams it natively; that passes through.
                choice.delta.tool_calls = [*(choice.delta.tool_calls or []), *(d for d in deltas if d)] or None
            yield chunk
    finally:
        # A reader that stops early closes this generator; the response underneath must close with it.
        close = getattr(chunks, "close", None) or getattr(chunks, "aclose", None)
        if close is not None:
            await close()
