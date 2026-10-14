"""Reads a streamed routing tool call, resolving the decision before the trailing fields.

Under `tool_choice: "required"` the function name streams before any argument, so the
decision is complete once the function name and the edge's own parameters have arrived.
"""

import asyncio
import json
from typing import Optional

from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)

# CPython may collect a task nobody holds, so keep a reference until it finishes. A hop that
# is never built still drains its stream rather than being cancelled mid-flight.
_TAILS: set = set()

REASONING_KEY = "reasoning"
CONFIDENCE_KEY = "confidence"
# Emitted by the router for observability; neither is read to make the routing decision.
TRAILING_KEYS = (REASONING_KEY, CONFIDENCE_KEY)
_TRAILING_TOKENS = tuple(f'"{k}"' for k in TRAILING_KEYS)


def _args_before_trailing(arguments: str) -> Optional[dict]:
    """Arguments emitted before the first trailing key, or None if none has arrived yet."""
    cuts = [arguments.index(token) for token in _TRAILING_TOKENS if token in arguments]
    if not cuts:
        return None
    try:
        return json.loads(arguments[: min(cuts)].rstrip().rstrip(",") + "}")
    except json.JSONDecodeError:
        return None


def _usage_of(chunk) -> dict:
    u = chunk.usage
    usage = {"input_tokens": u.prompt_tokens, "output_tokens": u.completion_tokens}
    if getattr(u, "completion_tokens_details", None):
        usage["reasoning_tokens"] = getattr(u.completion_tokens_details, "reasoning_tokens", None)
    if getattr(u, "prompt_tokens_details", None):
        usage["cached_tokens"] = getattr(u.prompt_tokens_details, "cached_tokens", None)
    return usage


class RoutingStreamReader:
    """Reads a streamed forced tool call, resolving the decision before the rationale."""

    def __init__(self, stream, tools: list):
        self._stream = stream
        self._iter = stream.__aiter__()
        self._required = {
            t["function"]["name"]: [
                k for k in (t["function"].get("parameters") or {}).get("required", []) if k not in TRAILING_KEYS
            ]
            for t in tools
        }
        self._index: Optional[int] = None
        self.function_name: Optional[str] = None
        self.arguments = ""
        self.service_tier = None
        self.usage: dict = {}
        self.early = False

    def _consume(self, chunk) -> None:
        if getattr(chunk, "service_tier", None):
            self.service_tier = chunk.service_tier
        if getattr(chunk, "usage", None):
            self.usage = _usage_of(chunk)
        if not chunk.choices:
            return
        for tool_call in getattr(chunk.choices[0].delta, "tool_calls", None) or []:
            if self._index is None:
                self._index = tool_call.index
            # parallel_tool_calls is off, but a provider that ignores it would otherwise
            # concatenate a second call's arguments onto the first and corrupt both.
            if tool_call.index != self._index:
                continue
            if tool_call.function.name:
                self.function_name = tool_call.function.name
            self.arguments += tool_call.function.arguments or ""

    def _decision(self) -> Optional[dict]:
        if not self.function_name:
            return None
        parsed = _args_before_trailing(self.arguments)
        # A model that emits the trailing keys first leaves edge parameters missing; keep draining.
        if parsed is None or any(k not in parsed for k in self._required.get(self.function_name, [])):
            return None
        return parsed

    def _parse_all(self) -> Optional[dict]:
        try:
            return json.loads(self.arguments) if self.arguments else {}
        except json.JSONDecodeError:
            logger.error(f"Routing arguments were not valid JSON: {self.arguments!r}")
            return None

    async def decide(self) -> Optional[dict]:
        """Iterate only as far as the routing decision. Returns the parsed arguments, or None."""
        async for chunk in self._iter:
            self._consume(chunk)
            decision = self._decision()
            if decision is not None:
                self.early = True
                return decision
        return self._parse_all() if self.function_name else None

    async def _close(self) -> None:
        closer = getattr(self._stream, "close", None) or getattr(self._stream, "aclose", None)
        if closer is None:
            return
        try:
            await closer()
        except Exception as e:
            logger.warning(f"Could not close the routing stream: {e}")

    async def finish(self) -> dict:
        """Drain the remainder for the trailing fields and the usage record in the last chunk."""
        try:
            async for chunk in self._iter:
                self._consume(chunk)
        except Exception as e:
            # Observability only: a stream that dies here costs the rationale, nothing else.
            logger.warning(f"Routing stream ended before the rationale: {e}")
        finally:
            # Reached on cancellation too, where abandoning the iterator would leak the response.
            await self._close()
        tail = {"usage": self.usage}
        full = self._parse_all() or {}
        return {**tail, **{k: full[k] for k in TRAILING_KEYS if k in full}}


async def read_routing_stream(stream, tools: list, overflowed: bool = False) -> Optional[dict]:
    """Resolve a streamed routing decision, leaving `routing_tail` to carry the rest."""
    reader = RoutingStreamReader(stream, tools)
    arguments = await reader.decide()
    if arguments is None:
        return None
    tail = None
    if reader.early:
        tail = asyncio.create_task(reader.finish())
        _TAILS.add(tail)
        tail.add_done_callback(_TAILS.discard)
    return {
        "function_name": reader.function_name,
        "arguments": arguments,
        "usage": reader.usage,
        "service_tier": reader.service_tier,
        "overflowed": overflowed,
        "routing_tail": tail,
    }
