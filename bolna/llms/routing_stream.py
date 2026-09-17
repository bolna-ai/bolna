"""Early-dispatch reader for the streamed graph-routing tool call.

The routing hop is serial in front of the answering LLM, so what matters is time to
decision, not time to last token. Under `tool_choice: "required"` the function name is
streamed before any argument, and `reasoning` and `confidence` are observability rather
than routing inputs. The decision is therefore complete once the function name and the
edge's own parameters have arrived. This reader returns at that point and drains the rest
in the background, so the rationale stays available for debugging without being on the
critical path, and the request on the wire is unchanged.
"""

import asyncio
import json
from typing import Optional

from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)

REASONING_KEY = "reasoning"
CONFIDENCE_KEY = "confidence"
# Emitted by the router for observability; neither is read to make the routing decision.
_TRAILING_KEYS = (REASONING_KEY, CONFIDENCE_KEY)
_TRAILING_TOKENS = tuple(f'"{k}"' for k in _TRAILING_KEYS)


def _cut_index(arguments: str) -> int:
    """Offset of the first observability key in the streamed arguments, or -1."""
    found = [arguments.index(tok) for tok in _TRAILING_TOKENS if tok in arguments]
    return min(found) if found else -1


def _close_prefix(arguments: str, cut: int) -> Optional[dict]:
    """Parse the arguments emitted before the cut by closing the truncated object."""
    head = arguments[:cut].rstrip().rstrip(",")
    try:
        return json.loads(head + "}")
    except json.JSONDecodeError:
        return None


def _usage_from(completion) -> dict:
    u = getattr(completion, "usage", None)
    if not u:
        return {}
    usage = {"input_tokens": u.prompt_tokens, "output_tokens": u.completion_tokens}
    details = getattr(u, "completion_tokens_details", None)
    if details:
        usage["reasoning_tokens"] = getattr(details, "reasoning_tokens", None)
    prompt_details = getattr(u, "prompt_tokens_details", None)
    if prompt_details:
        usage["cached_tokens"] = getattr(prompt_details, "cached_tokens", None)
    return usage


class RoutingStreamReader:
    """Consumes a streamed forced tool call, resolving the decision before the rationale.

    `decide()` iterates only as far as the routing decision. `finish()` resumes the same
    iterator to collect `reasoning` and the usage record that arrives in the final chunk.
    """

    def __init__(self, stream, tools: list, overflowed: bool = False):
        self._iter = stream.__aiter__()
        self._required = {
            t["function"]["name"]: [
                k for k in (t["function"].get("parameters") or {}).get("required", []) if k not in _TRAILING_KEYS
            ]
            for t in tools
        }
        self.overflowed = overflowed
        self.function_name: Optional[str] = None
        self.arguments = ""
        self.service_tier = None
        self.usage: dict = {}
        self.early = False
        self.exhausted = False

    def _consume(self, chunk) -> None:
        if getattr(chunk, "service_tier", None):
            self.service_tier = chunk.service_tier
        if getattr(chunk, "usage", None):
            self.usage = _usage_from(chunk)
        if not chunk.choices:
            return
        delta = chunk.choices[0].delta
        for tool_call in getattr(delta, "tool_calls", None) or []:
            if tool_call.function.name:
                self.function_name = tool_call.function.name
            self.arguments += tool_call.function.arguments or ""

    def _resolvable(self) -> Optional[dict]:
        """Arguments parsed from the prefix before the observability keys, or None."""
        if not self.function_name:
            return None
        cut = _cut_index(self.arguments)
        if cut < 0:
            return None
        parsed = _close_prefix(self.arguments, cut)
        if parsed is None:
            return None
        # A model that emits the observability keys first leaves edge parameters missing;
        # keep draining rather than dispatching on a half-formed decision.
        if any(k not in parsed for k in self._required.get(self.function_name, [])):
            return None
        return parsed

    async def decide(self) -> Optional[dict]:
        """Iterate until the decision is complete. Returns the parsed arguments, or None."""
        async for chunk in self._iter:
            self._consume(chunk)
            parsed = self._resolvable()
            if parsed is not None:
                self.early = True
                return parsed
        self.exhausted = True
        if not self.function_name:
            return None
        try:
            return json.loads(self.arguments) if self.arguments else {}
        except json.JSONDecodeError:
            logger.error(f"Routing arguments were not valid JSON: {self.arguments!r}")
            return None

    async def finish(self) -> dict:
        """Drain the remainder for the rationale, confidence and the final usage record."""
        if not self.exhausted:
            async for chunk in self._iter:
                self._consume(chunk)
            self.exhausted = True
        tail = {"usage": self.usage}
        try:
            full = json.loads(self.arguments) if self.arguments else {}
        except json.JSONDecodeError:
            logger.warning(f"Routing tail arguments were not valid JSON: {self.arguments!r}")
            return tail
        for key in _TRAILING_KEYS:
            if key in full:
                tail[key] = full[key]
        return tail


async def read_routing_stream(stream, tools: list, overflowed: bool = False) -> Optional[dict]:
    """Resolve a streamed routing decision, draining the rationale off the critical path.

    The returned `routing_tail` task carries `reasoning` and the usage record; it is None
    when the stream had already ended by the time the decision was complete.
    """
    reader = RoutingStreamReader(stream, tools, overflowed)
    arguments = await reader.decide()
    if arguments is None:
        return None

    tail = None
    if reader.early:
        tail = asyncio.create_task(reader.finish())
    return {
        "function_name": reader.function_name,
        "arguments": arguments,
        "usage": reader.usage,
        "service_tier": reader.service_tier,
        "overflowed": reader.overflowed,
        "routing_tail": tail,
        "decided_early": reader.early,
    }
