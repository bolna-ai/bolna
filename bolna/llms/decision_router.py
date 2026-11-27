"""Graph routing on a decision model: a hop's edges are one choice question, weighed together instead of decoded."""

import asyncio
import json
import time
from typing import Optional

from bolna.helpers.logger_config import configure_logger
from bolna.llms.decision_model import Decision, DecisionError, DecisionModel
from bolna.llms.routing_stream import CONFIDENCE_KEY, TRAILING_KEYS

logger = configure_logger(__name__)

INSTRUCTIONS = (
    "Select the one edge that matches the caller's latest completed response. "
    "If no edge matches, stay on the current node."
)
_SPEAKER = {"assistant": "Agent", "user": "Caller", "tool": "Tool result"}


def routing_context(messages: list) -> str:
    """The routing prompt as one context: its system prompt, then the turns the router was given."""
    system = "\n".join(m.get("content") or "" for m in messages if m["role"] == "system")
    turns = "\n".join(
        f"{_SPEAKER.get(m['role'], m['role'])}: {(m.get('content') or '').strip()}"
        for m in messages
        if m["role"] != "system"
    )
    return f"{system}\n\n# Conversation so far\n{turns}"


def edge_choices(tools: list) -> dict:
    """Each edge's routing tool, by name, with its description as what choosing it means."""
    return {t["function"]["name"]: t["function"].get("description") or t["function"]["name"] for t in tools}


def _has_own_parameters(edge: dict) -> bool:
    properties = (edge["function"].get("parameters") or {}).get("properties") or {}
    return any(k not in TRAILING_KEYS for k in properties)


class DecisionRouter:
    """Routes on a decision model; the LLM router fills edge parameters and routes the hops it cannot."""

    def __init__(self, decision_model: DecisionModel, llm_router, llm_router_identity: tuple):
        self.decision_model = decision_model
        self.llm_router = llm_router
        # (provider, model, reasoning effort), as the graph agent reports a routing call.
        self.llm_router_identity = llm_router_identity
        self._decides = decision_model.available

    async def route(self, messages, tools, tool_choice="required", meta_info=None):
        tools = json.loads(tools) if isinstance(tools, str) else tools
        decision = await self._decide(messages, tools) if self._decides else None
        if decision is None:
            routed = await self.llm_router.route(messages, tools, tool_choice, meta_info)
            # Marked so the hop is reported under the router that answered it.
            return routed and {**routed, "routed_by": self.llm_router_identity}
        parameters, tail = await self._edge_parameters(messages, tools, decision, meta_info)
        return {
            "function_name": decision.choice,
            "arguments": {**parameters, CONFIDENCE_KEY: decision.confidence},
            "usage": decision.usage,
            "service_tier": None,
            "overflowed": False,
            "routing_tail": tail,
        }

    async def _decide(self, messages, tools) -> Optional[Decision]:
        provider = self.decision_model.provider.value
        try:
            return await self.decision_model.choose(routing_context(messages), INSTRUCTIONS, edge_choices(tools))
        except DecisionError as e:
            logger.warning(f"No routing decision from {provider}, routing this hop on the LLM router: {e}")
        except Exception as e:
            # A failing API costs every hop its timeout, so the rest of the call routes on the LLM router.
            self._decides = False
            logger.error(f"Decision routing on {provider} failed, routing the call on the LLM router: {e!r}")
        return None

    async def _edge_parameters(self, messages, tools, decision: Decision, meta_info) -> tuple:
        """The chosen edge's own parameters from the LLM router, and the tail that reports that call."""
        name = decision.choice
        if not _has_own_parameters(next(t for t in tools if t["function"]["name"] == name)):
            return {}, None
        # Forced on the whole routing request: offered the edge alone, the LLM copies the caller's words into it.
        forced = {"type": "function", "function": {"name": name}}
        started = time.perf_counter()
        filled = await self.llm_router.route(messages, tools, forced, meta_info)
        latency_ms = round((time.perf_counter() - started) * 1000, 1)
        if not filled or filled.get("function_name") != name:
            logger.warning(f"Edge parameters for {name} could not be filled; transitioning without them")
            return {}, None
        parameters = {k: v for k, v in filled["arguments"].items() if k not in TRAILING_KEYS}
        return parameters, asyncio.create_task(self._hop_tail(decision, filled, latency_ms))

    async def _hop_tail(self, decision: Decision, filled: dict, fill_latency_ms: float) -> dict:
        """The hop's usage, and under `parameter_fill` the LLM call that filled its parameters."""
        fill_tail = await filled["routing_tail"] if filled.get("routing_tail") else {}
        provider, model, _ = self.llm_router_identity
        fill = {
            "routing_provider": provider,
            "routing_model": model,
            "latency_ms": fill_latency_ms,
            "usage": fill_tail.get("usage") or filled.get("usage") or {},
        }
        return {"usage": decision.usage, "parameter_fill": fill}
