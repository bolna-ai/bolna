"""Tests for router nodes (silent deterministic dispatch) in GraphAgent.

Covers:
  * config validation (GraphNode + GraphAgentConfig): router constraints and cycle rejection
  * _resolve_router_chain: single hop, catch-all fallback, priority order, multi-hop,
    runtime cycle bounding, hop routing_info shape
  * generate(): transition into a router resolves to a speaking node, entry-node dispatch
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from pydantic import ValidationError

from bolna.agent_types.graph_agent import GraphAgent, _DETERMINISTIC_REASONING_PREFIX
from bolna.models import GraphNode, GraphAgentConfig
from bolna.enums import NodeType
from bolna.llms.types import LLMStreamChunk


# ---------------------------------------------------------------------------
# Agent construction helper (mirrors test_expression_routing.py)
# ---------------------------------------------------------------------------


async def _async_iter(items):
    for item in items:
        yield item


def _make_agent(config):
    mock_llm = MagicMock()
    mock_llm.generate_stream = MagicMock(side_effect=lambda *a, **k: _async_iter([]))
    mock_llm.trigger_function_call = False
    mock_openai_client = MagicMock()
    mock_openai_llm_cls = MagicMock(return_value=mock_llm)

    with (
        patch("bolna.agent_types.graph_agent.OpenAI", return_value=mock_openai_client),
        patch("bolna.agent_types.graph_agent.SUPPORTED_LLM_PROVIDERS", {"openai": mock_openai_llm_cls}),
        patch("bolna.agent_types.graph_agent.OpenAiLLM", return_value=MagicMock()),
    ):
        agent = GraphAgent(config)

    agent._mock_llm = mock_llm
    return agent


def _base_config(nodes, current_node_id, **overrides):
    cfg = {
        "agent_information": "Test agent",
        "model": "gpt-4o-mini",
        "provider": "openai",
        "temperature": 0.7,
        "max_tokens": 150,
        "current_node_id": current_node_id,
        "nodes": nodes,
    }
    cfg.update(overrides)
    return cfg


def _expr(variable, operator, value=None):
    cond = {"variable": variable, "operator": operator}
    if value is not None:
        cond["value"] = value
    return {"logic": "and", "conditions": [cond]}


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestRouterNodeValidation:
    def test_valid_router_node(self):
        node = GraphNode(
            id="dispatch",
            node_type=NodeType.ROUTER,
            edges=[
                {
                    "to_node_id": "hindi",
                    "condition_type": "expression",
                    "expression": _expr("detected_language", "eq", "hi"),
                },
                {"to_node_id": "default", "condition_type": "unconditional"},
            ],
        )
        assert node.node_type == NodeType.ROUTER

    def test_router_with_prompt_rejected(self):
        with pytest.raises(ValidationError, match="never speaks"):
            GraphNode(
                id="dispatch",
                node_type=NodeType.ROUTER,
                prompt="say hi",
                edges=[{"to_node_id": "a", "condition_type": "unconditional"}],
            )

    def test_router_with_static_message_rejected(self):
        with pytest.raises(ValidationError, match="never speaks"):
            GraphNode(
                id="dispatch",
                node_type=NodeType.ROUTER,
                static_message="hi",
                edges=[{"to_node_id": "a", "condition_type": "unconditional"}],
            )

    def test_router_with_intent_edge_allowed(self):
        node = GraphNode(
            id="dispatch",
            node_type=NodeType.ROUTER,
            edges=[
                {"to_node_id": "a", "condition": "user wants a"},  # no condition_type -> llm/intent
                {"to_node_id": "b", "condition_type": "unconditional"},
            ],
        )
        assert node.node_type == NodeType.ROUTER

    def test_router_with_event_edge_rejected(self):
        with pytest.raises(ValidationError, match="event edge"):
            GraphNode(
                id="dispatch",
                node_type=NodeType.ROUTER,
                edges=[
                    {"to_node_id": "a", "condition_type": "event", "event_name": "ping"},
                    {"to_node_id": "b", "condition_type": "unconditional"},
                ],
            )

    def test_router_without_catch_all_rejected(self):
        with pytest.raises(ValidationError, match="catch-all"):
            GraphNode(
                id="dispatch",
                node_type=NodeType.ROUTER,
                edges=[
                    {
                        "to_node_id": "a",
                        "condition_type": "expression",
                        "expression": _expr("x", "eq", "1"),
                    },
                ],
            )

    def test_non_router_node_unconstrained(self):
        # A normal LLM node keeps its prompt and llm edges.
        node = GraphNode(id="greeting", prompt="Greet.", edges=[{"to_node_id": "a", "condition": "wants a"}])
        assert node.node_type == NodeType.LLM

    def test_router_cycle_rejected(self):
        nodes = [
            {
                "id": "r1",
                "node_type": "router",
                "edges": [{"to_node_id": "r2", "condition_type": "unconditional"}],
            },
            {
                "id": "r2",
                "node_type": "router",
                "edges": [{"to_node_id": "r1", "condition_type": "unconditional"}],
            },
        ]
        with pytest.raises(ValidationError, match="cycle"):
            GraphAgentConfig(**_base_config(nodes, "r1"))

    def test_router_edge_to_unknown_node_rejected(self):
        nodes = [
            {
                "id": "r1",
                "node_type": "router",
                "edges": [{"to_node_id": "does_not_exist", "condition_type": "unconditional"}],
            },
        ]
        with pytest.raises(ValidationError, match="unknown node"):
            GraphAgentConfig(**_base_config(nodes, "r1"))

    def test_router_chain_terminating_ok(self):
        nodes = [
            {
                "id": "r1",
                "node_type": "router",
                "edges": [{"to_node_id": "r2", "condition_type": "unconditional"}],
            },
            {
                "id": "r2",
                "node_type": "router",
                "edges": [{"to_node_id": "leaf", "condition_type": "unconditional"}],
            },
            {"id": "leaf", "prompt": "done", "edges": []},
        ]
        cfg = GraphAgentConfig(**_base_config(nodes, "r1"))
        assert len(cfg.nodes) == 3


# ---------------------------------------------------------------------------
# _resolve_router_chain
# ---------------------------------------------------------------------------


def _router_agent(current="dispatch", **ctx):
    nodes = [
        {
            "id": "dispatch",
            "node_type": "router",
            "edges": [
                {
                    "to_node_id": "hindi",
                    "condition": "hindi speaker",
                    "condition_type": "expression",
                    "expression": _expr("detected_language", "eq", "hi"),
                },
                {"to_node_id": "english", "condition": "default", "condition_type": "unconditional"},
            ],
        },
        {"id": "hindi", "prompt": "Respond in Hindi.", "edges": []},
        {"id": "english", "prompt": "Respond in English.", "edges": []},
    ]
    cfg = _base_config(nodes, current, context_data=ctx or {})
    return _make_agent(cfg)


class TestResolveRouterChain:
    @pytest.mark.asyncio
    async def test_single_hop_expression_match(self):
        agent = _router_agent(detected_language="hi")
        hops = await agent._resolve_router_chain([])
        assert agent.current_node_id == "hindi"
        assert len(hops) == 1
        assert hops[0]["previous_node"] == "dispatch"
        assert hops[0]["current_node"] == "hindi"
        assert hops[0]["routing_type"] == "deterministic"
        assert hops[0]["reasoning"].startswith(_DETERMINISTIC_REASONING_PREFIX)
        assert hops[0]["confidence"] == 1.0
        assert hops[0]["node_type"] == NodeType.LLM

    @pytest.mark.asyncio
    async def test_catch_all_fallback_when_no_expression_matches(self):
        agent = _router_agent(detected_language="fr")
        hops = await agent._resolve_router_chain([])
        assert agent.current_node_id == "english"
        assert len(hops) == 1
        assert hops[0]["current_node"] == "english"

    @pytest.mark.asyncio
    async def test_catch_all_wins_even_if_listed_first(self):
        # The unconditional is listed BEFORE the expression; it must still be a
        # fallback, not pre-empt a matching expression edge.
        nodes = [
            {
                "id": "dispatch",
                "node_type": "router",
                "edges": [
                    {"to_node_id": "fallback", "condition_type": "unconditional", "priority": 0},
                    {
                        "to_node_id": "vip",
                        "condition_type": "expression",
                        "priority": 1,
                        "expression": _expr("tier", "eq", "vip"),
                    },
                ],
            },
            {"id": "fallback", "prompt": "f", "edges": []},
            {"id": "vip", "prompt": "v", "edges": []},
        ]
        agent = _make_agent(_base_config(nodes, "dispatch", context_data={"tier": "vip"}))
        await agent._resolve_router_chain([])
        assert agent.current_node_id == "vip"

    @pytest.mark.asyncio
    async def test_expression_priority_order(self):
        nodes = [
            {
                "id": "dispatch",
                "node_type": "router",
                "edges": [
                    {
                        "to_node_id": "low",
                        "condition_type": "expression",
                        "priority": 10,
                        "expression": _expr("flag", "eq", "on"),
                    },
                    {
                        "to_node_id": "high",
                        "condition_type": "expression",
                        "priority": 1,
                        "expression": _expr("flag", "eq", "on"),
                    },
                    {"to_node_id": "default", "condition_type": "unconditional"},
                ],
            },
            {"id": "low", "prompt": "l", "edges": []},
            {"id": "high", "prompt": "h", "edges": []},
            {"id": "default", "prompt": "d", "edges": []},
        ]
        agent = _make_agent(_base_config(nodes, "dispatch", context_data={"flag": "on"}))
        await agent._resolve_router_chain([])
        assert agent.current_node_id == "high"  # lower priority number wins

    @pytest.mark.asyncio
    async def test_multi_hop_chain(self):
        nodes = [
            {
                "id": "r1",
                "node_type": "router",
                "edges": [{"to_node_id": "r2", "condition_type": "unconditional"}],
            },
            {
                "id": "r2",
                "node_type": "router",
                "edges": [{"to_node_id": "leaf", "condition_type": "unconditional"}],
            },
            {"id": "leaf", "prompt": "done", "edges": []},
        ]
        agent = _make_agent(_base_config(nodes, "r1"))
        hops = await agent._resolve_router_chain([])
        assert agent.current_node_id == "leaf"
        assert [h["current_node"] for h in hops] == ["r2", "leaf"]
        assert agent.node_history[-1] == "leaf"

    @pytest.mark.asyncio
    async def test_runtime_cycle_is_bounded(self):
        # Build a cyclic router graph directly as a dict (bypasses Pydantic
        # validation) to prove the visited-set prevents an infinite loop.
        nodes = [
            {
                "id": "r1",
                "node_type": "router",
                "edges": [{"to_node_id": "r2", "condition_type": "unconditional"}],
            },
            {
                "id": "r2",
                "node_type": "router",
                "edges": [{"to_node_id": "r1", "condition_type": "unconditional"}],
            },
        ]
        agent = _make_agent(_base_config(nodes, "r1"))
        hops = await agent._resolve_router_chain([])  # must return, not hang
        assert len(hops) == 2  # r1->r2, r2->r1, then r1 already visited -> stop

    @pytest.mark.asyncio
    async def test_no_hops_when_current_not_router(self):
        nodes = [{"id": "leaf", "prompt": "hi", "edges": []}]
        agent = _make_agent(_base_config(nodes, "leaf"))
        hops = await agent._resolve_router_chain([])
        assert hops == []
        assert agent.current_node_id == "leaf"


# ---------------------------------------------------------------------------
# generate() integration
# ---------------------------------------------------------------------------


async def _collect(agen):
    return [item async for item in agen]


class TestGenerateWithRouter:
    @pytest.mark.asyncio
    async def test_transition_into_router_resolves_and_speaks(self):
        nodes = [
            {
                "id": "greeting",
                "prompt": "Greet.",
                "edges": [{"to_node_id": "dispatch", "condition_type": "unconditional"}],
            },
            {
                "id": "dispatch",
                "node_type": "router",
                "edges": [
                    {
                        "to_node_id": "hindi",
                        "condition_type": "expression",
                        "expression": _expr("detected_language", "eq", "hi"),
                    },
                    {"to_node_id": "english", "condition_type": "unconditional"},
                ],
            },
            {"id": "hindi", "prompt": "Hindi.", "edges": []},
            {"id": "english", "prompt": "English.", "edges": []},
        ]
        agent = _make_agent(_base_config(nodes, "greeting", context_data={"detected_language": "hi"}))

        out = await _collect(agent.generate([{"role": "user", "content": "hello"}]))
        routing = [o["routing_info"] for o in out if isinstance(o, dict) and "routing_info" in o]

        # greeting -> dispatch (unconditional), then dispatch -> hindi (router hop)
        assert agent.current_node_id == "hindi"
        assert routing[-1]["current_node"] == "hindi"
        assert routing[-1]["node_type"] == NodeType.LLM
        # a router hop was emitted with deterministic routing
        assert any(r["current_node"] == "hindi" and r["routing_type"] == "deterministic" for r in routing)
        # the terminal (speaking) node produced a messages payload, i.e. it did not stay silent
        assert any(isinstance(o, dict) and "messages" in o for o in out)

    @pytest.mark.asyncio
    async def test_unresolvable_router_stays_silent(self):
        # A router that matches no expression and has no catch-all (built as a raw
        # dict, bypassing validation) must not fall through to speaking.
        nodes = [
            {
                "id": "dispatch",
                "node_type": "router",
                "edges": [
                    {
                        "to_node_id": "hindi",
                        "condition_type": "expression",
                        "expression": _expr("detected_language", "eq", "hi"),
                    },
                ],
            },
            {"id": "hindi", "prompt": "Hindi.", "edges": []},
        ]
        agent = _make_agent(_base_config(nodes, "dispatch", context_data={"detected_language": "fr"}))

        out = await _collect(agent.generate([{"role": "user", "content": "hi"}]))
        assert agent.current_node_id == "dispatch"  # never left the router
        assert not any(isinstance(o, dict) and "messages" in o for o in out)  # did not speak
        # ends the turn with a terminal end-of-stream chunk so the pipeline unwinds cleanly
        assert any(isinstance(o, LLMStreamChunk) and o.end_of_stream for o in out)

    @pytest.mark.asyncio
    async def test_entry_router_resolves_on_first_generate(self):
        nodes = [
            {
                "id": "entry",
                "node_type": "router",
                "edges": [
                    {
                        "to_node_id": "vip",
                        "condition_type": "expression",
                        "expression": _expr("tier", "eq", "vip"),
                    },
                    {"to_node_id": "standard", "condition_type": "unconditional"},
                ],
            },
            {"id": "vip", "prompt": "VIP.", "edges": []},
            {"id": "standard", "prompt": "Standard.", "edges": []},
        ]
        agent = _make_agent(_base_config(nodes, "entry", context_data={"tier": "vip"}))
        assert agent.current_node_id == "entry"  # not resolved at construction

        out = await _collect(agent.generate([{"role": "user", "content": "hi"}]))
        routing = [o["routing_info"] for o in out if isinstance(o, dict) and "routing_info" in o]

        assert agent.current_node_id == "vip"
        assert any(r["previous_node"] == "entry" and r["current_node"] == "vip" for r in routing)


# ---------------------------------------------------------------------------
# Intent edges on routers
# ---------------------------------------------------------------------------


def _intent_router_agent(**ctx):
    nodes = [
        {
            "id": "dispatch",
            "node_type": "router",
            "description": "Route the caller to the right department.",
            "edges": [
                {
                    "to_node_id": "hindi",
                    "condition_type": "expression",
                    "expression": _expr("detected_language", "eq", "hi"),
                },
                {"to_node_id": "billing", "condition": "user asks about billing", "function_name": "go_to_billing"},
                {"to_node_id": "support", "condition": "user needs technical support"},
                {"to_node_id": "general", "condition_type": "unconditional"},
            ],
        },
        {"id": "hindi", "prompt": "Hindi.", "edges": []},
        {"id": "billing", "prompt": "Billing.", "edges": []},
        {"id": "support", "prompt": "Support.", "edges": []},
        {"id": "general", "prompt": "General.", "edges": []},
    ]
    return _make_agent(_base_config(nodes, "dispatch", context_data=ctx or {}))


class TestRouterIntentRouting:
    @pytest.mark.asyncio
    async def test_expression_match_skips_intent_llm(self):
        agent = _intent_router_agent(detected_language="hi")
        with patch.object(agent, "_decide_next_node_llm", new_callable=AsyncMock) as mock_llm:
            await agent._resolve_router_chain([{"role": "user", "content": "namaste"}])
        mock_llm.assert_not_called()
        assert agent.current_node_id == "hindi"

    @pytest.mark.asyncio
    async def test_intent_routes_when_no_expression_matches(self):
        agent = _intent_router_agent(detected_language="en")
        picked = (
            "billing",
            {"topic": "invoice"},
            320.0,
            [{"role": "system", "content": "routing"}],
            [{"type": "function"}],
            "asks about an invoice",
            0.9,
            {"input_tokens": 12},
        )
        with patch.object(agent, "_decide_next_node_llm", new_callable=AsyncMock, return_value=picked) as mock_llm:
            hops = await agent._resolve_router_chain([{"role": "user", "content": "my invoice is wrong"}])

        mock_llm.assert_called_once()
        offered = mock_llm.call_args.args[1]  # only intent edges go to the routing LLM
        assert {e["to_node_id"] for e in offered} == {"billing", "support"}
        assert agent.current_node_id == "billing"
        assert agent.context_data["topic"] == "invoice"
        assert len(hops) == 1
        assert hops[0]["routing_type"] == "llm"
        assert hops[0]["routing_messages"] is not None
        assert hops[0]["extracted_params"] == {"topic": "invoice"}
        assert hops[0]["confidence"] == 0.9

    @pytest.mark.asyncio
    async def test_intent_no_match_falls_to_catch_all(self):
        agent = _intent_router_agent(detected_language="en")
        stay = (None, None, 250.0, [{"role": "system", "content": "routing"}], [], "unclear", 0.3, None)
        with patch.object(agent, "_decide_next_node_llm", new_callable=AsyncMock, return_value=stay) as mock_llm:
            hops = await agent._resolve_router_chain([{"role": "user", "content": "ummm"}])

        mock_llm.assert_called_once()
        assert agent.current_node_id == "general"
        assert hops[0]["routing_type"] == "deterministic"
        assert "intent: no match" in hops[0]["routing_expression"]
        # the intent call's telemetry is carried onto the catch-all hop, not dropped
        assert hops[0]["routing_messages"] is not None
        assert hops[0]["routing_model"] is not None

    @pytest.mark.asyncio
    async def test_one_intent_call_per_chain(self):
        # r1 routes via intent to r2 (another intent router); r2 must take its
        # catch-all instead of making a second LLM call.
        nodes = [
            {
                "id": "r1",
                "node_type": "router",
                "edges": [
                    {"to_node_id": "r2", "condition": "user wants a specialist"},
                    {"to_node_id": "general", "condition_type": "unconditional"},
                ],
            },
            {
                "id": "r2",
                "node_type": "router",
                "edges": [
                    {"to_node_id": "billing", "condition": "billing question"},
                    {"to_node_id": "support", "condition_type": "unconditional"},
                ],
            },
            {"id": "billing", "prompt": "b", "edges": []},
            {"id": "support", "prompt": "s", "edges": []},
            {"id": "general", "prompt": "g", "edges": []},
        ]
        agent = _make_agent(_base_config(nodes, "r1"))
        pick_r2 = ("r2", None, 300.0, [{"role": "system", "content": "r"}], [], "specialist", 0.8, None)
        with patch.object(agent, "_decide_next_node_llm", new_callable=AsyncMock, return_value=pick_r2) as mock_llm:
            hops = await agent._resolve_router_chain([{"role": "user", "content": "hi"}])

        assert mock_llm.call_count == 1
        assert agent.current_node_id == "support"  # r2's catch-all, no second call
        assert [h["current_node"] for h in hops] == ["r2", "support"]


# ---------------------------------------------------------------------------
# Regression fixes (post-review)
# ---------------------------------------------------------------------------


class TestRouterFixes:
    @pytest.mark.asyncio
    async def test_node_turns_is_zero_on_router_entry(self):
        # A stuck-detection edge (_node_turns gte 1) on a router must NOT fire the
        # instant the router is entered mid-turn: node turns are 0 on arrival.
        nodes = [
            {
                "id": "chat",
                "prompt": "Chat.",
                "edges": [{"to_node_id": "dispatch", "condition_type": "unconditional"}],
            },
            {
                "id": "dispatch",
                "node_type": "router",
                "edges": [
                    {
                        "to_node_id": "escalation",
                        "condition_type": "expression",
                        "expression": _expr("_node_turns", "gte", 1),
                    },
                    {"to_node_id": "normal", "condition_type": "unconditional"},
                ],
            },
            {"id": "escalation", "prompt": "Esc.", "edges": []},
            {"id": "normal", "prompt": "Normal.", "edges": []},
        ]
        agent = _make_agent(_base_config(nodes, "chat"))
        history = [{"role": "user", "content": f"m{i}"} for i in range(3)] + [{"role": "assistant", "content": "hi"}]
        await _collect(agent.generate(history))
        assert agent.current_node_id == "normal"  # not "escalation"

    @pytest.mark.asyncio
    async def test_entry_router_landed_node_speaks_without_yank(self):
        # The node an entry router resolves to must speak this turn, not be re-routed
        # away by its own unconditional out-edge.
        nodes = [
            {
                "id": "entry",
                "node_type": "router",
                "edges": [
                    {"to_node_id": "vip", "condition_type": "expression", "expression": _expr("tier", "eq", "vip")},
                    {"to_node_id": "standard", "condition_type": "unconditional"},
                ],
            },
            {"id": "vip", "prompt": "VIP.", "edges": [{"to_node_id": "wrapup", "condition_type": "unconditional"}]},
            {"id": "standard", "prompt": "Std.", "edges": []},
            {"id": "wrapup", "prompt": "Wrap.", "edges": []},
        ]
        agent = _make_agent(_base_config(nodes, "entry", context_data={"tier": "vip"}))
        out = await _collect(agent.generate([{"role": "user", "content": "hi"}]))
        assert agent.current_node_id == "vip"  # not "wrapup"
        assert any(isinstance(o, dict) and "messages" in o for o in out)  # vip spoke

    @pytest.mark.asyncio
    async def test_catch_all_lowest_priority_wins(self):
        nodes = [
            {
                "id": "dispatch",
                "node_type": "router",
                "edges": [
                    {"to_node_id": "generic", "condition_type": "unconditional", "priority": 5},
                    {"to_node_id": "premium", "condition_type": "unconditional", "priority": 1},
                ],
            },
            {"id": "generic", "prompt": "g", "edges": []},
            {"id": "premium", "prompt": "p", "edges": []},
        ]
        agent = _make_agent(_base_config(nodes, "dispatch"))
        await agent._resolve_router_chain([{"role": "user", "content": "x"}])
        assert agent.current_node_id == "premium"  # priority 1 < 5, not declaration order

    @pytest.mark.asyncio
    async def test_silence_flag_threaded_into_hops(self):
        agent = _router_agent(detected_language="hi")
        hops = await agent._resolve_router_chain([{"role": "user", "content": "[silence] "}])
        assert hops[0]["is_silence_trigger"] is True


class TestFirstDeliveryHold:
    """A node entered by a transition must speak its first question before it can route out
    (BOLNA-1582): an utterance arriving before that first TTS turn is delivered is context only."""

    def _nodes(self):
        return [
            {"id": "A", "prompt": "A.", "edges": [{"to_node_id": "B", "condition_type": "unconditional"}]},
            {
                "id": "B",
                "prompt": "Ask Q1.",
                "edges": [{"to_node_id": "C", "condition": "user answered", "function_name": "transition_to_C"}],
            },
            {"id": "C", "prompt": "C.", "edges": []},
        ]

    _PICK_C = ("C", {}, 1.0, [{"role": "system", "content": "routing"}], [], "user answered", 0.9, None)

    @pytest.mark.asyncio
    async def test_holds_and_speaks_when_first_response_undelivered(self):
        agent = _make_agent(_base_config(self._nodes(), "A"))
        agent._advance_to_node("B", 0)
        assert agent._active_node_first_response_delivered is False

        with patch.object(
            agent, "_decide_next_node_llm", new_callable=AsyncMock, return_value=self._PICK_C
        ) as mock_llm:
            out = await _collect(agent.generate([{"role": "user", "content": "haanji boliye"}]))

        mock_llm.assert_not_called()
        assert agent.current_node_id == "B"
        routing = [o["routing_info"] for o in out if isinstance(o, dict) and "routing_info" in o]
        assert routing[-1]["routing_type"] == "hold"
        assert routing[-1]["transitioned"] is False
        assert any(isinstance(o, dict) and "messages" in o for o in out)

    @pytest.mark.asyncio
    async def test_routes_after_first_response_delivered(self):
        agent = _make_agent(_base_config(self._nodes(), "A"))
        agent._advance_to_node("B", 0)
        agent.mark_first_response_delivered()
        assert agent._active_node_first_response_delivered is True

        with patch.object(
            agent, "_decide_next_node_llm", new_callable=AsyncMock, return_value=self._PICK_C
        ) as mock_llm:
            await _collect(agent.generate([{"role": "user", "content": "yes I answered"}]))

        mock_llm.assert_called_once()
        assert agent.current_node_id == "C"

    @pytest.mark.asyncio
    async def test_initial_node_routes_on_first_turn(self):
        agent = _make_agent(_base_config(self._nodes(), "B"))
        assert agent._active_node_first_response_delivered is True

        with patch.object(
            agent, "_decide_next_node_llm", new_callable=AsyncMock, return_value=self._PICK_C
        ) as mock_llm:
            await _collect(agent.generate([{"role": "user", "content": "answer"}]))

        mock_llm.assert_called_once()
        assert agent.current_node_id == "C"

    @pytest.mark.asyncio
    async def test_advance_resets_delivered_flag(self):
        agent = _make_agent(_base_config(self._nodes(), "B"))
        assert agent._active_node_first_response_delivered is True
        agent._advance_to_node("C", 0)
        assert agent._active_node_first_response_delivered is False
