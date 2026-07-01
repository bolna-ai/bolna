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

    def test_router_with_llm_edge_rejected(self):
        with pytest.raises(ValidationError, match="expression or unconditional"):
            GraphNode(
                id="dispatch",
                node_type=NodeType.ROUTER,
                edges=[
                    {"to_node_id": "a", "condition": "user wants a"},  # no condition_type -> llm
                    {"to_node_id": "b", "condition_type": "unconditional"},
                ],
            )

    def test_router_with_event_edge_rejected(self):
        with pytest.raises(ValidationError, match="expression or unconditional"):
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
    def test_single_hop_expression_match(self):
        agent = _router_agent(detected_language="hi")
        hops = agent._resolve_router_chain([])
        assert agent.current_node_id == "hindi"
        assert len(hops) == 1
        assert hops[0]["previous_node"] == "dispatch"
        assert hops[0]["current_node"] == "hindi"
        assert hops[0]["routing_type"] == "deterministic"
        assert hops[0]["reasoning"].startswith(_DETERMINISTIC_REASONING_PREFIX)
        assert hops[0]["confidence"] == 1.0
        assert hops[0]["node_type"] == NodeType.LLM

    def test_catch_all_fallback_when_no_expression_matches(self):
        agent = _router_agent(detected_language="fr")
        hops = agent._resolve_router_chain([])
        assert agent.current_node_id == "english"
        assert len(hops) == 1
        assert hops[0]["current_node"] == "english"

    def test_catch_all_wins_even_if_listed_first(self):
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
        agent._resolve_router_chain([])
        assert agent.current_node_id == "vip"

    def test_expression_priority_order(self):
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
        agent._resolve_router_chain([])
        assert agent.current_node_id == "high"  # lower priority number wins

    def test_multi_hop_chain(self):
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
        hops = agent._resolve_router_chain([])
        assert agent.current_node_id == "leaf"
        assert [h["current_node"] for h in hops] == ["r2", "leaf"]
        assert agent.node_history[-1] == "leaf"

    def test_runtime_cycle_is_bounded(self):
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
        hops = agent._resolve_router_chain([])  # must return, not hang
        assert len(hops) == 2  # r1->r2, r2->r1, then r1 already visited -> stop

    def test_no_hops_when_current_not_router(self):
        nodes = [{"id": "leaf", "prompt": "hi", "edges": []}]
        agent = _make_agent(_base_config(nodes, "leaf"))
        hops = agent._resolve_router_chain([])
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
