"""edge_order "priority": an intent edge ranked above a matching expression edge is tried first."""

from unittest.mock import AsyncMock, patch

from bolna.agent_types.graph_agent import _DETERMINISTIC_REASONING_PREFIX
from bolna.enums import EdgeOrder
from bolna.models import GraphNode
from tests.test_router_nodes import _base_config, _expr, _make_agent

_HISTORY = [{"role": "user", "content": "I want to talk to a person"}]
_NO_DECISION = (None, None, 210.0, [{"role": "system", "content": "routing"}], [{"type": "function"}], None, None, None)


def _edges(human_priority=1):
    return [
        {
            "to_node_id": "hindi",
            "condition_type": "expression",
            "expression": _expr("detected_language", "eq", "hi"),
            "priority": 2,
        },
        {"to_node_id": "human", "condition": "user wants a human", "priority": human_priority},
        {"to_node_id": "billing", "condition": "user asks about billing"},
        {"to_node_id": "general", "condition_type": "unconditional"},
    ]


def _agent(node_type="llm", edge_order=None, human_priority=1, **ctx):
    start = {"id": "start", "node_type": node_type, "edges": _edges(human_priority)}
    if node_type == "llm":
        start["prompt"] = "Help the caller."
    if edge_order:
        start["edge_order"] = edge_order
    nodes = [start] + [{"id": n, "prompt": n, "edges": []} for n in ("hindi", "human", "billing", "general")]
    return _make_agent(_base_config(nodes, "start", context_data=ctx))


def _picked(node_id):
    return (node_id, {}, 300.0, [{"role": "system", "content": "routing"}], [], "asked for a person", 0.9, {})


class TestNodeEdgeOrder:
    async def test_intent_ranked_above_matched_expression_wins(self):
        agent = _agent(edge_order="priority", detected_language="hi")
        with patch.object(agent, "_decide_next_node_llm", new_callable=AsyncMock, return_value=_picked("human")) as llm:
            result = await agent.decide_next_node_with_functions(_HISTORY)

        assert result[0] == "human"
        assert [e["to_node_id"] for e in llm.call_args.args[1]] == ["human"]
        assert llm.call_args.kwargs["default_edge"]["to_node_id"] == "hindi"

    async def test_matched_expression_stands_when_routing_returns_no_decision(self):
        agent = _agent(edge_order="priority", detected_language="hi")
        with patch.object(agent, "_decide_next_node_llm", new_callable=AsyncMock, return_value=_NO_DECISION):
            result = await agent.decide_next_node_with_functions(_HISTORY)

        assert result[0] == "hindi"
        assert result[5].startswith(_DETERMINISTIC_REASONING_PREFIX)
        assert result[3] is not None  # the spent routing call is still logged
        assert "intent: no match" in agent._last_deterministic_eval

    async def test_tie_goes_to_the_expression_without_a_routing_call(self):
        agent = _agent(edge_order="priority", human_priority=2, detected_language="hi")
        with patch.object(agent, "_decide_next_node_llm", new_callable=AsyncMock) as llm:
            result = await agent.decide_next_node_with_functions(_HISTORY)

        llm.assert_not_called()
        assert result[0] == "hindi"

    async def test_tiered_node_ignores_priority_across_edge_types(self):
        agent = _agent(detected_language="hi")
        with patch.object(agent, "_decide_next_node_llm", new_callable=AsyncMock) as llm:
            result = await agent.decide_next_node_with_functions(_HISTORY)

        llm.assert_not_called()
        assert result[0] == "hindi"

    async def test_no_matched_expression_routes_over_every_intent_with_the_catch_all(self):
        agent = _agent(edge_order="priority", detected_language="en")
        with patch.object(
            agent, "_decide_next_node_llm", new_callable=AsyncMock, return_value=_picked("billing")
        ) as llm:
            result = await agent.decide_next_node_with_functions(_HISTORY)

        assert result[0] == "billing"
        assert [e["to_node_id"] for e in llm.call_args.args[1]] == ["human", "billing"]
        assert llm.call_args.kwargs["default_edge"]["to_node_id"] == "general"


class TestRouterEdgeOrder:
    async def test_intent_ranked_above_matched_expression_wins(self):
        agent = _agent(node_type="router", edge_order="priority", detected_language="hi")
        with patch.object(agent, "_decide_next_node_llm", new_callable=AsyncMock, return_value=_picked("human")):
            hops = await agent._resolve_router_chain(_HISTORY)

        assert agent.current_node_id == "human"
        assert hops[0]["routing_type"] == "llm"

    async def test_no_decision_lands_on_the_matched_expression_not_the_catch_all(self):
        agent = _agent(node_type="router", edge_order="priority", detected_language="hi")
        with patch.object(agent, "_decide_next_node_llm", new_callable=AsyncMock, return_value=_NO_DECISION) as llm:
            hops = await agent._resolve_router_chain(_HISTORY)

        assert llm.call_args.kwargs["default_edge"]["to_node_id"] == "hindi"
        assert agent.current_node_id == "hindi"
        assert hops[0]["routing_type"] == "deterministic"
        assert hops[0]["routing_messages"] is not None


class TestEdgeOrderConfig:
    def test_defaults_to_tiered(self):
        assert GraphNode(id="n").edge_order == EdgeOrder.TIERED

    def test_accepts_priority(self):
        assert GraphNode(id="n", edge_order="priority").edge_order == EdgeOrder.PRIORITY
