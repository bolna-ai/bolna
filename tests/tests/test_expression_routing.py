"""Tests for expression-based routing in GraphAgent.

Covers: _classify_edges, deterministic match skips LLM, no-match falls through
to LLM, backward compatibility, unconditional edges, priority ordering, mixed edges.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from bolna.agent_types.graph_agent import GraphAgent, _DETERMINISTIC_REASONING_PREFIX
from bolna.helpers.expression_evaluator import evaluate_condition


# ---------------------------------------------------------------------------
# Helpers (reused from test_routing_reasoning_confidence.py)
# ---------------------------------------------------------------------------


def _make_config(**overrides):
    defaults = {
        "agent_information": "Test agent",
        "model": "gpt-4o-mini",
        "provider": "openai",
        "temperature": 0.7,
        "max_tokens": 150,
        "current_node_id": "greeting",
        "nodes": [
            {
                "id": "greeting",
                "prompt": "Greet the user.",
                "edges": [
                    {
                        "to_node_id": "booking",
                        "condition": "user wants to book",
                        "function_name": "go_to_booking",
                        "parameters": {"appointment_type": "string"},
                    },
                ],
            },
            {
                "id": "booking",
                "prompt": "Book the appointment.",
                "edges": [],
            },
            {
                "id": "hindi_agent",
                "prompt": "Respond in Hindi.",
                "edges": [],
            },
            {
                "id": "escalation",
                "prompt": "Escalate.",
                "edges": [],
            },
            {
                "id": "fallback",
                "prompt": "Fallback.",
                "edges": [],
            },
        ],
    }
    defaults.update(overrides)
    return defaults


def _make_agent(config_overrides=None):
    cfg = _make_config(**(config_overrides or {}))
    mock_llm = MagicMock()
    mock_llm.generate_stream = AsyncMock(return_value=_async_iter([]))
    mock_llm.trigger_function_call = False
    mock_openai_client = MagicMock()
    mock_openai_llm_cls = MagicMock(return_value=mock_llm)

    with (
        patch("bolna.agent_types.graph_agent.OpenAI", return_value=mock_openai_client),
        patch("bolna.agent_types.graph_agent.SUPPORTED_LLM_PROVIDERS", {"openai": mock_openai_llm_cls}),
        patch("bolna.agent_types.graph_agent.OpenAiLLM", return_value=MagicMock()),
    ):
        agent = GraphAgent(cfg)

    agent._mock_llm = mock_llm
    return agent


async def _async_iter(items):
    for item in items:
        yield item


def _mock_routing_response(function_name, function_args_dict):
    mock_tool_call = MagicMock()
    mock_tool_call.function.name = function_name
    mock_tool_call.function.arguments = json.dumps(function_args_dict)
    mock_message = MagicMock()
    mock_message.tool_calls = [mock_tool_call]
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    return mock_response


# ---------------------------------------------------------------------------
# _edge_function_name
# ---------------------------------------------------------------------------


class TestEdgeFunctionName:
    def test_explicit_name(self):
        assert (
            GraphAgent._edge_function_name({"function_name": "go_to_booking", "to_node_id": "booking"})
            == "go_to_booking"
        )

    def test_auto_generated_name(self):
        assert GraphAgent._edge_function_name({"to_node_id": "booking"}) == "transition_to_booking"


# ---------------------------------------------------------------------------
# _classify_edges
# ---------------------------------------------------------------------------


class TestClassifyEdges:
    def test_all_llm_edges(self):
        agent = _make_agent()
        edges = [
            {"to_node_id": "a", "condition": "wants a"},
            {"to_node_id": "b", "condition": "wants b", "condition_type": "llm"},
        ]
        det, llm = agent._classify_edges(edges)
        assert len(det) == 0
        assert len(llm) == 2

    def test_all_expression_edges(self):
        agent = _make_agent()
        edges = [
            {"to_node_id": "a", "condition_type": "expression", "expression": {"logic": "and", "conditions": []}},
            {"to_node_id": "b", "condition_type": "unconditional"},
        ]
        det, llm = agent._classify_edges(edges)
        assert len(det) == 2
        assert len(llm) == 0

    def test_mixed_edges(self):
        agent = _make_agent()
        edges = [
            {"to_node_id": "a", "condition_type": "expression", "expression": {}},
            {"to_node_id": "b", "condition": "user intent"},
            {"to_node_id": "c", "condition_type": "unconditional"},
        ]
        det, llm = agent._classify_edges(edges)
        assert len(det) == 2
        assert len(llm) == 1

    def test_priority_sorting(self):
        agent = _make_agent()
        edges = [
            {"to_node_id": "b", "condition_type": "expression", "priority": 10, "expression": {}},
            {"to_node_id": "a", "condition_type": "expression", "priority": 1, "expression": {}},
            {"to_node_id": "c", "condition_type": "unconditional", "priority": 999},
        ]
        det, llm = agent._classify_edges(edges)
        assert [e["to_node_id"] for e in det] == ["a", "b", "c"]

    def test_none_condition_type_is_llm(self):
        """Backward compat: edges with no condition_type are treated as LLM."""
        agent = _make_agent()
        edges = [{"to_node_id": "x", "condition": "something", "condition_type": None}]
        det, llm = agent._classify_edges(edges)
        assert len(det) == 0
        assert len(llm) == 1

    def test_none_priority_does_not_crash(self):
        """priority: None (from JSON null) should not crash sorting."""
        agent = _make_agent()
        edges = [
            {"to_node_id": "a", "condition_type": "expression", "priority": None, "expression": {}},
            {"to_node_id": "b", "condition_type": "expression", "priority": None, "expression": {}},
            {"to_node_id": "c", "condition": "fallback", "priority": None},
        ]
        det, llm = agent._classify_edges(edges)
        assert len(det) == 2
        assert len(llm) == 1


# ---------------------------------------------------------------------------
# Expression match skips LLM
# ---------------------------------------------------------------------------


class TestExpressionMatchSkipsLLM:
    @pytest.mark.asyncio
    async def test_expression_match_returns_instantly(self):
        """When an expression edge matches, LLM should NOT be called."""
        agent = _make_agent(
            {
                "nodes": [
                    {
                        "id": "greeting",
                        "prompt": "Greet.",
                        "edges": [
                            {
                                "to_node_id": "hindi_agent",
                                "condition": "Hindi speaker",
                                "condition_type": "expression",
                                "expression": {
                                    "logic": "or",
                                    "conditions": [
                                        {"variable": "detected_language", "operator": "eq", "value": "hindi"},
                                        {"variable": "detected_language", "operator": "eq", "value": "hi"},
                                    ],
                                },
                            },
                            {
                                "to_node_id": "booking",
                                "condition": "user wants to book",
                                "function_name": "go_to_booking",
                            },
                        ],
                    },
                    {"id": "hindi_agent", "prompt": "Hindi.", "edges": []},
                    {"id": "booking", "prompt": "Book.", "edges": []},
                ],
            }
        )

        agent.context_data = {"detected_language": "hindi"}

        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            result = await agent.decide_next_node_with_functions([{"role": "user", "content": "namaste"}])
            # LLM should NOT have been called
            mock_thread.assert_not_called()

        next_node, params, latency_ms, msgs, tools, reasoning, confidence, usage_info = result
        assert next_node == "hindi_agent"
        assert confidence == 1.0
        assert reasoning.startswith(_DETERMINISTIC_REASONING_PREFIX)
        assert latency_ms < 50  # should be near-instant

    @pytest.mark.asyncio
    async def test_unconditional_match_returns_instantly(self):
        agent = _make_agent(
            {
                "nodes": [
                    {
                        "id": "greeting",
                        "prompt": "Greet.",
                        "edges": [
                            {"to_node_id": "fallback", "condition": "Default", "condition_type": "unconditional"},
                        ],
                    },
                    {"id": "fallback", "prompt": "Fallback.", "edges": []},
                ],
            }
        )

        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            result = await agent.decide_next_node_with_functions([{"role": "user", "content": "hello"}])
            mock_thread.assert_not_called()

        assert result[0] == "fallback"
        assert result[6] == 1.0  # confidence


# ---------------------------------------------------------------------------
# No expression match → falls through to LLM
# ---------------------------------------------------------------------------


class TestFallThroughToLLM:
    @pytest.mark.asyncio
    async def test_expression_no_match_falls_to_llm(self):
        agent = _make_agent(
            {
                "nodes": [
                    {
                        "id": "greeting",
                        "prompt": "Greet.",
                        "edges": [
                            {
                                "to_node_id": "hindi_agent",
                                "condition": "Hindi speaker",
                                "condition_type": "expression",
                                "expression": {
                                    "logic": "and",
                                    "conditions": [
                                        {"variable": "detected_language", "operator": "eq", "value": "hindi"},
                                    ],
                                },
                            },
                            {
                                "to_node_id": "booking",
                                "condition": "user wants to book",
                                "function_name": "go_to_booking",
                            },
                        ],
                    },
                    {"id": "hindi_agent", "prompt": "Hindi.", "edges": []},
                    {"id": "booking", "prompt": "Book.", "edges": []},
                ],
            }
        )

        agent.context_data = {"detected_language": "en"}

        mock_resp = _mock_routing_response(
            "go_to_booking",
            {
                "reasoning": "User wants to book",
                "confidence": 0.9,
            },
        )

        with patch("asyncio.to_thread", new_callable=AsyncMock, return_value=mock_resp):
            result = await agent.decide_next_node_with_functions([{"role": "user", "content": "I want to book"}])

        next_node, params, latency_ms, msgs, tools, reasoning, confidence, usage_info = result
        assert next_node == "booking"
        assert confidence == 0.9
        assert not reasoning.startswith(_DETERMINISTIC_REASONING_PREFIX)

    @pytest.mark.asyncio
    async def test_llm_tools_exclude_expression_edges(self):
        """When falling through to LLM, tools should only contain LLM edges."""
        agent = _make_agent(
            {
                "nodes": [
                    {
                        "id": "greeting",
                        "prompt": "Greet.",
                        "edges": [
                            {
                                "to_node_id": "hindi_agent",
                                "condition_type": "expression",
                                "condition": "Hindi",
                                "expression": {
                                    "logic": "and",
                                    "conditions": [
                                        {"variable": "detected_language", "operator": "eq", "value": "hindi"},
                                    ],
                                },
                            },
                            {
                                "to_node_id": "booking",
                                "condition": "user wants to book",
                                "function_name": "go_to_booking",
                            },
                        ],
                    },
                    {"id": "hindi_agent", "prompt": "Hindi.", "edges": []},
                    {"id": "booking", "prompt": "Book.", "edges": []},
                ],
            }
        )

        agent.context_data = {"detected_language": "en"}

        mock_resp = _mock_routing_response(
            "stay_on_current_node",
            {
                "reasoning": "No match",
                "confidence": 0.5,
            },
        )

        with patch("asyncio.to_thread", new_callable=AsyncMock, return_value=mock_resp):
            result = await agent.decide_next_node_with_functions([{"role": "user", "content": "hi"}])

        # Check that tools (result[4]) only has go_to_booking + stay_on_current_node
        tools = result[4]
        tool_names = [t["function"]["name"] for t in tools]
        assert "go_to_booking" in tool_names
        assert "stay_on_current_node" in tool_names
        # Expression edge should NOT appear as an LLM tool
        assert "transition_to_hindi_agent" not in tool_names


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    @pytest.mark.asyncio
    async def test_existing_config_no_condition_type(self):
        """Edges without condition_type still work as LLM edges."""
        agent = _make_agent()  # default config has no condition_type
        agent.context_data = {}

        mock_resp = _mock_routing_response(
            "go_to_booking",
            {
                "appointment_type": "oil_change",
                "reasoning": "User wants oil change",
                "confidence": 0.95,
            },
        )

        with patch("asyncio.to_thread", new_callable=AsyncMock, return_value=mock_resp):
            result = await agent.decide_next_node_with_functions([{"role": "user", "content": "I want an oil change"}])

        assert result[0] == "booking"
        assert result[6] == 0.95


# ---------------------------------------------------------------------------
# Priority ordering
# ---------------------------------------------------------------------------


class TestPriorityOrdering:
    @pytest.mark.asyncio
    async def test_lower_priority_evaluated_first(self):
        """Edge with lower priority number should be evaluated first."""
        agent = _make_agent(
            {
                "nodes": [
                    {
                        "id": "greeting",
                        "prompt": "Greet.",
                        "edges": [
                            {
                                "to_node_id": "fallback",
                                "condition_type": "unconditional",
                                "condition": "Default",
                                "priority": 999,
                            },
                            {
                                "to_node_id": "escalation",
                                "condition_type": "expression",
                                "condition": "Turn limit",
                                "priority": 1,
                                "expression": {
                                    "logic": "and",
                                    "conditions": [
                                        {"variable": "_node_turns", "operator": "gte", "value": 3},
                                    ],
                                },
                            },
                        ],
                    },
                    {"id": "escalation", "prompt": "Escalate.", "edges": []},
                    {"id": "fallback", "prompt": "Fallback.", "edges": []},
                ],
            }
        )

        # With enough turns, expression edge (priority 1) should match first
        history = [{"role": "user", "content": f"msg {i}"} for i in range(5)]
        agent.context_data = {}

        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            result = await agent.decide_next_node_with_functions(history)
            mock_thread.assert_not_called()

        assert result[0] == "escalation"

    @pytest.mark.asyncio
    async def test_unconditional_with_high_priority_is_fallback(self):
        """Unconditional edge with high priority should only match if expression doesn't."""
        agent = _make_agent(
            {
                "nodes": [
                    {
                        "id": "greeting",
                        "prompt": "Greet.",
                        "edges": [
                            {
                                "to_node_id": "escalation",
                                "condition_type": "expression",
                                "condition": "Turn limit",
                                "priority": 1,
                                "expression": {
                                    "logic": "and",
                                    "conditions": [
                                        {"variable": "_node_turns", "operator": "gte", "value": 10},
                                    ],
                                },
                            },
                            {
                                "to_node_id": "fallback",
                                "condition_type": "unconditional",
                                "condition": "Default",
                                "priority": 999,
                            },
                        ],
                    },
                    {"id": "escalation", "prompt": "Escalate.", "edges": []},
                    {"id": "fallback", "prompt": "Fallback.", "edges": []},
                ],
            }
        )

        # Only 2 turns — expression won't match, unconditional will
        history = [{"role": "user", "content": "hi"}, {"role": "user", "content": "hello"}]
        agent.context_data = {}

        result = await agent.decide_next_node_with_functions(history)
        assert result[0] == "fallback"


# ---------------------------------------------------------------------------
# Turn count injection
# ---------------------------------------------------------------------------


class TestTurnCounts:
    @pytest.mark.asyncio
    async def test_node_turns_computed(self):
        agent = _make_agent(
            {
                "nodes": [
                    {
                        "id": "greeting",
                        "prompt": "Greet.",
                        "edges": [
                            {
                                "to_node_id": "escalation",
                                "condition_type": "expression",
                                "condition": "Too many turns",
                                "expression": {
                                    "logic": "and",
                                    "conditions": [
                                        {"variable": "_node_turns", "operator": "gte", "value": 3},
                                    ],
                                },
                            },
                        ],
                    },
                    {"id": "escalation", "prompt": "Escalate.", "edges": []},
                ],
            }
        )
        agent.context_data = {}
        agent.current_node_entry_index = 0

        history = [
            {"role": "user", "content": "msg1"},
            {"role": "assistant", "content": "reply1"},
            {"role": "user", "content": "msg2"},
            {"role": "assistant", "content": "reply2"},
            {"role": "user", "content": "msg3"},
        ]

        result = await agent.decide_next_node_with_functions(history)
        assert result[0] == "escalation"
        assert agent.context_data["_node_turns"] == 3
        assert agent.context_data["_total_turns"] == 3

    @pytest.mark.asyncio
    async def test_total_turns_with_node_entry_offset(self):
        agent = _make_agent(
            {
                "nodes": [
                    {
                        "id": "greeting",
                        "prompt": "Greet.",
                        "edges": [
                            {
                                "to_node_id": "escalation",
                                "condition_type": "expression",
                                "condition": "Total turns high",
                                "expression": {
                                    "logic": "and",
                                    "conditions": [
                                        {"variable": "_total_turns", "operator": "gte", "value": 4},
                                    ],
                                },
                            },
                        ],
                    },
                    {"id": "escalation", "prompt": "Escalate.", "edges": []},
                ],
            }
        )
        agent.context_data = {}
        agent.current_node_entry_index = 4  # entered at message index 4

        history = [
            {"role": "user", "content": "msg1"},
            {"role": "assistant", "content": "reply1"},
            {"role": "user", "content": "msg2"},
            {"role": "assistant", "content": "reply2"},
            # node entry here
            {"role": "user", "content": "msg3"},
            {"role": "assistant", "content": "reply3"},
            {"role": "user", "content": "msg4"},
        ]

        result = await agent.decide_next_node_with_functions(history)
        assert result[0] == "escalation"
        assert agent.context_data["_total_turns"] == 4
        assert agent.context_data["_node_turns"] == 2


# ---------------------------------------------------------------------------
# routing_info includes routing_type
# ---------------------------------------------------------------------------


class TestRoutingInfoType:
    @pytest.mark.asyncio
    async def test_deterministic_routing_type(self):
        agent = _make_agent(
            {
                "nodes": [
                    {
                        "id": "greeting",
                        "prompt": "Greet.",
                        "edges": [
                            {
                                "to_node_id": "hindi_agent",
                                "condition_type": "expression",
                                "condition": "Hindi",
                                "expression": {
                                    "logic": "and",
                                    "conditions": [
                                        {"variable": "detected_language", "operator": "eq", "value": "hindi"},
                                    ],
                                },
                            },
                        ],
                    },
                    {"id": "hindi_agent", "prompt": "Hindi.", "edges": []},
                ],
            }
        )

        async def fake_stream(*args, **kwargs):
            return
            yield

        agent._mock_llm.generate_stream = fake_stream
        agent.context_data = {"detected_language": "hindi"}

        with patch("asyncio.to_thread", new_callable=AsyncMock):
            chunks = []
            async for chunk in agent.generate(
                [{"role": "user", "content": "namaste"}],
                meta_info={"detected_language": "hindi"},
            ):
                chunks.append(chunk)

        routing_info = chunks[0].get("routing_info")
        assert routing_info["routing_type"] == "deterministic"
        assert routing_info["confidence"] == 1.0

    @pytest.mark.asyncio
    async def test_llm_routing_type(self):
        agent = _make_agent()

        async def fake_stream(*args, **kwargs):
            return
            yield

        agent._mock_llm.generate_stream = fake_stream
        agent.context_data = {}

        mock_resp = _mock_routing_response(
            "go_to_booking",
            {
                "appointment_type": "oil_change",
                "reasoning": "User wants to book",
                "confidence": 0.9,
            },
        )

        with patch("asyncio.to_thread", new_callable=AsyncMock, return_value=mock_resp):
            chunks = []
            async for chunk in agent.generate(
                [{"role": "user", "content": "book me"}],
                meta_info={},
            ):
                chunks.append(chunk)

        routing_info = chunks[0].get("routing_info")
        assert routing_info["routing_type"] == "llm"


class TestTypedCoercion:
    """variable_types coerces expression comparisons into the right domain."""

    def test_number_avoids_lexicographic(self):
        # declared number -> numeric comparison: 9 > 10 is False
        cond = {"variable": "recipient_data.age", "operator": "gt", "value": 10}
        ctx = {"recipient_data": {"age": "9"}}
        assert not evaluate_condition(cond, ctx, {"recipient_data.age": "number"})

    def test_string_string_lexicographic_is_the_bug_typing_fixes(self):
        # both sides strings compare lexicographically ("9" > "10" is True); declaring number corrects it
        cond = {"variable": "recipient_data.age", "operator": "gt", "value": "10"}
        ctx = {"recipient_data": {"age": "9"}}
        assert evaluate_condition(cond, ctx)
        assert not evaluate_condition(cond, ctx, {"recipient_data.age": "number"})

    def test_resolution_is_exact_path(self):
        # keys match the condition's variable exactly; a leaf-only key does not apply
        cond = {"variable": "recipient_data.age", "operator": "gt", "value": "100"}
        ctx = {"recipient_data": {"age": "9"}}
        assert not evaluate_condition(cond, ctx, {"recipient_data.age": "number"})  # 9 > 100 numeric -> False
        assert evaluate_condition(cond, ctx, {"age": "number"})  # leaf key ignored -> "9" > "100" lexically -> True

    def test_boolean_string_tokens(self):
        cond = {"variable": "is_member", "operator": "eq", "value": True}
        assert evaluate_condition(cond, {"is_member": "true"}, {"is_member": "boolean"})
        assert not evaluate_condition(cond, {"is_member": "false"}, {"is_member": "boolean"})

    def test_string_match(self):
        cond = {"variable": "plan", "operator": "eq", "value": "premium"}
        assert evaluate_condition(cond, {"plan": "premium"}, {"plan": "string"})

    def test_uncoercible_fails_closed(self):
        cond = {"variable": "age", "operator": "gt", "value": 10}
        assert not evaluate_condition(cond, {"age": "abc"}, {"age": "number"})

    def test_inference_bool_fix_without_declaration(self):
        cond = {"variable": "is_member", "operator": "eq", "value": True}
        assert evaluate_condition(cond, {"is_member": "true"})

    def test_inference_numeric_cross_preserved(self):
        cond = {"variable": "age", "operator": "gte", "value": 18}
        assert evaluate_condition(cond, {"age": "21"})

    def test_inference_bool_literal_does_not_hijack_numbers(self):
        # a bool literal must not change numeric/string comparisons for non-bool-like actuals
        assert evaluate_condition({"variable": "x", "operator": "gt", "value": True}, {"x": 5})  # 5.0 > 1.0
        assert not evaluate_condition({"variable": "x", "operator": "eq", "value": True}, {"x": 5})  # 5 != True
        assert evaluate_condition({"variable": "x", "operator": "neq", "value": False}, {"x": "hello"})

    def test_invalid_declared_type_falls_back_to_inference(self):
        cond = {"variable": "age", "operator": "gt", "value": "10"}
        with patch("bolna.helpers.expression_evaluator.logger") as mock_logger:
            result = evaluate_condition(cond, {"age": "9"}, {"age": "int"})
        assert result  # inference path: "9" > "10" lexically
        assert mock_logger.warning.call_count == 1  # unknown type warned, then ignored

    def test_neq_and_lt_with_declared_number(self):
        assert evaluate_condition({"variable": "age", "operator": "lt", "value": 18}, {"age": "16"}, {"age": "number"})
        assert evaluate_condition({"variable": "age", "operator": "neq", "value": 18}, {"age": "16"}, {"age": "number"})

    def test_membership_coerces_declared_type(self):
        # a declared type coerces the value and the list elements: "21" -> 21.0 in [18.0, 21.0, 25.0]
        cond = {"variable": "age", "operator": "in", "value": [18, 21, 25]}
        assert evaluate_condition(cond, {"age": "21"}, {"age": "number"})
        assert not evaluate_condition(cond, {"age": "20"}, {"age": "number"})


class TestVariableTypesThreading:
    """self.variable_types flows from the agent into deterministic evaluation."""

    def _age_edge(self):
        return {
            "to_node_id": "adult",
            "condition_type": "expression",
            "expression": {
                "logic": "and",
                "conditions": [{"variable": "recipient_data.age", "operator": "gte", "value": 18}],
            },
        }

    def test_threaded_through_deterministic_edges(self):
        agent = _make_agent()
        agent.variable_types = {"recipient_data.age": "number"}
        edge = self._age_edge()

        agent.context_data = {"recipient_data": {"age": "16"}}
        matched, _ = agent._evaluate_deterministic_edges([edge])
        assert matched is None  # 16 >= 18 is False

        agent.context_data = {"recipient_data": {"age": "21"}}
        matched, _ = agent._evaluate_deterministic_edges([edge])
        assert matched is edge  # 21 >= 18 is True

    def test_coercion_failure_warns_once_per_edge(self):
        # the evaluate + describe passes must not double-log the coercion warning
        agent = _make_agent()
        agent.variable_types = {"recipient_data.age": "number"}
        agent.context_data = {"recipient_data": {"age": "abc"}}
        with patch("bolna.helpers.expression_evaluator.logger") as mock_logger:
            matched, _ = agent._evaluate_deterministic_edges([self._age_edge()])
        assert matched is None
        assert mock_logger.warning.call_count == 1

    def test_unknown_type_warns_once_per_edge(self):
        # an invalid declared type warns once per edge, not once per evaluate + describe resolve
        agent = _make_agent()
        agent.variable_types = {"recipient_data.age": "int"}  # not a valid VariableType
        agent.context_data = {"recipient_data": {"age": "21"}}
        with patch("bolna.helpers.expression_evaluator.logger") as mock_logger:
            agent._evaluate_deterministic_edges([self._age_edge()])
        assert mock_logger.warning.call_count == 1
