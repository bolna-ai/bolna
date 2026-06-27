"""Tests for typed agent state (working memory).

Covers the read path (pinned state block), the two write paths (tool-result
assignments and the built-in update_state tool), seeding-time coercion, and that
typed routing reads state after a flag is set.
"""

import asyncio
import json

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from bolna.agent_types.graph_agent import GraphAgent
from bolna.helpers.expression_evaluator import evaluate_condition, resolve_variable, set_variable
from bolna.helpers.agent_state import (
    apply_state_assignments,
    apply_state_updates,
    format_state_block,
)


# ---------------------------------------------------------------------------
# update_state write path (write path B)
# ---------------------------------------------------------------------------


class TestUpdateStateWrites:
    def test_bare_names_written_under_state_namespace_and_coerced(self):
        ctx = {"state": {}}
        vt = {"state.otp_verified": "boolean", "state.total": "number"}
        written = apply_state_updates(ctx, {"otp_verified": "true", "total": "420"}, vt)
        assert ctx["state"]["otp_verified"] is True
        assert ctx["state"]["total"] == 420.0
        assert set(written) == {"state.otp_verified", "state.total"}

    def test_empty_updates_noop(self):
        ctx = {"state": {}}
        assert apply_state_updates(ctx, {}, {}) == []


# ---------------------------------------------------------------------------
# Tool-result assignment write path (write path A)
# ---------------------------------------------------------------------------


class TestToolResultAssignment:
    def test_dotpath_response_field_coerced_into_state(self):
        ctx = {"state": {}}
        source = {"data": {"limit": "4800"}}
        written = apply_state_assignments(
            ctx, {"state.credit_limit": "data.limit"}, source, {"state.credit_limit": "number"}
        )
        assert ctx["state"]["credit_limit"] == 4800.0
        assert written == ["state.credit_limit"]

    def test_missing_source_field_skipped(self):
        ctx = {"state": {}}
        written = apply_state_assignments(ctx, {"state.x": "a.b"}, {"a": {}}, {"state.x": "number"})
        assert written == []
        assert ctx["state"] == {}


# ---------------------------------------------------------------------------
# Pinned state block (read path)
# ---------------------------------------------------------------------------


class TestStateBlockRender:
    def test_renders_declared_present_values(self):
        ctx = {"recipient_data": {"age": 42.0}, "state": {"otp_verified": True}}
        vt = {"recipient_data.age": "number", "state.otp_verified": "boolean"}
        block = format_state_block(ctx, vt)
        assert "## Current state" in block
        assert "state.otp_verified: true" in block
        assert "recipient_data.age: 42.0" in block

    def test_empty_without_variable_types(self):
        assert format_state_block({"state": {"x": 1}}, {}) == ""
        assert format_state_block({}, {"state.x": "number"}) == ""


# ---------------------------------------------------------------------------
# Typed routing reads state after a flag is set
# ---------------------------------------------------------------------------


class TestTypedRoutingOnState:
    def test_edge_condition_matches_after_flag_set(self):
        ctx = {"state": {}}
        vt = {"state.otp_verified": "boolean"}
        cond = {"variable": "state.otp_verified", "operator": "eq", "value": True}
        assert evaluate_condition(cond, ctx, vt) is False  # missing -> no match
        apply_state_updates(ctx, {"otp_verified": "true"}, vt)
        assert evaluate_condition(cond, ctx, vt) is True

    def test_string_value_still_routes_via_declared_bool(self):
        # The flag arrives as the string "false"; declared boolean makes it route correctly.
        ctx = {"state": {"otp_verified": "false"}}
        vt = {"state.otp_verified": "boolean"}
        cond = {"variable": "state.otp_verified", "operator": "eq", "value": False}
        assert evaluate_condition(cond, ctx, vt) is True


# ---------------------------------------------------------------------------
# set_variable dot-notation
# ---------------------------------------------------------------------------


class TestSetVariable:
    def test_creates_intermediate_dicts(self):
        ctx = {}
        set_variable(ctx, "state.nested.flag", True)
        assert ctx == {"state": {"nested": {"flag": True}}}
        assert resolve_variable(ctx, "state.nested.flag") is True


# ---------------------------------------------------------------------------
# GraphAgent injects the state block into its messages
# ---------------------------------------------------------------------------


def _make_graph_agent(variable_types, context_data):
    cfg = {
        "agent_information": "Test agent",
        "model": "gpt-4o-mini",
        "provider": "openai",
        "temperature": 0.7,
        "max_tokens": 150,
        "current_node_id": "greeting",
        "nodes": [{"id": "greeting", "prompt": "Greet the user.", "edges": []}],
        "variable_types": variable_types,
        "context_data": context_data,
    }
    mock_llm = MagicMock()
    mock_llm.trigger_function_call = False
    with (
        patch("bolna.agent_types.graph_agent.OpenAI", return_value=MagicMock()),
        patch("bolna.agent_types.graph_agent.SUPPORTED_LLM_PROVIDERS", {"openai": MagicMock(return_value=mock_llm)}),
        patch("bolna.agent_types.graph_agent.OpenAiLLM", return_value=MagicMock()),
    ):
        return GraphAgent(cfg)


class TestGraphAgentStateBlock:
    def test_build_messages_appends_state_block(self):
        agent = _make_graph_agent(
            {"state.otp_verified": "boolean"},
            {"recipient_data": {}, "state": {"otp_verified": True}},
        )
        messages = asyncio.run(agent._build_messages([{"role": "user", "content": "hi"}]))
        system_contents = [m["content"] for m in messages if m["role"] == "system"]
        assert any("## Current state" in c and "state.otp_verified: true" in c for c in system_contents)

    def test_no_state_block_without_variable_types(self):
        agent = _make_graph_agent({}, {"recipient_data": {}})
        messages = asyncio.run(agent._build_messages([{"role": "user", "content": "hi"}]))
        assert not any("## Current state" in m["content"] for m in messages if m["role"] == "system")


# ---------------------------------------------------------------------------
# update_state tool injection (write path B wiring)
# ---------------------------------------------------------------------------


class TestUpdateStateToolInjection:
    def _inject(self, variable_types, api_tools=None):
        from bolna.agent_manager.task_manager import TaskManager

        stub = MagicMock()
        stub.variable_types = variable_types
        stub.kwargs = {"api_tools": api_tools} if api_tools is not None else {}
        TaskManager._TaskManager__inject_update_state_tool.__get__(stub, TaskManager)()
        return stub.kwargs.get("api_tools")

    def test_injects_typed_tool_for_writable_vars(self):
        api_tools = self._inject(
            {"state.otp_verified": "boolean", "state.total": "number", "recipient_data.age": "number"}
        )
        assert "update_state" in api_tools["tools_params"]
        tool = next(t for t in api_tools["tools"] if t["function"]["name"] == "update_state")
        props = tool["function"]["parameters"]["properties"]
        # only state.* vars become params, with their JSON-schema types
        assert props == {"otp_verified": {"type": "boolean"}, "total": {"type": "number"}}
        assert tool["function"]["parameters"]["required"] == []

    def test_noop_without_writable_state_vars(self):
        api_tools = self._inject({"recipient_data.age": "number"})
        assert api_tools is None

    def test_enum_var_emits_enum_schema(self):
        api_tools = self._inject({"state.status": {"type": "enum", "values": ["pending", "approved", "rejected"]}})
        tool = next(t for t in api_tools["tools"] if t["function"]["name"] == "update_state")
        prop = tool["function"]["parameters"]["properties"]["status"]
        assert prop == {"type": "string", "enum": ["pending", "approved", "rejected"]}


# ---------------------------------------------------------------------------
# Enum type (backward-compatible with the bare-string form)
# ---------------------------------------------------------------------------


class TestEnumType:
    _VT = {"state.status": {"type": "enum", "values": ["pending", "approved", "rejected"]}}

    def test_update_coerces_enum_to_string_and_stores(self):
        ctx = {"state": {}}
        apply_state_updates(ctx, {"status": "approved"}, self._VT)
        assert ctx["state"]["status"] == "approved"

    def test_out_of_set_value_stored_anyway(self):
        # write-time membership is a warning, not a hard reject (the tool schema is the
        # real enforcement); the value is still stored.
        ctx = {"state": {}}
        apply_state_updates(ctx, {"status": "weird"}, self._VT)
        assert ctx["state"]["status"] == "weird"

    def test_routing_eq_on_enum(self):
        ctx = {"state": {"status": "approved"}}
        assert evaluate_condition({"variable": "state.status", "operator": "eq", "value": "approved"}, ctx, self._VT)
        assert not evaluate_condition({"variable": "state.status", "operator": "eq", "value": "pending"}, ctx, self._VT)

    def test_routing_in_on_enum(self):
        ctx = {"state": {"status": "approved"}}
        cond = {"variable": "state.status", "operator": "in", "value": ["approved", "rejected"]}
        assert evaluate_condition(cond, ctx, self._VT) is True

    def test_bare_string_form_still_works(self):
        ctx = {"state": {}}
        apply_state_updates(ctx, {"flag": "true"}, {"state.flag": "boolean"})
        assert ctx["state"]["flag"] is True
