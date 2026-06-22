"""Tests for tool scope (GLOBAL vs NODE) in graph agents and the end_call injection.

Covers:
* GraphAgent._tools_for_node resolution: global tools visible everywhere, node-scoped tools
  only on their nodes, the forced tool always visible, and the escalation-node bug
  (end_call must be ABSENT on intermediate nodes when NODE-scoped) -> fixed.
* The per-call tools= override reaching the LLM request (openai chat + litellm), and tools=None
  leaving behavior unchanged. The responses path is covered via _parse_tools.
* _inject_end_call_tool: global vs node scope, description override, dedup, str tools list.
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from bolna.agent_types.graph_agent import GraphAgent
from bolna.agent_manager.task_manager import _inject_end_call_tool
from bolna.enums import ToolScope
from bolna.constants import END_CALL_TOOL_DEFINITION


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tool(name):
    return {
        "type": "function",
        "function": {"name": name, "description": name, "parameters": {"type": "object", "properties": {}}},
    }


def _make_graph_agent(nodes, current_node_id="n1"):
    cfg = {
        "agent_information": "t",
        "model": "gpt-4o-mini",
        "current_node_id": current_node_id,
        "nodes": nodes,
    }
    with (
        patch("bolna.agent_types.graph_agent.OpenAI", return_value=MagicMock()),
        patch("bolna.agent_types.graph_agent.SUPPORTED_LLM_PROVIDERS", {"openai": MagicMock(return_value=MagicMock())}),
        patch("bolna.agent_types.graph_agent.OpenAiLLM", return_value=MagicMock()),
    ):
        agent = GraphAgent(cfg)
    return agent


class _StubLLM:
    """Minimal stand-in for the response LLM that _tools_for_node reads."""

    def __init__(self, tools, api_params):
        self.trigger_function_call = True
        self.tools = tools
        self.api_params = api_params


def _names(subset):
    return None if subset is None else [t["function"]["name"] for t in subset]


# ---------------------------------------------------------------------------
# _tools_for_node
# ---------------------------------------------------------------------------


def test_global_tools_present_on_every_node():
    nodes = [{"id": "n1", "edges": []}, {"id": "n2", "edges": []}]
    agent = _make_graph_agent(nodes)
    agent.llm = _StubLLM([_tool("a"), _tool("b")], {"a": {}, "b": {"scope": "global"}})
    # All global -> nothing filtered -> None (LLM uses self.tools) on every node.
    assert agent._tools_for_node(agent.get_node_by_id("n1")) is None
    assert agent._tools_for_node(agent.get_node_by_id("n2")) is None


def test_node_scoped_tool_visible_only_on_its_nodes():
    nodes = [{"id": "n1", "edges": []}, {"id": "n2", "edges": []}]
    agent = _make_graph_agent(nodes)
    agent.llm = _StubLLM(
        [_tool("g1"), _tool("g2"), _tool("search")],
        {"g1": {}, "g2": {}, "search": {"scope": "node", "nodes": ["n2"]}},
    )
    assert _names(agent._tools_for_node(agent.get_node_by_id("n1"))) == ["g1", "g2"]
    assert agent._tools_for_node(agent.get_node_by_id("n2")) is None  # all visible -> fallback


def test_node_scoped_tool_with_empty_nodes_is_visible_nowhere():
    nodes = [{"id": "n1", "edges": []}]
    agent = _make_graph_agent(nodes)
    agent.llm = _StubLLM([_tool("g"), _tool("orphan")], {"g": {}, "orphan": {"scope": "node", "nodes": []}})
    assert _names(agent._tools_for_node(agent.get_node_by_id("n1"))) == ["g"]


def test_end_call_node_scoped_absent_on_intermediate_present_and_forced_on_terminal():
    """The production bug: NODE-scoped end_call must be unreachable on an intermediate node."""
    nodes = [
        {"id": "escalation", "node_type": "llm", "prompt": "p", "edges": []},
        {"id": "goodbye", "node_type": "llm", "prompt": "bye", "function_call": "end_call", "edges": []},
    ]
    agent = _make_graph_agent(nodes, current_node_id="escalation")
    agent.llm = _StubLLM(
        [_tool("transfer"), END_CALL_TOOL_DEFINITION],
        {"transfer": {}, "end_call": {"pre_call_message": None, "scope": "node", "nodes": ["goodbye"]}},
    )
    # Intermediate node: end_call is NOT offered -> the model cannot end the call here.
    assert _names(agent._tools_for_node(agent.get_node_by_id("escalation"))) == ["transfer"]
    # Terminal node: end_call is visible (nothing filtered -> None fallback to self.tools).
    assert agent._tools_for_node(agent.get_node_by_id("goodbye")) is None
    # And it is forced there, with the forced tool guaranteed present in the per-call list.
    agent.current_node_id = "goodbye"
    assert agent._get_tool_choice_for_node(history=[]) == {"type": "function", "function": {"name": "end_call"}}
    # Execution invariant: api_params keeps end_call regardless of node (so the call can run).
    assert "end_call" in agent.llm.api_params


def test_end_call_global_present_on_intermediate_node():
    """GLOBAL scope reproduces the original behavior: end_call offered everywhere."""
    nodes = [
        {"id": "escalation", "edges": []},
        {"id": "goodbye", "function_call": "end_call", "edges": []},
    ]
    agent = _make_graph_agent(nodes, current_node_id="escalation")
    agent.llm = _StubLLM(
        [_tool("transfer"), END_CALL_TOOL_DEFINITION, _tool("node_only")],
        {"transfer": {}, "end_call": {"scope": "global"}, "node_only": {"scope": "node", "nodes": ["goodbye"]}},
    )
    # On the intermediate node node_only is filtered but the GLOBAL end_call remains offered.
    assert _names(agent._tools_for_node(agent.get_node_by_id("escalation"))) == ["transfer", "end_call"]


def test_forced_name_keeps_tool_visible_only_while_force_survives():
    """A forced tool stays visible even if scoped elsewhere; when the force is dropped it is hidden."""
    nodes = [{"id": "x", "function_call": "T", "edges": []}]
    agent = _make_graph_agent(nodes, current_node_id="x")
    # G global, T+S both scoped to 'other'; S (a third tool) keeps the subset != full so it's inspectable.
    agent.llm = _StubLLM(
        [_tool("G"), _tool("T"), _tool("S")],
        {"G": {}, "T": {"scope": "node", "nodes": ["other"]}, "S": {"scope": "node", "nodes": ["other"]}},
    )
    x = agent.get_node_by_id("x")
    # force survives -> T visible despite being scoped to 'other'; S still hidden.
    assert _names(agent._tools_for_node(x, "T")) == ["G", "T"]
    # force dropped (forced_name None) -> T hidden (scoped to 'other', not 'x').
    assert _names(agent._tools_for_node(x, None)) == ["G"]


def test_tools_for_node_handles_str_tools_and_no_function_calling():
    nodes = [{"id": "n1", "edges": []}]
    agent = _make_graph_agent(nodes)
    # str tools list with a node-scoped tool absent here.
    agent.llm = _StubLLM(
        json.dumps([_tool("g"), _tool("s")]), {"g": {}, "s": {"scope": "node", "nodes": ["other"]}}
    )
    assert _names(agent._tools_for_node(agent.get_node_by_id("n1"))) == ["g"]
    # trigger_function_call False -> no override.
    agent.llm.trigger_function_call = False
    assert agent._tools_for_node(agent.get_node_by_id("n1")) is None


# ---------------------------------------------------------------------------
# Per-call tools= override at the LLM layer
# ---------------------------------------------------------------------------


class _Sentinel(Exception):
    pass


def _make_openai_llm():
    from bolna.llms.openai_llm import OpenAiLLM

    return OpenAiLLM(
        model="gpt-4o-mini",
        llm_key="sk-dummy",
        api_tools={"tools": [_tool("a"), _tool("b")], "tools_params": {"a": {}, "b": {}}},
    )


@pytest.mark.asyncio
async def test_openai_tools_override_reaches_request():
    llm = _make_openai_llm()
    captured = {}

    async def fake_create(**kwargs):
        captured.update(kwargs)
        raise _Sentinel()

    llm.async_client.chat.completions.create = fake_create
    only = [_tool("a")]
    with pytest.raises(_Sentinel):
        async for _ in llm._generate_stream_chat([{"role": "user", "content": "hi"}], tools=only):
            pass
    assert captured["tools"] == only


@pytest.mark.asyncio
async def test_openai_tools_none_uses_self_tools():
    llm = _make_openai_llm()
    captured = {}

    async def fake_create(**kwargs):
        captured.update(kwargs)
        raise _Sentinel()

    llm.async_client.chat.completions.create = fake_create
    with pytest.raises(_Sentinel):
        async for _ in llm._generate_stream_chat([{"role": "user", "content": "hi"}]):
            pass
    assert captured["tools"] == llm.tools


@pytest.mark.asyncio
async def test_openai_empty_tools_omits_tools_and_tool_choice():
    """An empty per-call tools list (all tools scoped away) must omit tools entirely, not send []."""
    llm = _make_openai_llm()
    captured = {}

    async def fake_create(**kwargs):
        captured.update(kwargs)
        raise _Sentinel()

    llm.async_client.chat.completions.create = fake_create
    with pytest.raises(_Sentinel):
        async for _ in llm._generate_stream_chat([{"role": "user", "content": "hi"}], tools=[]):
            pass
    assert "tools" not in captured
    assert "tool_choice" not in captured


def test_openai_base_parse_tools_override():
    llm = _make_openai_llm()
    only = [_tool("a")]
    assert llm._parse_tools(only) == only  # responses path uses the override
    assert llm._parse_tools(None) == llm.tools  # back-compat
    assert llm._parse_tools() == llm.tools


def _make_openai_llm_with_end_call():
    from bolna.llms.openai_llm import OpenAiLLM

    return OpenAiLLM(
        model="gpt-4o-mini",
        llm_key="sk-dummy",
        api_tools={
            "tools": [_tool("transfer"), END_CALL_TOOL_DEFINITION],
            "tools_params": {
                "transfer": {},
                "end_call": {"pre_call_message": None, "scope": "node", "nodes": ["goodbye"]},
            },
        },
    )


def test_text_rescue_blocked_when_tool_not_offered_this_turn():
    """A text-emitted end_call must NOT execute on a node where it is not offered (node scoping)."""
    llm = _make_openai_llm_with_end_call()
    model_args = {"tools": [_tool("transfer")]}  # end_call hidden this turn
    with patch.object(llm, "_parse_text_tool_call", return_value=("end_call", '{"reason": "x"}')):
        chunk = llm._try_rescue_text_tool_call("end_call(...)", model_args, {}, "bye", None)
    assert chunk is None


def test_text_rescue_executes_when_tool_offered():
    llm = _make_openai_llm_with_end_call()
    model_args = {"tools": [END_CALL_TOOL_DEFINITION]}
    with patch.object(llm, "_parse_text_tool_call", return_value=("end_call", '{"reason": "x"}')):
        chunk = llm._try_rescue_text_tool_call("end_call(...)", model_args, {}, "bye", None)
    assert chunk is not None and chunk.is_function_call


@pytest.mark.asyncio
async def test_litellm_tools_override_reaches_request():
    from bolna.llms.litellm import LiteLLM

    llm = LiteLLM(
        model="gpt-4o-mini",
        api_tools={"tools": [_tool("a"), _tool("b")], "tools_params": {"a": {}, "b": {}}},
    )
    captured = {}

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        raise _Sentinel()

    only = [_tool("b")]
    with patch("bolna.llms.litellm.acompletion", fake_acompletion):
        with pytest.raises(_Sentinel):
            async for _ in llm.generate_stream([{"role": "user", "content": "hi"}], tools=only):
                pass
    assert captured["tools"] == only


# ---------------------------------------------------------------------------
# _inject_end_call_tool
# ---------------------------------------------------------------------------


def test_inject_end_call_node_scope():
    at = _inject_end_call_tool(None, scope=ToolScope.NODE, nodes=["goodbye", "wrong_person_closing"])
    assert at["tools_params"]["end_call"] == {
        "pre_call_message": None,
        "scope": "node",
        "nodes": ["goodbye", "wrong_person_closing"],
    }
    assert at["tools"][0]["function"]["name"] == "end_call"


def test_inject_end_call_global_scope_with_description_and_existing_tool():
    at = {
        "tools": [_tool("x")],
        "tools_params": {"x": {}},
    }
    at = _inject_end_call_tool(at, scope=ToolScope.GLOBAL, nodes=[], description="custom end desc")
    assert at["tools_params"]["end_call"]["scope"] == "global"
    assert at["tools_params"]["end_call"]["nodes"] == []
    end_call_def = next(t for t in at["tools"] if t["function"]["name"] == "end_call")
    assert end_call_def["function"]["description"] == "custom end desc"
    assert len(at["tools"]) == 2


def test_inject_end_call_is_idempotent():
    at = _inject_end_call_tool(None, scope=ToolScope.GLOBAL, nodes=[])
    before = len(at["tools"])
    at = _inject_end_call_tool(at, scope=ToolScope.NODE, nodes=["z"])  # second call is a no-op
    assert len(at["tools"]) == before
    assert at["tools_params"]["end_call"]["scope"] == "global"  # original scope preserved


def test_inject_end_call_handles_str_tools_list():
    at = _inject_end_call_tool({"tools": "[]", "tools_params": {}}, scope=ToolScope.GLOBAL, nodes=[])
    assert isinstance(at["tools"], list)
    assert at["tools"][0]["function"]["name"] == "end_call"
