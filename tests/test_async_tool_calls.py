"""run_async custom tools: the turn carries on with a placeholder and the late result replaces it in history."""

import asyncio
import json
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bolna.agent_manager.task_manager import TaskManager
from bolna.constants import ASYNC_TOOL_SUBMITTED_RESULT
from bolna.helpers.conversation_history import ConversationHistory
from bolna.models import ToolModel

TOOL = "create_lead"
TOOL_CALL_ID = "call_abc123"
API_BODY = '{"lead_id": "L-42", "status": "created"}'


class _SlowApi:
    """trigger_api stand-in that blocks until released, recording what happened before it returned."""

    def __init__(self):
        self.release = asyncio.Event()
        self.started = asyncio.Event()
        self.calls = 0

    async def __call__(self, **kwargs):
        self.calls += 1
        self.started.set()
        await self.release.wait()
        return {"body": API_BODY, "status_code": 200, "content_type": "application/json", "error": None}


def _make_self(tool_conf, graph_agent=False):
    history = ConversationHistory([{"role": "system", "content": "sys"}, {"role": "user", "content": "add me"}])
    history.append_assistant("Sure.", turn_id=1)
    me = SimpleNamespace(
        check_if_user_online=True,
        run_id="run-1",
        kwargs={"api_tools": {"tools_params": {TOOL: tool_conf}}},
        hangup_triggered=False,
        conversation_ended=False,
        _caller_disconnected=MagicMock(return_value=False),
        wait_for_current_message=AsyncMock(),
        fire_pre_call_webhook=MagicMock(),
        conversation_history=history,
        llm_config={"model": "gpt-test"},
        conversation_config={},
        context_data={},
        tools={"llm_agent": SimpleNamespace(context_data={})},
        _async_tool_tasks=set(),
        execute_function_call_task=None,
        function_tool_api_call_details=[],
        _spawn_followup_meta_info=lambda meta: dict(meta),
        _extract_api_call_runtime_args=MagicMock(return_value={}),
        _start_api_call_detail=MagicMock(return_value={}),
        _finalize_api_call_detail=MagicMock(),
        _TaskManager__is_graph_agent=MagicMock(return_value=graph_agent),
    )
    for name in (
        "_call_api_tool",
        "_merge_tool_response_into_context",
        "_log_tool_response",
        "_start_async_tool_call",
        "_run_async_tool_call",
        "_on_async_tool_done",
    ):
        setattr(me, name, types.MethodType(getattr(TaskManager, name), me))
    return me


def _execute(me):
    return TaskManager._TaskManager__execute_function_call(
        me,
        "https://crm.example/leads",  # url
        "POST",  # method
        json.dumps({"name": "%(name)s"}),  # param
        None,  # api_token
        None,  # headers
        {},  # model_args
        {"turn_id": 1, "sequence_id": 1},  # meta_info
        "synthesizer",  # next_step
        TOOL,  # called_fun
        model_response=[{"id": TOOL_CALL_ID, "type": "function", "function": {"name": TOOL, "arguments": "{}"}}],
        tool_call_id=TOOL_CALL_ID,
        name="Asha",
        textual_response=None,
    )


def _tool_results(history):
    return [m["content"] for m in history.messages if m.get("role") == "tool" and m.get("tool_call_id") == TOOL_CALL_ID]


@pytest.fixture
def slow_api():
    api = _SlowApi()
    with (
        patch("bolna.agent_manager.task_manager.trigger_api", new=api),
        patch("bolna.agent_manager.task_manager.convert_to_request_log"),
        patch("bolna.agent_manager.task_manager.format_messages"),
    ):
        yield api


async def test_async_tool_follow_up_runs_before_the_api_returns(slow_api):
    me = _make_self({"url": "https://crm.example/leads", "run_async": True})
    seen_by_follow_up = []
    me._TaskManager__do_llm_generation = AsyncMock(
        side_effect=lambda messages, *a, **k: seen_by_follow_up.append((slow_api.release.is_set(), messages))
    )

    await asyncio.wait_for(_execute(me), timeout=1)

    me._TaskManager__do_llm_generation.assert_awaited_once()
    released, messages = seen_by_follow_up[0]
    assert not released
    assert {"role": "tool", "tool_call_id": TOOL_CALL_ID, "content": ASYNC_TOOL_SUBMITTED_RESULT} in messages
    assert len(me._async_tool_tasks) == 1

    slow_api.release.set()
    await asyncio.gather(*me._async_tool_tasks)

    assert slow_api.calls == 1
    assert _tool_results(me.conversation_history) == [API_BODY]
    assert not me._async_tool_tasks


async def test_sync_tool_still_waits_for_the_api(slow_api):
    me = _make_self({"url": "https://crm.example/leads"})
    me._TaskManager__do_llm_generation = AsyncMock()

    execution = asyncio.create_task(_execute(me))
    await asyncio.wait_for(slow_api.started.wait(), timeout=1)
    await asyncio.sleep(0)
    me._TaskManager__do_llm_generation.assert_not_awaited()

    slow_api.release.set()
    await asyncio.wait_for(execution, timeout=1)

    messages = me._TaskManager__do_llm_generation.await_args.args[0]
    assert {"role": "tool", "tool_call_id": TOOL_CALL_ID, "content": API_BODY} in messages
    assert not me._async_tool_tasks


async def test_async_result_merges_into_graph_context(slow_api):
    me = _make_self({"url": "https://crm.example/leads", "run_async": True}, graph_agent=True)
    me._TaskManager__do_llm_generation = AsyncMock()

    await asyncio.wait_for(_execute(me), timeout=1)
    assert "lead_id" not in me.context_data

    slow_api.release.set()
    await asyncio.gather(*me._async_tool_tasks)

    assert me.context_data["lead_id"] == "L-42"
    assert me.tools["llm_agent"].context_data["lead_id"] == "L-42"


async def test_async_tool_is_skipped_once_the_call_is_over(slow_api):
    me = _make_self({"url": "https://crm.example/leads", "run_async": True})
    me.conversation_ended = True
    me._TaskManager__do_llm_generation = AsyncMock()

    await asyncio.wait_for(_execute(me), timeout=1)

    assert slow_api.calls == 0
    assert not me._async_tool_tasks
    me._TaskManager__do_llm_generation.assert_not_awaited()


def test_replace_tool_result_updates_history_and_interim_only_for_that_call():
    history = ConversationHistory([{"role": "user", "content": "hi"}])
    history.append_assistant(None, tool_calls=[{"id": "a"}, {"id": "b"}])
    history.append_tool_result("a", "pending a")
    history.append_tool_result("b", "pending b")
    history.sync_interim()

    assert history.replace_tool_result("a", "done a")
    assert [m["content"] for m in history.messages if m.get("role") == "tool"] == ["done a", "pending b"]
    assert [m["content"] for m in history.interim if m.get("role") == "tool"] == ["done a", "pending b"]
    assert not history.replace_tool_result("missing", "x")
    assert not history.replace_tool_result("", "x")


def test_run_async_survives_tool_config_validation():
    tools = ToolModel(tools_params={TOOL: {"url": "https://crm.example/leads", "run_async": True}, "other": {}})
    assert tools.tools_params[TOOL].run_async is True
    assert tools.tools_params["other"].run_async is False
