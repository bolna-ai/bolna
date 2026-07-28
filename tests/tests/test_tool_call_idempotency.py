"""Regression tests for BLT-018 — an opt-in once-per-call tool re-emitted on rapid closing acks
must not re-hit its external API (and its pre-call filler must not be re-spoken).

Covers the success predicate and the guard inside __execute_function_call (trigger_api patched):
a once-per-call tool fires at most once regardless of args, a FAILED first call still permits a
retry, and a tool not marked idempotent is never deduped.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bolna.agent_manager.task_manager import TaskManager


MOD = "bolna.agent_manager.task_manager"


# ---------------------------------------------------------------- success predicate


def test_execution_succeeded_only_on_clean_2xx():
    ok = TaskManager._tool_execution_succeeded
    assert ok({"status_code": 200, "error": None}) is True
    assert ok({"status_code": 201, "error": None}) is True
    assert ok({"status_code": 200, "error": "Request timed out"}) is False  # trigger_api error body
    assert ok({"status_code": 500, "error": None}) is False
    assert ok({"status_code": None, "error": None}) is False


def test_is_once_per_call_reads_opt_in_flag():
    tm = TaskManager.__new__(TaskManager)
    tm.kwargs = {"api_tools": {"tools_params": {"a": {"idempotent": True}, "b": {}}}}
    assert tm._is_once_per_call("a") is True
    assert tm._is_once_per_call("b") is False
    assert tm._is_once_per_call("missing") is False


# ---------------------------------------------------------------- the guard in __execute_function_call


def _make_task_manager(idempotent):
    tm = TaskManager.__new__(TaskManager)
    tm.run_id = "run-1"
    tm.check_if_user_online = True
    tm.hangup_triggered = False
    tm.conversation_ended = False
    tm.executed_once_per_call_tools = set()
    tm.context_data = {}
    tm.tools = {}
    tm.execute_function_call_task = "sentinel"
    tm.conversation_config = {"check_if_user_online": True}
    tm.llm_config = {"model": "gpt-4o-mini"}
    tm.kwargs = {
        "api_tools": {
            "tools_params": {
                "custom_task_reschedule_call": {"url": "http://x", "method": "POST", "idempotent": idempotent}
            }
        }
    }
    tm.conversation_history = MagicMock()
    tm.conversation_history.get_copy.return_value = []
    tm.wait_for_current_message = AsyncMock()
    tm._extract_api_call_runtime_args = MagicMock(return_value={})
    tm._start_api_call_detail = MagicMock(return_value=None)
    tm._finalize_api_call_detail = MagicMock()
    tm._spawn_followup_meta_info = MagicMock(return_value={"turn_id": 9, "response_uid": "u9"})
    tm._TaskManager__is_graph_agent = MagicMock(return_value=False)
    tm._TaskManager__do_llm_generation = AsyncMock()
    return tm


async def _run_tool(tm, mentioned_time="later"):
    await tm._TaskManager__execute_function_call(
        url="http://x",
        method="POST",
        param={},
        api_token=None,
        headers={},
        model_args={},
        meta_info={"turn_id": 8, "response_uid": "u8"},
        next_step=None,
        called_fun="custom_task_reschedule_call",
        execution_id="run-1",
        function_name="reschedule_call",
        mentionedTime=mentioned_time,
        model_response=[{"id": "call_1"}],
        tool_call_id="call_1",
    )


def _patchers(trigger):
    # Pass explicit mocks (patch.multiple only auto-creates for DEFAULT), so assert on `trigger`.
    return patch.multiple(
        MOD,
        trigger_api=trigger,
        prepare_api_request=MagicMock(return_value={"request_body": None, "api_params": None, "headers": {}}),
        computed_api_response=AsyncMock(return_value=([], [])),
        format_messages=MagicMock(return_value=[]),
        convert_to_request_log=MagicMock(),
    )


OK_RESPONSE = {"status_code": 200, "error": None, "body": "{}", "content_type": "application/json"}


@pytest.mark.asyncio
async def test_second_call_to_once_per_call_tool_skips_api():
    tm = _make_task_manager(idempotent=True)
    trigger = AsyncMock(return_value=OK_RESPONSE)
    with _patchers(trigger):
        await _run_tool(tm)  # first call executes
        await _run_tool(tm)  # second (same tool) -> skipped
    assert trigger.await_count == 1  # API hit only once
    assert "custom_task_reschedule_call" in tm.executed_once_per_call_tools
    # skip path fed the result back through a follow-up and kept the "are you still there" timer alive
    assert tm.check_if_user_online is True
    assert tm._TaskManager__do_llm_generation.await_count == 2


@pytest.mark.asyncio
async def test_once_per_call_dedupes_even_with_different_args():
    # once-per-call is by tool name: a re-emission with different args is still skipped, and the
    # pre-call filler (suppressed in the stream loop by the same membership) won't be re-spoken.
    tm = _make_task_manager(idempotent=True)
    trigger = AsyncMock(return_value=OK_RESPONSE)
    with _patchers(trigger):
        await _run_tool(tm, mentioned_time="later")
        await _run_tool(tm, mentioned_time="tomorrow 5pm")
    assert trigger.await_count == 1


@pytest.mark.asyncio
async def test_failed_once_per_call_permits_retry():
    tm = _make_task_manager(idempotent=True)
    # trigger_api returns an error body (timeout) rather than raising -> must NOT record as done.
    err = {"status_code": 200, "error": "Request timed out", "body": "ERROR", "content_type": "text/plain"}
    trigger = AsyncMock(return_value=err)
    with _patchers(trigger):
        await _run_tool(tm)
        await _run_tool(tm)
    assert trigger.await_count == 2  # retry actually re-hits the API
    assert tm.executed_once_per_call_tools == set()


@pytest.mark.asyncio
async def test_non_idempotent_tool_never_deduped():
    tm = _make_task_manager(idempotent=False)
    trigger = AsyncMock(return_value=OK_RESPONSE)
    with _patchers(trigger):
        await _run_tool(tm)
        await _run_tool(tm)
    assert trigger.await_count == 2  # repeatable tool runs both times
    assert tm.executed_once_per_call_tools == set()
