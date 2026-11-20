from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from bolna.agent_manager.task_manager import TaskManager


def _make_followup_task_manager():
    """Build the minimum TaskManager state needed to exercise run() teardown."""
    task_manager = TaskManager.__new__(TaskManager)
    task_manager._is_conversation_task = lambda: False
    task_manager._process_followup_task = AsyncMock()

    task_manager.task_config = {"task_type": "webhook"}
    task_manager.webhook_response = {"status": "ok"}
    task_manager.input_parameters = {}
    task_manager.run_id = None
    task_manager.task_id = "test-task"

    for attribute in (
        "llm_task",
        "first_message_task_new",
        "llm_queue_task",
        "execute_function_call_task",
        "_lid_idle_watcher_task",
        "handoff_prewarm_task",
        "_post_switch_fallback_task",
    ):
        setattr(task_manager, attribute, None)

    task_manager.handoff_audio_cache = {"cached": b"audio"}
    task_manager.observable_variables = {}
    task_manager.tools = {}
    task_manager.kwargs = {"task_manager_instance": task_manager}
    task_manager.conversation_recording = {"input": {"data": b"input"}}
    task_manager.conversation_history = object()
    task_manager.request_logs = ["request"]
    task_manager.function_tool_api_call_details = [{"name": "tool"}]
    return task_manager


@pytest.mark.asyncio
async def test_run_propagates_followup_task_error_after_cleanup():
    task_manager = _make_followup_task_manager()
    task_manager._process_followup_task.side_effect = RuntimeError("provider failed")

    with pytest.raises(RuntimeError, match="provider failed"):
        await TaskManager.run(task_manager)

    assert task_manager.handoff_audio_cache == {}
    assert task_manager.conversation_history is None
    assert task_manager.request_logs == []
    assert task_manager.function_tool_api_call_details == []


@pytest.mark.asyncio
async def test_run_returns_followup_output_after_cleanup():
    task_manager = _make_followup_task_manager()
    tool = SimpleNamespace(task_manager_instance=task_manager)
    task_manager.tools["output"] = tool

    output = await TaskManager.run(task_manager)

    assert output == {"status": {"status": "ok"}, "task_type": "webhook"}
    task_manager._process_followup_task.assert_awaited_once_with()
    assert tool.task_manager_instance is None
    assert task_manager.tools == {}
