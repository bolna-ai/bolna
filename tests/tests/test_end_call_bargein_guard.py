"""Guards the end_call hangup actuation against barge-in resurrection.

Reproduces the failure where the end_call tool fired and generated a goodbye,
but a user barge-in cancelled the conversation turn task before the disconnect
ran. hangup_triggered was never set, so the transcriber kept feeding new turns
and the agent looped goodbyes until the user dropped (actual_hangup_reason
stayed null in the experiment outcome).

The fix: _end_call_in_progress is set the instant the end_call tool fires
(before the goodbye is generated), and _listen_transcriber drops user speech
while a hangup or end_call actuation is underway, so barge-in can no longer
cancel the in-flight hangup.
"""

import asyncio
import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from bolna.agent_manager.task_manager import TaskManager


def _ignore(hangup_triggered, end_call_in_progress, has_transfer=False):
    fake = SimpleNamespace(
        hangup_triggered=hangup_triggered,
        _end_call_in_progress=end_call_in_progress,
        has_transfer=has_transfer,
    )
    return TaskManager._should_ignore_transcriber_input(fake)


def test_ignores_input_during_end_call_actuation():
    # end_call fired but hangup_triggered not yet set (goodbye still generating).
    assert _ignore(hangup_triggered=False, end_call_in_progress=True) is True


def test_ignores_input_after_hangup_locked():
    assert _ignore(hangup_triggered=True, end_call_in_progress=False) is True


def test_processes_input_during_normal_conversation():
    assert _ignore(hangup_triggered=False, end_call_in_progress=False) is False


# An interim transcript that crosses the interruption threshold. In the real
# call this is what fired "Condition for interruption hit" -> __cleanup_downstream_tasks
# -> "Cancelling LLM Task", killing the in-flight end_call handler.
_INTERIM_BARGEIN = {
    "data": {"type": "interim_transcript_received", "content": "कोई order ही नहीं"},
    "meta_info": {"io": "plivo", "sequence_id": 2},
}


def _make_tm(*, end_call_in_progress, hangup_triggered):
    tm = MagicMock()
    tm.hangup_triggered = hangup_triggered
    tm._end_call_in_progress = end_call_in_progress
    tm.has_transfer = False
    tm.stream = True
    tm.response_in_pipeline = False
    tm.transcriber_output_queue = asyncio.Queue()
    tm.process_transcriber_request = AsyncMock(return_value=0)
    tm._set_call_details = MagicMock()
    tm._get_next_step = MagicMock(return_value="llm")
    tm.tools = {"input": MagicMock(), "transcriber": MagicMock()}
    tm.tools["input"].welcome_message_played = MagicMock(return_value=True)
    tm.conversation_history.is_duplicate_user = MagicMock(return_value=False)
    tm.interruption_manager.should_trigger_interruption = MagicMock(return_value=True)
    tm._TaskManager__cleanup_downstream_tasks = AsyncMock()
    tm._end_call_on_component_error = AsyncMock()
    tm.task_config = {"tools_config": {"transcriber": {"provider": "deepgram"}}}
    tm._should_ignore_transcriber_input = TaskManager._should_ignore_transcriber_input.__get__(tm, TaskManager)
    tm._listen_transcriber = TaskManager._listen_transcriber.__get__(tm, TaskManager)
    return tm


async def _drive_with_bargein(tm):
    await tm.transcriber_output_queue.put(_INTERIM_BARGEIN)
    try:
        await asyncio.wait_for(tm._listen_transcriber(), timeout=0.3)
    except asyncio.TimeoutError:
        pass


@pytest.mark.asyncio
async def test_bargein_does_not_cancel_turn_during_end_call():
    tm = _make_tm(end_call_in_progress=True, hangup_triggered=False)
    await _drive_with_bargein(tm)
    tm._set_call_details.assert_not_called()
    tm._TaskManager__cleanup_downstream_tasks.assert_not_called()


@pytest.mark.asyncio
async def test_bargein_cancels_turn_during_normal_conversation():
    tm = _make_tm(end_call_in_progress=False, hangup_triggered=False)
    await _drive_with_bargein(tm)
    tm._TaskManager__cleanup_downstream_tasks.assert_awaited_once()


class TestSourceGuards:
    """Catch accidental removal of the wiring that closes the barge-in race."""

    def test_end_call_branch_sets_in_progress_flag(self):
        src = inspect.getsource(TaskManager._TaskManager__execute_function_call)
        assert "_end_call_in_progress = True" in src, (
            "end_call branch must set _end_call_in_progress before generating the goodbye"
        )

    def test_listen_transcriber_uses_the_guard(self):
        src = inspect.getsource(TaskManager._listen_transcriber)
        assert "_should_ignore_transcriber_input" in src, (
            "_listen_transcriber must drop user speech while a hangup/end_call actuation is underway"
        )
