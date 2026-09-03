"""Regression: end_call teardown must not cancel the task it is running on.

The end_call tool is handled inside __do_llm_generation, which runs in llm_task. Teardown
then cancelled llm_task before awaiting stop_handler(), so the CancelledError landed on the
hangup instead of on a stray generation, and the input handler was never stopped.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from bolna.agent_manager.task_manager import TaskManager
from bolna.constants import END_CALL_FUNCTION_PREFIX


class _RecordingInputHandler:
    def __init__(self):
        self.stop_calls = 0

    async def stop_handler(self):
        await asyncio.sleep(0)  # a suspension point, like the real sip-trunk drain wait
        self.stop_calls += 1


class _FakeOutputHandler:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def _make_task_manager(input_handler):
    task_manager = TaskManager.__new__(TaskManager)
    task_manager._end_of_conversation_in_progress = False
    task_manager.conversation_ended = False
    task_manager.ended_by_assistant = False
    task_manager.hangup_triggered = False
    task_manager.hangup_message_queued = False
    task_manager.turn_based_conversation = False
    task_manager.llm_task = None
    task_manager.tools = {"input": input_handler, "output": _FakeOutputHandler()}
    task_manager.voicemail_handler = SimpleNamespace(cancel_task=lambda: None)

    async def _already_flushed():
        return None

    task_manager.wait_for_current_message = _already_flushed
    return task_manager


def _end_of_conversation(task_manager):
    return task_manager._TaskManager__process_end_of_conversation()


async def test_teardown_from_inside_the_llm_task_still_stops_the_input_handler():
    handler = _RecordingInputHandler()
    task_manager = _make_task_manager(handler)

    turn = asyncio.create_task(_end_of_conversation(task_manager))
    task_manager.llm_task = turn
    await asyncio.gather(turn, return_exceptions=True)

    assert not turn.cancelled(), "teardown cancelled the task it was running on"
    assert handler.stop_calls == 1
    assert task_manager.conversation_ended


async def test_a_concurrent_llm_task_is_still_cancelled():
    handler = _RecordingInputHandler()
    task_manager = _make_task_manager(handler)
    stray = asyncio.create_task(asyncio.sleep(30))
    task_manager.llm_task = stray

    await _end_of_conversation(task_manager)
    await asyncio.gather(stray, return_exceptions=True)

    assert stray.cancelled()


async def test_end_call_hangup_teardown_is_not_awaited_inline():
    """The end_call tool is handled inside __do_llm_generation's own LLM_GENERATION_TIMEOUT_S
    window. process_call_hangup's teardown can legitimately outlast that window, so it must be
    fired off detached rather than awaited inline here - otherwise a slow-but-normal teardown
    gets cancelled mid-way by the same timeout meant to catch a hung LLM. BOLNA-2563."""
    tm = TaskManager.__new__(TaskManager)
    tm.run_id = "test-run"
    tm.conversation_history = MagicMock()
    tm.hangup_decision_at = None
    tm.interruption_manager = MagicMock()
    tm.wait_for_current_message = AsyncMock()

    hangup_started = asyncio.Event()

    async def _slow_teardown():
        hangup_started.set()
        await asyncio.sleep(0.2)

    tm.process_call_hangup = AsyncMock(side_effect=_slow_teardown)

    meta_info = {"request_id": "r1", "turn_id": 1, "response_uid": "u1"}
    resp = {"model_response": [], "tool_call_id": "tc1", "textual_response": "Goodbye!"}

    start = asyncio.get_event_loop().time()
    await TaskManager._TaskManager__execute_function_call(
        tm, None, None, None, None, None, None, meta_info, None, END_CALL_FUNCTION_PREFIX, **resp
    )
    elapsed = asyncio.get_event_loop().time() - start

    assert elapsed < 0.1  # returned well before the mocked 0.2s teardown finished
    await asyncio.wait_for(hangup_started.wait(), timeout=1.0)  # the detached task did start
