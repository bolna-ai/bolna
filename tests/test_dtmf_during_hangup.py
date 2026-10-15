"""Key presses follow the same gate as speech while the call is hanging up."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from bolna.agent_manager.task_manager import TaskManager


def _tm(hangup_triggered, interruptible_window):
    tm = TaskManager.__new__(TaskManager)
    tm.queues = {"dtmf": asyncio.Queue()}
    tm.dtmf_events = []
    tm.conversation_start_init_ts = 0
    tm.tools = {"input": MagicMock(io_provider="plivo")}
    tm.has_transfer = False
    tm.hangup_triggered = hangup_triggered
    tm._end_call_in_progress = False
    tm.conversation_ended = False
    tm._hangup_interruptible_window = interruptible_window
    tm._TaskManager__get_updated_meta_info = lambda meta_info: {**meta_info, "sequence_id": 1}
    tm._handle_transcriber_output = AsyncMock()
    return tm


async def _press(tm, digits):
    tm.queues["dtmf"].put_nowait(digits)
    listener = asyncio.create_task(tm.inject_digits_to_conversation())
    await asyncio.sleep(0.01)
    listener.cancel()


async def test_a_key_press_cannot_cut_a_goodbye_the_caller_cannot_talk_over():
    tm = _tm(hangup_triggered=True, interruptible_window=False)
    await _press(tm, "1")

    tm._handle_transcriber_output.assert_not_awaited()
    assert [e["digit"] for e in tm.dtmf_events] == ["1"]  # still recorded


async def test_a_key_press_interrupts_a_goodbye_the_caller_may_talk_over():
    tm = _tm(hangup_triggered=True, interruptible_window=True)
    await _press(tm, "1")

    tm._handle_transcriber_output.assert_awaited_once()
