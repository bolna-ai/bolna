"""ACKing chunks must not close the interruptible goodbye window during playout."""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import bolna.agent_manager.task_manager as task_manager_module
from bolna.agent_manager.task_manager import TaskManager
from bolna.helpers.mark_event_meta_data import MarkEventMetaData


@pytest.fixture
def playout(monkeypatch):
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(task_manager_module, "time", SimpleNamespace(time=lambda: clock.now))
    tm = TaskManager.__new__(TaskManager)
    tm.tools = {"input": SimpleNamespace(input_stream_ended=False)}
    tm.conversation_ended = False
    tm.hangup_mark_event_timeout = 10
    tm._turn_audio_flushed = asyncio.Event()
    tm._turn_audio_flushed.set()
    tm.mark_event_meta_data = MarkEventMetaData()

    steps = []
    timeouts = []
    real_wait_for = asyncio.wait_for

    async def wait_for(awaitable, timeout):
        timeouts.append(timeout)
        return await real_wait_for(awaitable, timeout=timeout)

    async def mark_changed():
        assert steps, "unexpected extra playout wait"
        steps.pop(0)()

    monkeypatch.setattr(
        task_manager_module,
        "asyncio",
        SimpleNamespace(wait_for=wait_for, TimeoutError=asyncio.TimeoutError),
    )
    tm.mark_event_meta_data.mark_changed = SimpleNamespace(clear=lambda: None, set=lambda: None, wait=mark_changed)

    def queue(mark_id, duration):
        tm.mark_event_meta_data.update_data(mark_id, {"type": "audio", "duration": duration, "sent_ts": clock.now})

    return tm, clock, steps, timeouts, queue


@pytest.mark.parametrize("barge_in", [False, True])
async def test_long_goodbye_stays_interruptible_until_played_or_cancelled(playout, monkeypatch, barge_in):
    tm, clock, steps, timeouts, queue = playout
    for index in range(15):
        queue(str(index), 1)

    tm.check_if_user_online = True
    tm.interruptible_hangup_message = True
    tm._hangup_interruptible_window = False
    tm.run_id = "test-run"
    tm.conversation_history = MagicMock()
    tm._enter_hangup_state = MagicMock()
    tm._TaskManager__arm_end_call_teardown = MagicMock()
    monkeypatch.setattr(task_manager_module, "convert_to_request_log", lambda *a, **kw: None)

    def ack_most_chunks():
        clock.now = 113
        for index in range(13):
            tm.mark_event_meta_data.fetch_data(str(index))

    def finish():
        # The old deadline was 100 + 2 remaining seconds + 10 grace = 112,
        # so end_call had already closed the window before this caller onset.
        clock.now = 114 if barge_in else 115
        assert tm._hangup_interruptible_window is True
        tm._TaskManager__arm_end_call_teardown.assert_not_called()
        if barge_in:
            tm._hangup_cancelled = True
            tm.mark_event_meta_data.clear_data()
        else:
            tm.mark_event_meta_data.fetch_data("13")
            tm.mark_event_meta_data.fetch_data("14")

    steps.extend([ack_most_chunks, finish])
    await tm._TaskManager__execute_function_call(
        url=None,
        method=None,
        param=None,
        api_token=None,
        headers=None,
        model_args=None,
        meta_info={"turn_id": 1, "response_uid": "reply"},
        next_step="synthesizer",
        called_fun="end_call",
        textual_response="A long final message. Goodbye!",
        model_response=None,
        tool_call_id="end-call",
    )

    assert timeouts == [3.0, 25.0, 12.0]
    assert not steps
    assert tm._hangup_interruptible_window is False
    assert tm._TaskManager__arm_end_call_teardown.call_count == (0 if barge_in else 1)


async def test_missing_ack_keeps_original_deadline_after_partial_playback(playout):
    tm, clock, steps, timeouts, queue = playout
    queue("first", 12)
    queue("last", 3)

    def ack_first():
        clock.now = 112
        tm.mark_event_meta_data.fetch_data("first")

    def expire():
        clock.now += timeouts[-1]

    steps.extend([ack_first, expire])
    await tm.wait_for_current_message()

    assert clock.now == 125  # Full 15s of audio plus 10s grace, once.
    assert timeouts == [3.0, 25.0, 13.0]
    assert "last" in tm.mark_event_meta_data.mark_event_meta_data
    assert not steps


async def test_late_audio_extends_deadline_once_even_after_prior_marks_ack(playout):
    tm, clock, steps, timeouts, queue = playout
    queue("first", 2)

    def queue_late_audio():
        clock.now = 105
        tm.mark_event_meta_data.fetch_data("first")
        queue("late", 15)

    def wake_without_ack():
        clock.now = 110

    def expire():
        clock.now += timeouts[-1]

    steps.extend([queue_late_audio, wake_without_ack, expire])
    await tm.wait_for_current_message()

    assert clock.now == 130  # Late audio ends at 120, then one grace period.
    assert timeouts == [3.0, 12.0, 25.0, 20.0]
    assert not steps
