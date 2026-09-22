"""A telephony call whose audio is never confirmed played must still be able to end."""

import asyncio
import time
from types import SimpleNamespace

from bolna.agent_manager.task_manager import TaskManager
from bolna.constants import STALL_HANGUP_HARD_CAP_S
from bolna.enums import HangupReason
from bolna.input_handlers.telephony import TelephonyInputHandler

SILENT = STALL_HANGUP_HARD_CAP_S + 600


class _TickLimit(Exception):
    """Stops the otherwise endless loop after a bounded number of ticks."""


def _telephony_task_manager(last_transmitted_timestamp, *, hang_conversation_after=12, call_terminate=0, age=SILENT):
    """A TaskManager carrying only the state the completion loop reads."""
    tm = TaskManager.__new__(TaskManager)
    now = time.time()
    tm.is_web_based_call = False
    tm.has_transfer = False
    tm.start_time = now - age
    tm.stream_sid_ts = None
    tm.task_config = {
        "task_config": {"call_terminate": call_terminate},
        "tools_config": {"input": {"provider": "plivo"}},
    }
    tm.welcome_message_delay = 0
    tm.last_transmitted_timestamp = last_transmitted_timestamp
    tm.hangup_triggered = False
    tm.conversation_ended = False
    tm.hangup_triggered_at = None
    tm.hangup_mark_event_timeout = 10
    tm.llm_task = None
    tm.execute_function_call_task = None
    tm._s2s_tool_tasks = ()
    tm.mark_event_meta_data = SimpleNamespace(get_audio_playing_until=lambda: 0.0)
    tm.time_since_last_spoken_human_word = now - age
    tm.hang_conversation_after = hang_conversation_after
    tm.response_in_pipeline = False
    tm._synthesis_awaiting_first_audio = False
    tm.repeat_after_silence_seconds = 0
    tm.trigger_user_online_message_after = 6
    tm.asked_if_user_is_still_there = False
    tm.tools = {"input": SimpleNamespace(is_audio_being_played_to_user=lambda: False)}
    tm.hangup_detail = None
    tm.hangups = []
    tm.ended = []

    async def _hangup_after_goodbye(reason):
        tm.hangups.append(reason)

    async def _end_of_conversation():
        tm.ended.append(True)

    tm._hangup_after_goodbye = _hangup_after_goodbye
    tm._TaskManager__process_end_of_conversation = _end_of_conversation
    return tm


async def _drive(tm, monkeypatch, ticks=8):
    """Run the real loop, fast, for a bounded number of ticks."""
    real_sleep = asyncio.sleep
    state = {"n": 0}

    async def fast_sleep(delay, *args, **kwargs):
        state["n"] += 1
        if state["n"] > ticks:
            raise _TickLimit
        return await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", fast_sleep)
    try:
        await TaskManager._TaskManager__check_for_completion(tm)
        return "loop_exited"
    except _TickLimit:
        return "still_spinning"


async def test_silence_hangup_fires_before_the_first_mark_ack(monkeypatch):
    tm = _telephony_task_manager(0)
    assert await _drive(tm, monkeypatch) == "loop_exited"
    assert tm.hangups == [HangupReason.INACTIVITY_TIMEOUT]


async def test_silence_hangup_still_fires_after_an_ack(monkeypatch):
    tm = _telephony_task_manager(time.time() - SILENT)
    assert await _drive(tm, monkeypatch) == "loop_exited"
    assert tm.hangups == [HangupReason.INACTIVITY_TIMEOUT]


async def test_a_young_call_with_no_ack_is_left_alone(monkeypatch):
    tm = _telephony_task_manager(0, age=2)
    assert await _drive(tm, monkeypatch) == "still_spinning"
    assert tm.hangups == []


async def test_playing_welcome_audio_holds_the_silence_clock(monkeypatch):
    tm = _telephony_task_manager(0)
    tm.mark_event_meta_data = SimpleNamespace(get_audio_playing_until=lambda: time.time() + 30)
    assert await _drive(tm, monkeypatch) == "still_spinning"
    assert tm.hangups == []


async def test_setup_before_the_stream_is_ready_does_not_count_as_silence(monkeypatch):
    # The websocket connects before the carrier's stream can carry audio; the agent is not
    # silent during that window.
    tm = _telephony_task_manager(0, age=SILENT)
    tm.stream_sid_ts = time.time() * 1000
    assert await _drive(tm, monkeypatch) == "still_spinning"
    assert tm.hangups == []


async def test_a_welcome_still_being_synthesized_is_not_silence(monkeypatch):
    # The welcome carries sequence_id -1, so nothing else marks the pipeline busy for it.
    tm = _telephony_task_manager(0)
    tm.stream_sid_ts = (time.time() - 30) * 1000
    tm._synthesis_awaiting_first_audio = True
    assert await _drive(tm, monkeypatch) == "still_spinning"
    assert tm.hangups == []


async def test_a_welcome_that_never_arrives_still_hits_the_stall_backstop(monkeypatch):
    tm = _telephony_task_manager(0)
    tm.stream_sid_ts = (time.time() - SILENT) * 1000
    tm._synthesis_awaiting_first_audio = True
    assert await _drive(tm, monkeypatch) == "loop_exited"
    assert tm.hangups == [HangupReason.INACTIVITY_TIMEOUT]


async def test_call_terminate_caps_a_telephony_call(monkeypatch):
    tm = _telephony_task_manager(time.time(), hang_conversation_after=0, call_terminate=600, age=900)
    assert await _drive(tm, monkeypatch) == "loop_exited"
    assert tm.ended == [True]
    assert tm.hangup_detail == HangupReason.MAX_DURATION_REACHED


async def test_call_terminate_does_not_cut_a_call_below_its_cap(monkeypatch):
    tm = _telephony_task_manager(time.time(), hang_conversation_after=0, call_terminate=600, age=120)
    assert await _drive(tm, monkeypatch) == "still_spinning"
    assert tm.ended == []


async def test_a_chat_session_is_never_capped(monkeypatch):
    # call_terminate is a call cap; a turn-based chat has no call to end.
    tm = _telephony_task_manager(time.time(), hang_conversation_after=0, call_terminate=600, age=900)
    tm.task_config["tools_config"]["input"]["provider"] = "default"
    assert await _drive(tm, monkeypatch) == "still_spinning"
    assert tm.ended == []


async def test_a_transferred_call_is_never_capped(monkeypatch):
    # The media leg stays up while two humans talk, and bolna must not end that.
    tm = _telephony_task_manager(time.time(), hang_conversation_after=0, call_terminate=600, age=900)
    tm.has_transfer = True
    assert await _drive(tm, monkeypatch) == "still_spinning"
    assert tm.ended == []


async def test_an_unset_call_terminate_caps_nothing(monkeypatch):
    # Absent must not be read as the schema default, which would cut every call at 90s.
    tm = _telephony_task_manager(time.time(), hang_conversation_after=0, call_terminate=0, age=90000)
    assert await _drive(tm, monkeypatch) == "still_spinning"
    assert tm.ended == []


def test_hangup_uses_the_known_sid_when_no_start_event_arrived():
    handler = TelephonyInputHandler.__new__(TelephonyInputHandler)
    handler.call_sid = None
    handler._fallback_call_sid = None

    assert handler.get_call_sid() is None
    handler.set_fallback_call_sid("known-sid")
    assert handler.get_call_sid() == "known-sid"
