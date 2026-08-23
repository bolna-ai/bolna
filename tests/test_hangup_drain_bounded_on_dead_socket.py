"""The hangup drain must never outlive the audio it is waiting on.

Run 1e4980b0 (vobiz, 2026-08-13) logged "final hangup chunk has not been sent yet" every 500ms
for 40 minutes: the media socket was already dead, so the mark ack that flips hangup_sent() could
never arrive. Teardown never reached the final save and a full TaskManager stayed pinned until the
pod died. The drain now carries its own deadline, and the completion watchdog no longer goes blind
on a call where nothing was ever acked.
"""

import asyncio
import time

from bolna.agent_manager.task_manager import TaskManager
from bolna.helpers.conversation_history import ConversationHistory

GOODBYE = "Thank you for your time. Goodbye."


class _OutputStub:
    def __init__(self, *, acked=False, closed=False):
        self._acked = acked
        self._closed = closed
        self.hangup_sent_forced = False

    def hangup_sent(self):
        return self._acked

    def is_closed(self):
        return self._closed

    def set_hangup_sent(self):
        self.hangup_sent_forced = True
        self._acked = True

    def close(self):
        self._closed = True


class _InputStub:
    def __init__(self):
        self.stopped = False

    async def stop_handler(self):
        self.stopped = True


class _MarksStub:
    """Holds one goodbye chunk the provider has been sent but has not acked."""

    def __init__(self, duration):
        self.mark_event_meta_data = {
            "aud1": {"type": "agent_hangup", "duration": duration, "sent_ts": time.time()}
        }


class _VoicemailStub:
    def cancel_task(self):
        pass


def _make_tm(output, *, pending_duration=0.0, grace=0.4):
    tm = TaskManager.__new__(TaskManager)
    tm.tools = {"output": output, "input": _InputStub()}  # no transcriber: skips the 2s settle
    tm.conversation_history = ConversationHistory()
    tm.call_hangup_message_config = GOODBYE  # call_hangup_message is a language-aware property
    tm.language = "en"
    tm.hangup_triggered = True
    tm.hangup_message_queued = True
    tm.hangup_mark_event_timeout = grace
    tm.conversation_ended = False
    tm.ended_by_assistant = False
    tm._end_of_conversation_in_progress = False
    tm.llm_task = None
    tm.turn_based_conversation = False
    tm.voicemail_handler = _VoicemailStub()
    tm.mark_event_meta_data = _MarksStub(pending_duration)

    async def _already_flushed():
        return

    tm.wait_for_current_message = _already_flushed  # covered by its own test module
    return tm


async def _teardown(tm):
    # asyncio.wait_for is the assertion: an unbounded drain fails here instead of hanging the suite.
    await asyncio.wait_for(tm._TaskManager__process_end_of_conversation(), timeout=5.0)


async def test_dead_output_socket_aborts_the_drain_immediately():
    tm = _make_tm(_OutputStub(closed=True), pending_duration=30.0, grace=10.0)

    start = time.monotonic()
    await _teardown(tm)

    assert time.monotonic() - start < 0.5  # not the 40s the queued audio would imply
    assert tm.conversation_ended is True
    assert tm.tools["input"].stopped is True


async def test_drain_is_bounded_when_the_ack_never_arrives():
    # Live-looking socket that simply stops acking — the half-dead case is_closed() cannot see.
    tm = _make_tm(_OutputStub(), pending_duration=0.1, grace=0.4)

    start = time.monotonic()
    await _teardown(tm)
    elapsed = time.monotonic() - start

    assert 0.4 < elapsed < 2.0  # waited out queued audio + grace, then gave up
    assert tm.conversation_ended is True


async def test_drain_returns_as_soon_as_the_ack_lands():
    output = _OutputStub()
    tm = _make_tm(output, pending_duration=5.0, grace=10.0)

    async def _ack_soon():
        await asyncio.sleep(0.05)
        output._acked = True

    acker = asyncio.create_task(_ack_soon())
    start = time.monotonic()
    await _teardown(tm)
    await acker

    assert time.monotonic() - start < 2.0  # broke on the ack, not on the 15s deadline
    assert tm.history[-1]["content"] == GOODBYE
    assert tm.history[-1]["message_category"] == "agent_hangup"


async def test_watchdog_forces_teardown_when_no_mark_was_ever_acked():
    # last_transmitted_timestamp stays 0 for a call whose socket died before the first ack.
    # That idle guard used to swallow the hangup branch below it, so nothing ever forced the ack.
    output = _OutputStub()
    tm = TaskManager.__new__(TaskManager)
    tm.tools = {"output": output}
    tm.is_web_based_call = False
    tm.last_transmitted_timestamp = 0
    tm.hangup_triggered = True
    tm.conversation_ended = False
    tm.hangup_triggered_at = None  # goodbye synth never finished, so this was never stamped
    tm.hangup_decision_at = time.time() - 100
    tm.hangup_mark_event_timeout = 10

    ended = asyncio.Event()

    async def _fake_eoc():
        tm.conversation_ended = True
        ended.set()

    tm._TaskManager__process_end_of_conversation = _fake_eoc

    await asyncio.wait_for(tm._TaskManager__check_for_completion(), timeout=10.0)

    assert output.hangup_sent_forced is True
    assert ended.is_set()
