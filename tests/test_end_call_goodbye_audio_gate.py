"""
Regression test for the end_call goodbye being dropped from recording + transcript.

When the LLM calls the end_call tool and speaks the goodbye in the same completion,
__execute_function_call enters hangup (_enter_hangup_state) before the goodbye audio
flushes. From that point _should_ignore_transcriber_input() is True, so
_listen_transcriber swallows the user's UtteranceEnd and never resets
InterruptionManager.callee_speaking. get_audio_send_status() then keeps returning WAIT
for the goodbye, wait_for_current_message() times out, and the call is torn down before
the goodbye reaches the caller (absent from recording) or the SEND branch that commits
it to history (absent from transcript).

Fix: _enter_hangup_state() releases the audio gate via on_user_speech_ended(), so the
buffered goodbye flushes on the next transmit pass.
"""

import asyncio
import os
import sys
import time
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bolna.agent_manager.task_manager import TaskManager  # noqa: E402
from bolna.agent_manager.interruption_manager import InterruptionManager  # noqa: E402

GOODBYE_SEQ = 9
SPEECH_ENDED = {"data": {"type": "speech_ended"}, "meta_info": {}}
INTERIM = {
    "data": {"type": "interim_transcript_received", "content": "haan boliye"},
    "meta_info": {},
}


def _im_with_speaking_user():
    im = InterruptionManager()
    im.sequence_ids.add(GOODBYE_SEQ)  # goodbye response registered
    im.on_user_speech_started()  # user mid-utterance -> callee_speaking=True
    return im


def _ignore_tm(im, *, end_call_in_progress):
    """Mock TaskManager wired to drive the real _listen_transcriber ignore gate."""
    tm = MagicMock()
    tm.interruption_manager = im
    tm.transcriber_output_queue = asyncio.Queue()
    tm.hangup_triggered = False
    tm._end_call_in_progress = end_call_in_progress
    tm.has_transfer = False
    tm.transcriber_duration = 0
    tm._log_transcriber_connection_error = AsyncMock()
    tm.process_transcriber_request = AsyncMock(return_value=0)
    tm.stream = True
    tm._set_call_details = MagicMock()
    tm._get_next_step = MagicMock(return_value="llm")
    tm._should_ignore_transcriber_input = lambda: TaskManager._should_ignore_transcriber_input(tm)
    return tm


async def _drive_listen_transcriber(tm, messages):
    for m in messages:
        tm.transcriber_output_queue.put_nowait(m)
    tm.transcriber_output_queue.put_nowait({"data": "transcriber_connection_closed", "meta_info": {}})
    await asyncio.wait_for(TaskManager._listen_transcriber(tm), timeout=2.0)


def test_speech_ended_swallowed_during_hangup():
    """Defect mechanism: the late UtteranceEnd cannot reset the gate on its own."""
    im = _im_with_speaking_user()
    tm = _ignore_tm(im, end_call_in_progress=True)
    asyncio.run(_drive_listen_transcriber(tm, [SPEECH_ENDED]))
    assert im.callee_speaking is True  # swallowed -> stays stuck without the fix


def _enter_hangup(im):
    """Drive the real _enter_hangup_state against a stub carrying this InterruptionManager."""
    tm = MagicMock()
    tm.interruption_manager = im
    tm.hangup_decision_at = None
    TaskManager._enter_hangup_state(tm)
    return tm


def test_enter_hangup_releases_audio_gate():
    """Fix: entering hangup opens the gate so the goodbye can flush (SEND)."""
    im = _im_with_speaking_user()
    assert im.get_audio_send_status(GOODBYE_SEQ, history_length=1) == "WAIT"
    _enter_hangup(im)
    assert im.get_audio_send_status(GOODBYE_SEQ, history_length=1) == "SEND"


def test_release_on_production_history_length():
    """Production path (history_length > 2 exercises the grace branch): a stuck-VAD goodbye
    still flushes after hangup. utterance_end_time == -1 (no clean UtteranceEnd for the
    in-flight speech) makes the grace check a no-op, so the gate returns SEND once hangup
    clears callee_speaking."""
    im = _im_with_speaking_user()
    assert im.utterance_end_time == -1
    assert im.get_audio_send_status(GOODBYE_SEQ, history_length=3) == "WAIT"
    _enter_hangup(im)
    assert im.get_audio_send_status(GOODBYE_SEQ, history_length=3) == "SEND"


def test_grace_period_holds_briefly_then_clears():
    """After hangup releases callee_speaking, the grace branch can only delay the goodbye
    by at most incremental_delay, never indefinitely."""
    im = _im_with_speaking_user()
    _enter_hangup(im)

    # Recent UtteranceEnd -> within grace window -> WAIT
    im.utterance_end_time = time.time() * 1000
    assert im.get_audio_send_status(GOODBYE_SEQ, history_length=3) == "WAIT"

    # Older than incremental_delay -> grace elapsed -> SEND (does not stick)
    im.utterance_end_time = time.time() * 1000 - (im.incremental_delay + 50)
    assert im.get_audio_send_status(GOODBYE_SEQ, history_length=3) == "SEND"


def test_gate_still_holds_outside_hangup():
    """Regression: a speaking user still gates audio when no hangup is in progress."""
    im = _im_with_speaking_user()
    assert im.get_audio_send_status(GOODBYE_SEQ, history_length=1) == "WAIT"


def test_transcript_still_ignored_during_hangup():
    """Precondition the fix relies on: while hangup drains the goodbye, transcriber input
    is still dropped, so a late interim cannot start a new turn or clear the mark queue."""
    im = _im_with_speaking_user()
    tm = _ignore_tm(im, end_call_in_progress=True)
    asyncio.run(_drive_listen_transcriber(tm, [INTERIM]))
    assert tm.process_transcriber_request.await_count == 0


if __name__ == "__main__":
    test_speech_ended_swallowed_during_hangup()
    test_enter_hangup_releases_audio_gate()
    test_release_on_production_history_length()
    test_grace_period_holds_briefly_then_clears()
    test_gate_still_holds_outside_hangup()
    test_transcript_still_ignored_during_hangup()
    print("all tests passed")
