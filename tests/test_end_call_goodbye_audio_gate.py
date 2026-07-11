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


def test_enter_hangup_releases_audio_gate():
    """Fix: entering hangup opens the gate so the goodbye can flush (SEND)."""
    im = _im_with_speaking_user()
    assert im.get_audio_send_status(GOODBYE_SEQ, history_length=1) == "WAIT"

    tm = MagicMock()
    tm.interruption_manager = im
    tm.hangup_decision_at = None
    TaskManager._enter_hangup_state(tm)

    assert im.get_audio_send_status(GOODBYE_SEQ, history_length=1) == "SEND"


def test_gate_still_holds_outside_hangup():
    """Regression: a speaking user still gates audio when no hangup is in progress."""
    im = _im_with_speaking_user()
    assert im.get_audio_send_status(GOODBYE_SEQ, history_length=1) == "WAIT"


def test_transcript_still_ignored_during_hangup():
    """Regression: no new turn is started from transcriber input after end_call."""
    im = _im_with_speaking_user()
    tm = _ignore_tm(im, end_call_in_progress=True)
    asyncio.run(_drive_listen_transcriber(tm, [INTERIM]))
    assert tm.process_transcriber_request.await_count == 0


if __name__ == "__main__":
    test_speech_ended_swallowed_during_hangup()
    test_enter_hangup_releases_audio_gate()
    test_gate_still_holds_outside_hangup()
    test_transcript_still_ignored_during_hangup()
    print("all tests passed")
