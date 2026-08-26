"""A hangup goodbye must survive the transcriber socket idle-closing mid-goodbye.

When _listen_transcriber breaks on that close, gather() returns and run()'s terminal sync_history
would otherwise trim the goodbye to the prefix the caller had heard so far. Teardown waits for the
goodbye to drain first.
"""

import asyncio
import inspect
import time


from bolna.agent_manager.task_manager import TaskManager
from bolna.helpers.conversation_history import ConversationHistory

# values from the log
FULL_GOODBYE = (
    "No worries at all, Sir. You can reach out whenever you are free. "
    "Thank you so much for your time. Have a wonderful day ahead. Goodbye. "
)
HEARD_PREFIX = "No worries at all, Sir. You can reach out whenever you are"  # 58 chars
GOODBYE_SENT_TS = 1779789373.8576052
GOODBYE_DURATION = 7.20975
TEARDOWN_TS = 1779789377.264  # ~3.4s into the goodbye


def _content_mark(text, turn_id, response_uid, duration, sent_ts):
    return {
        "type": "",
        "text_synthesized": text,
        "turn_id": turn_id,
        "response_uid": response_uid,
        "duration": duration,
        "sent_ts": sent_ts,
        "counter": 12,
    }


class _InputStub:
    last_heard_turn_id = None
    last_heard_response_uid = None
    response_heard_by_user = ""

    def get_response_heard_for_response(self, _uid):
        return ""

    def get_response_heard_for_turn(self, _turn_id):
        return ""

    def get_current_mark_started_time(self):
        return 0.0

    def reset_response_heard_by_user(self):
        pass


class _MarkMetaStub:
    def __init__(self, marks=None):
        self.pending_marks = dict(marks or {})
        self.mark_changed = asyncio.Event()

    def get_heard_text_for_response(self, _uid):
        return ""

    def get_heard_text_for_turn(self, _turn_id):
        return ""

    def drain(self):
        self.pending_marks.clear()
        self.mark_changed.set()


def _make_tm(history, marks=None):
    tm = TaskManager.__new__(TaskManager)
    tm.conversation_history = history
    tm.tools = {"input": _InputStub()}
    tm.mark_event_meta_data = _MarkMetaStub(marks)
    tm._turn_msg_map = {}
    tm.hangup_triggered = False
    tm.conversation_ended = False
    tm.hangup_mark_event_timeout = 10
    tm._turn_audio_flushed = asyncio.Event()
    tm._turn_audio_flushed.set()
    return tm


def _turn7(history):
    return next(m for m in history.messages if m.get("turn_id") == 7)


async def test_terminal_sync_history_trims_goodbye_to_heard_prefix():
    history = ConversationHistory()
    history.append_user("please send the review later")
    history.append_assistant(FULL_GOODBYE.strip(), turn_id=7, response_uid="r7")
    tm = _make_tm(history)

    marks = [
        ("pre7", {"type": "pre_mark_message", "turn_id": 7, "response_uid": "r7", "counter": 11}),
        ("aud7", _content_mark(FULL_GOODBYE, 7, "r7", GOODBYE_DURATION, GOODBYE_SENT_TS)),
    ]
    await tm.sync_history(marks, interruption_processed_at=TEARDOWN_TS)

    assert _turn7(history)["content"] == HEARD_PREFIX
    assert _turn7(history)["content"] != FULL_GOODBYE.strip()


async def test_draining_before_teardown_keeps_full_goodbye():
    history = ConversationHistory()
    history.append_assistant(FULL_GOODBYE.strip(), turn_id=7, response_uid="r7")
    marks = {"aud7": _content_mark(FULL_GOODBYE, 7, "r7", GOODBYE_DURATION, GOODBYE_SENT_TS)}
    tm = _make_tm(history, marks)
    tm.hangup_triggered = True
    tm.conversation_ended = False

    async def _drain_soon():
        await asyncio.sleep(0.05)
        tm.mark_event_meta_data.drain()

    drainer = asyncio.create_task(_drain_soon())

    # run()'s terminal block, post-fix
    start = time.monotonic()
    if tm.hangup_triggered and not tm.conversation_ended:
        await tm.wait_for_current_message()
    elapsed = time.monotonic() - start
    await drainer

    assert 0.04 < elapsed < 3.0  # waited for the drain, not zero and not the full grace

    has_pending = len(tm.mark_event_meta_data.pending_marks) > 0
    if has_pending:
        await tm.sync_history(tm.mark_event_meta_data.pending_marks.items(), time.time())
    assert _turn7(history)["content"] == FULL_GOODBYE.strip()


async def test_wait_is_bounded_when_marks_never_ack():
    # Worst case: Plivo never acks the goodbye and conversation_ended stays False.
    # The wait must still return via the grace deadline, never loop forever.
    history = ConversationHistory()
    history.append_assistant(FULL_GOODBYE.strip(), turn_id=7, response_uid="r7")
    marks = {"aud7": _content_mark(FULL_GOODBYE, 7, "r7", 0.1, GOODBYE_SENT_TS)}
    tm = _make_tm(history, marks)
    tm.hangup_triggered = True
    tm.conversation_ended = False
    tm.hangup_mark_event_timeout = 0.2  # shrink the grace for the test

    start = time.monotonic()
    await asyncio.wait_for(tm.wait_for_current_message(), timeout=5.0)  # hard-fails if it hangs
    elapsed = time.monotonic() - start

    assert elapsed < 2.0  # bounded by duration + grace, not endless
    assert len(tm.mark_event_meta_data.pending_marks) > 0  # returned without the mark ever acking


def test_run_gates_terminal_sync_on_in_flight_hangup():
    src = inspect.getsource(TaskManager.run)
    gate_idx = src.find("if self.hangup_triggered and not self.conversation_ended:")
    sync_idx = src.find(
        "await self.sync_history(\n                        self.mark_event_meta_data.pending_marks.items(),"
    )
    assert gate_idx != -1 and sync_idx != -1
    assert gate_idx < sync_idx
