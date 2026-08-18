"""The final chunk's mark never arrived before hangup, so end-of-call sync trimmed the last
agent message even though the tail audio played."""

import time

import pytest

from bolna.agent_manager.task_manager import TaskManager
from bolna.helpers.conversation_history import ConversationHistory
from bolna.helpers.mark_event_meta_data import MarkEventMetaData

FULL_TEXT = (
    "aapka final price rupees two thousand five hundred fifty seven padega abhi ye deal limited time ke liye hai"
)
PREFIX = "aapka final price "
TAIL = "rupees two thousand five hundred fifty seven padega abhi ye deal limited time ke liye hai"


class _InputStub:
    last_heard_turn_id = 8
    last_heard_response_uid = "r8"
    response_heard_by_user = ""

    def get_response_heard_for_response(self, uid):
        return ""

    def get_response_heard_for_turn(self, turn_id):
        return ""

    def get_current_mark_started_time(self):
        return 0.0


def _make_tm():
    tm = TaskManager.__new__(TaskManager)
    history = ConversationHistory()
    history.append_user("haan main purchase karne mein interested hoon")
    history.append_assistant(FULL_TEXT, turn_id=8, response_uid="r8")
    tm.conversation_history = history
    tm.tools = {"input": _InputStub()}
    tm._turn_msg_map = {}

    marks = MarkEventMetaData()
    # ACKed prefix chunk: sent, then its mark came back.
    marks.update_data(
        "m1",
        {
            "type": "",
            "turn_id": 8,
            "response_uid": "r8",
            "sequence_id": 8,
            "duration": 2.0,
            "text_synthesized": PREFIX,
            "sent_ts": time.time() - 30,
        },
    )
    acked = marks.fetch_data("m1")
    marks.record_heard_text(acked, PREFIX)
    # Tail chunk (9.6s audio): sent, mark never arrived before teardown.
    marks.update_data(
        "m2",
        {
            "type": "",
            "turn_id": 8,
            "response_uid": "r8",
            "sequence_id": 8,
            "duration": 9.6,
            "text_synthesized": TAIL,
            "sent_ts": time.time() - 28,
        },
    )
    tm.mark_event_meta_data = marks
    return tm, history, marks


def _assistant_content(history):
    return next(m["content"] for m in reversed(history.messages) if m["role"] == "assistant")


@pytest.mark.asyncio
async def test_end_of_call_credits_unacked_tail_played_after_last_ack():
    tm, history, marks = _make_tm()
    teardown_ts = marks.get_last_ack_ts_for_turn(8) + 20  # well past the 9.6s tail
    await tm.sync_history(marks.mark_event_meta_data.items(), teardown_ts, extend_with_playback_estimate=True)
    assert _assistant_content(history) == FULL_TEXT


@pytest.mark.asyncio
async def test_end_of_call_mid_chunk_hangup_credits_proportional_word_trimmed():
    # PR review replay: last ACK 6.97s before teardown vs 9.61s tail → ~72% heard, word-aligned.
    tm, history, marks = _make_tm()
    teardown_ts = marks.get_last_ack_ts_for_turn(8) + 6.97
    await tm.sync_history(marks.mark_event_meta_data.items(), teardown_ts, extend_with_playback_estimate=True)
    content = _assistant_content(history)
    assert content.startswith(PREFIX + "rupees")  # tail partially credited
    assert len(content) < len(FULL_TEXT)
    assert FULL_TEXT.startswith(content)  # word-aligned prefix of the real text


@pytest.mark.asyncio
async def test_end_of_call_zero_elapsed_keeps_strict_ack_trim():
    tm, history, marks = _make_tm()
    teardown_ts = marks.get_last_ack_ts_for_turn(8)  # hangup at the last confirmed instant
    await tm.sync_history(marks.mark_event_meta_data.items(), teardown_ts, extend_with_playback_estimate=True)
    assert _assistant_content(history) == PREFIX.strip()


@pytest.mark.asyncio
async def test_interruption_path_still_trims_to_acked_text():
    tm, history, marks = _make_tm()
    await tm.sync_history(marks.mark_event_meta_data.items(), time.time())
    assert _assistant_content(history) == PREFIX.strip()
