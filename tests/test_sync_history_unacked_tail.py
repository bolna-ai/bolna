"""Audio whose mark had not come back yet still played: sync_history credits the share played
by the barge-in or hangup instead of trimming the turn to its ACKed text."""

import time


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


def _make_tm(prefix=PREFIX, tail=TAIL):
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
            "text_synthesized": prefix,
            "sent_ts": time.time() - 30,
        },
    )
    acked = marks.fetch_data("m1")
    marks.record_heard_text(acked, prefix)
    # Tail chunk (9.6s audio): sent, mark never arrived before teardown.
    marks.update_data(
        "m2",
        {
            "type": "",
            "turn_id": 8,
            "response_uid": "r8",
            "sequence_id": 8,
            "duration": 9.6,
            "text_synthesized": tail,
            "sent_ts": time.time() - 28,
        },
    )
    tm.mark_event_meta_data = marks
    return tm, history, marks


def _assistant_content(history):
    return next(m["content"] for m in reversed(history.messages) if m["role"] == "assistant")


async def test_credits_unacked_tail_that_finished_playing():
    tm, history, marks = _make_tm()
    teardown_ts = marks.get_last_ack_ts_for_turn(8) + 20  # well past the 9.6s tail
    await tm.sync_history(marks.mark_event_meta_data.items(), teardown_ts)
    assert _assistant_content(history) == FULL_TEXT


async def test_credits_word_trimmed_share_of_tail_cut_mid_chunk():
    # PR review replay: last ACK 6.97s before teardown vs 9.61s tail → ~72% heard, word-aligned.
    tm, history, marks = _make_tm()
    teardown_ts = marks.get_last_ack_ts_for_turn(8) + 6.97
    await tm.sync_history(marks.mark_event_meta_data.items(), teardown_ts)
    content = _assistant_content(history)
    assert content.startswith(PREFIX + "rupees")  # tail partially credited
    assert len(content) < len(FULL_TEXT)
    assert FULL_TEXT.startswith(content)  # word-aligned prefix of the real text


async def test_no_playback_since_last_ack_keeps_acked_text():
    tm, history, marks = _make_tm()
    teardown_ts = marks.get_last_ack_ts_for_turn(8)  # hangup at the last confirmed instant
    await tm.sync_history(marks.mark_event_meta_data.items(), teardown_ts)
    assert _assistant_content(history) == PREFIX.strip()


async def test_barge_in_credits_played_share_of_unacked_chunk():
    tm, history, marks = _make_tm()
    clear_ts = marks.get_last_ack_ts_for_turn(8) + 6.97
    marks.clear_data()  # what the output handler does on barge-in before the sync
    await tm.sync_history(marks.fetch_cleared_mark_event_data().items(), clear_ts)
    assert _assistant_content(history) == PREFIX + "rupees two thousand five hundred fifty seven padega abhi ye"


async def test_chunk_sent_after_last_ack_counts_from_its_send():
    tm, history, marks = _make_tm()
    last_ack_ts = marks.get_last_ack_ts_for_turn(8)
    marks.mark_event_meta_data["m2"]["sent_ts"] = last_ack_ts + 3  # TTS gap: prefix done, tail not yet sent
    await tm.sync_history(marks.mark_event_meta_data.items(), last_ack_ts + 4)
    assert _assistant_content(history) == PREFIX + "rupees"  # 1s of the 9.6s tail, not 4s


async def test_acked_text_without_ack_time_gets_no_tail_credit():
    tm, history, marks = _make_tm()
    tm.mark_event_meta_data = MarkEventMetaData()  # ACK times unknown, heard text still reported
    tm.mark_event_meta_data.update_data("m2", dict(marks.mark_event_meta_data["m2"]))
    tm.tools["input"].get_response_heard_for_response = lambda uid: PREFIX
    await tm.sync_history(tm.mark_event_meta_data.mark_event_meta_data.items(), time.time() + 20)
    assert _assistant_content(history) == PREFIX.strip()


async def test_tail_joins_without_space_when_chunk_boundary_splits_a_word():
    tm, history, marks = _make_tm(prefix="aapka final pr", tail="ice " + TAIL)
    await tm.sync_history(marks.mark_event_meta_data.items(), marks.get_last_ack_ts_for_turn(8) + 2)
    assert _assistant_content(history) == "aapka final price rupees two"
