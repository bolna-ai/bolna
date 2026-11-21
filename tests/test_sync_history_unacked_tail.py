"""A chunk whose mark hadn't come back yet was trimmed from history although most of it played:
at hangup (final chunk) and at barge-in (the chunk playing when the caller cut in)."""

import time

from bolna.agent_manager.interruption_manager import InterruptionManager
from bolna.agent_manager.task_manager import TaskManager
from bolna.helpers.conversation_history import ConversationHistory
from bolna.helpers.mark_event_meta_data import MarkEventMetaData

FULL_TEXT = (
    "aapka final price rupees two thousand five hundred fifty seven padega abhi ye deal limited time ke liye hai"
)
PREFIX = "aapka final price "
TAIL = "rupees two thousand five hundred fifty seven padega abhi ye deal limited time ke liye hai"


class InputStub:
    last_heard_turn_id = 8
    last_heard_response_uid = "r8"
    response_heard_by_user = ""

    def get_response_heard_for_response(self, uid):
        return ""

    def get_response_heard_for_turn(self, turn_id):
        return ""

    def get_current_mark_started_time(self):
        return 0.0


def chunk(text, duration, sent_ts, turn_id=8):
    return {
        "type": "",
        "turn_id": turn_id,
        "response_uid": "r8",
        "sequence_id": 8,
        "duration": duration,
        "text_synthesized": text,
        "sent_ts": sent_ts,
    }


def make_tm(assistant_text=FULL_TEXT):
    tm = TaskManager.__new__(TaskManager)
    tm.interruption_manager = InterruptionManager()
    history = ConversationHistory()
    history.append_user("haan main purchase karne mein interested hoon")
    history.append_assistant(assistant_text, turn_id=8, response_uid="r8")
    tm.conversation_history = history
    tm.tools = {"input": InputStub()}
    tm._turn_msg_map = {}
    tm._pending_assistant_history = {}
    tm._last_spoken_assistant = None
    tm.mark_event_meta_data = MarkEventMetaData()
    return tm, history, tm.mark_event_meta_data


def ack(marks, mark_id, data):
    marks.update_data(mark_id, data)
    acked = marks.fetch_data(mark_id)
    marks.record_heard_text(acked, data["text_synthesized"])


def make_acked_prefix_tm(tail_sent_ts=None):
    """ACKed prefix chunk, then a 9.6s tail chunk whose mark never came back."""
    tm, history, marks = make_tm()
    now = time.time()
    ack(marks, "m1", chunk(PREFIX, 2.0, now - 30))
    marks.update_data("m2", chunk(TAIL, 9.6, now - 28 if tail_sent_ts is None else tail_sent_ts))
    return tm, history, marks


def assistant_content(history):
    return next(m["content"] for m in reversed(history.messages) if m["role"] == "assistant")


async def test_end_of_call_credits_unacked_tail_played_after_last_ack():
    tm, history, marks = make_acked_prefix_tm()
    teardown_ts = marks.get_last_ack_ts_for_turn(8) + 20  # well past the 9.6s tail
    await tm.sync_history(marks.mark_event_meta_data.items(), teardown_ts, extend_with_playback_estimate=True)
    assert assistant_content(history) == FULL_TEXT


async def test_end_of_call_mid_chunk_hangup_credits_proportional_word_trimmed():
    # PR review replay: last ACK 6.97s before teardown vs 9.61s tail → ~72% heard, word-aligned.
    tm, history, marks = make_acked_prefix_tm()
    teardown_ts = marks.get_last_ack_ts_for_turn(8) + 6.97
    await tm.sync_history(marks.mark_event_meta_data.items(), teardown_ts, extend_with_playback_estimate=True)
    content = assistant_content(history)
    assert content.startswith(PREFIX + "rupees")  # tail partially credited
    assert len(content) < len(FULL_TEXT)
    assert FULL_TEXT.startswith(content)  # word-aligned prefix of the real text


async def test_end_of_call_zero_elapsed_keeps_strict_ack_trim():
    tm, history, marks = make_acked_prefix_tm()
    teardown_ts = marks.get_last_ack_ts_for_turn(8)  # hangup at the last confirmed instant
    await tm.sync_history(marks.mark_event_meta_data.items(), teardown_ts, extend_with_playback_estimate=True)
    assert assistant_content(history) == PREFIX.strip()


async def test_without_playback_estimate_trims_to_acked_text():
    tm, history, marks = make_acked_prefix_tm()
    await tm.sync_history(marks.mark_event_meta_data.items(), time.time())
    assert assistant_content(history) == PREFIX.strip()


async def test_barge_in_credits_the_chunk_playing_up_to_the_callers_onset():
    # Replays a prod barge-in: the playing 9.6s chunk was cut 9s in, the caller started 7.5s in.
    tm, history, marks = make_acked_prefix_tm()
    last_ack = marks.get_last_ack_ts_for_turn(8)
    await tm.sync_history(
        marks.mark_event_meta_data.items(),
        last_ack + 9.0,
        extend_with_playback_estimate=True,
        caller_onset=last_ack + 7.5,
    )
    assert assistant_content(history) == (
        "aapka final price rupees two thousand five hundred fifty seven padega abhi ye deal"
    )


async def test_barge_in_onset_before_the_last_ack_credits_nothing_after_it():
    # The caller started during the ACKed chunk; everything after its ACK was talked over.
    tm, history, marks = make_acked_prefix_tm()
    last_ack = marks.get_last_ack_ts_for_turn(8)
    await tm.sync_history(
        marks.mark_event_meta_data.items(),
        last_ack + 20,
        extend_with_playback_estimate=True,
        caller_onset=last_ack - 0.5,
    )
    assert assistant_content(history) == "aapka final price"


async def test_tail_window_starts_when_the_tail_was_sent_after_the_last_ack():
    # The tail was still in LLM/TTS 5s after the prefix ACKed; that gap isn't playback.
    tm, history, marks = make_tm()
    now = time.time()
    ack(marks, "m1", chunk(PREFIX, 2.0, now - 30))
    last_ack = marks.get_last_ack_ts_for_turn(8)
    marks.update_data("m2", chunk(TAIL, 9.6, last_ack + 5))
    await tm.sync_history(marks.mark_event_meta_data.items(), last_ack + 8, extend_with_playback_estimate=True)
    assert assistant_content(history) == "aapka final price rupees two thousand five"


async def test_lost_mark_from_before_the_last_ack_is_not_credited_as_tail():
    tm, history, marks = make_tm()
    now = time.time()
    marks.update_data("m0", chunk("lost words ", 1.0, now - 31))  # its mark never came back
    ack(marks, "m1", chunk(PREFIX, 2.0, now - 30))
    marks.update_data("m2", chunk(TAIL, 9.6, now - 28))
    await tm.sync_history(
        marks.mark_event_meta_data.items(),
        marks.get_last_ack_ts_for_turn(8) + 20,
        extend_with_playback_estimate=True,
    )
    assert assistant_content(history) == FULL_TEXT


async def test_text_less_chunks_count_toward_playback_time():
    # Azure: a sentence's audio chunks carry no text, only its last chunk does.
    tm, history, marks = make_tm()
    now = time.time()
    ack(marks, "m1", chunk(PREFIX, 2.0, now - 30))
    marks.update_data("m2", chunk("", 4.0, now - 29))
    marks.update_data("m3", chunk(TAIL, 0.2, now - 28))
    last_ack = marks.get_last_ack_ts_for_turn(8)
    await tm.sync_history(
        marks.mark_event_meta_data.items(),
        last_ack + 3,
        extend_with_playback_estimate=True,
        caller_onset=last_ack + 2,
    )
    assert assistant_content(history) == "aapka final price"


async def test_tail_joins_a_chunk_boundary_inside_a_word():
    tm, history, marks = make_tm()
    now = time.time()
    ack(marks, "m1", chunk("aapka final pri", 2.0, now - 30))
    marks.update_data("m2", chunk("ce " + TAIL, 9.6, now - 28))
    await tm.sync_history(
        marks.mark_event_meta_data.items(),
        marks.get_last_ack_ts_for_turn(8) + 20,
        extend_with_playback_estimate=True,
    )
    assert assistant_content(history) == FULL_TEXT


async def test_first_send_fallback_stops_at_the_callers_onset():
    # No chunk ACKed yet: credit runs from the first send, but only up to the onset.
    tm, history, marks = make_tm(assistant_text=TAIL)
    now = time.time()
    marks.update_data("m1", chunk(TAIL, 9.6, now - 9))
    await tm.sync_history(marks.mark_event_meta_data.items(), now, caller_onset=now - 6)
    assert assistant_content(history) == "rupees two thousand five"


def test_cleared_marks_are_handed_out_once():
    marks = MarkEventMetaData()
    marks.update_data("m1", chunk(TAIL, 9.6, time.time()))
    marks.clear_data()
    assert list(marks.fetch_cleared_mark_event_data()) == ["m1"]
    assert marks.fetch_cleared_mark_event_data() == {}


def test_barge_in_onset_prefers_the_earliest_current_signal():
    im = InterruptionManager()
    im.on_user_speech_ended()
    im.last_speech_ended_ts -= 5  # the caller's previous utterance ended 5s ago
    im.note_ignored_interim()
    ignored_at = im.first_ignored_interim_ts
    assert im.get_barge_in_onset() == ignored_at
    assert im.get_barge_in_onset(ignored_at - 0.3) == ignored_at - 0.3


def test_barge_in_onset_drops_signals_from_an_earlier_utterance():
    im = InterruptionManager()
    im.note_ignored_interim()
    stale_provider_onset = time.time()
    im.on_user_speech_ended()
    assert im.get_barge_in_onset(stale_provider_onset) is None
    im.note_ignored_interim()  # a new utterance's first ignored interim replaces the stale one
    assert im.first_ignored_interim_ts > im.last_speech_ended_ts
