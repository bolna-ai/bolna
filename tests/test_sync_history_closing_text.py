"""Synthesizers that put a request's text on its closing packet only (Azure, OpenAI-compatible PCM streaming) leave
text-less marks for the audio before it. A barge-in measures playback over all of them, from the turn's first mark,
so text not yet played is not credited as heard."""

from bolna.agent_manager.task_manager import TaskManager
from bolna.helpers.conversation_history import ConversationHistory


def _mark(turn_id, text, duration, sent_ts, mark_type="", closing_pause=False):
    return {
        "type": mark_type,
        "turn_id": turn_id,
        "response_uid": None,
        "text_synthesized": text,
        "duration": duration,
        "sent_ts": sent_ts,
        "closing_pause": closing_pause,
    }


def _selected_before(mark_events_data, target_turn_id):
    """The selection sync_history made before closing-packet text was handled."""
    chunks = []
    for _, mark_data in mark_events_data:
        text = mark_data.get("text_synthesized", "")
        if target_turn_id is not None and mark_data.get("turn_id") != target_turn_id:
            continue
        if mark_data.get("type", "") in ["pre_mark_message", "backchanneling"] or not text:
            continue
        chunks.append({"text": text, "duration": mark_data.get("duration", 0), "sent_ts": mark_data.get("sent_ts", 0)})
    return chunks


def _closing_text_marks(closing_pause=True):
    """Three sentences, each 3 s of audio then a 0.3 s closing packet with its text, all sent within 0.1 s. Our TTS
    servers' closing packet is the pause after the speech; Azure's last chunk still carries speech."""
    marks, sent_ts = [], 100.0
    for index, sentence in enumerate(["First sentence. ", "Second sentence. ", "Third sentence. "]):
        for part in range(3):
            marks.append((f"m{index}{part}", _mark(8, "", 1.0, sent_ts)))
            sent_ts += 0.005
        marks.append((f"m{index}c", _mark(8, sentence, 0.3, sent_ts, closing_pause=closing_pause)))
        sent_ts += 0.005
    return marks


def test_marks_that_all_carry_text_are_selected_as_before():
    marks = [(f"m{i}", _mark(8, f"word{i} ", 0.4, 100.0 + i * 0.4)) for i in range(5)]
    marks.append(("p", _mark(8, "", 0.0, 99.0, mark_type="pre_mark_message")))
    marks.append(("o", _mark(7, "other turn ", 0.4, 98.0)))
    assert TaskManager._pending_text_chunks(marks, 8) == _selected_before(marks, 8)


def test_marks_with_no_text_select_nothing_as_before():
    marks = [(f"m{i}", _mark(8, "", 0.4, 100.0 + i * 0.4)) for i in range(5)]
    assert TaskManager._pending_text_chunks(marks, 8) == _selected_before(marks, 8) == []


def test_closing_text_counts_once_the_speech_before_it_played():
    chunks = TaskManager._pending_text_chunks(_closing_text_marks(), 8)
    sentence = [("", 1.0)] * 3
    expected = []
    for text in ["First sentence. ", "Second sentence. ", "Third sentence. "]:
        expected += sentence + [(text, 0), ("", 0.3)]
    assert [(chunk["text"], chunk["duration"]) for chunk in chunks] == expected
    assert chunks[0]["sent_ts"] == 100.0


def test_a_closing_chunk_that_carries_speech_keeps_its_own_time():
    chunks = TaskManager._pending_text_chunks(_closing_text_marks(closing_pause=False), 8)
    assert [(chunk["text"], chunk["duration"]) for chunk in chunks[:4]] == [("", 1.0)] * 3 + [("First sentence. ", 0.3)]


class _Input:
    last_heard_turn_id = None
    last_heard_response_uid = None
    response_heard_by_user = ""
    response_heard_by_response = {}
    response_heard_by_turn = {}

    def get_current_mark_started_time(self):
        return 0.0


class _MarkMeta:
    heard_text_by_response = {}
    heard_text_by_turn = {}


def _sync_target():
    history = ConversationHistory()
    history.append_user("hello")
    history.append_assistant("First sentence. Second sentence. Third sentence.", turn_id=8)
    tm = TaskManager.__new__(TaskManager)
    tm.conversation_history = history
    tm.tools = {"input": _Input()}
    tm.mark_event_meta_data = _MarkMeta()
    tm._turn_msg_map = {}
    tm._last_spoken_assistant = None
    tm._pending_assistant_history = {}
    return tm, history


async def test_a_barge_in_inside_the_second_sentence_credits_the_first_only():
    tm, history = _sync_target()
    # 4.0 s after the first packet went out: the first sentence (3.3 s) is over, the second is under way.
    await tm.sync_history(_closing_text_marks(), interruption_processed_at=104.0)
    assert [m["content"] for m in history.messages if m["role"] == "assistant"] == ["First sentence."]


async def test_a_barge_in_in_the_pause_after_a_sentence_credits_all_of_it():
    tm, history = _sync_target()
    # 3.2 s in: the first sentence's speech is over and its closing pause is playing.
    await tm.sync_history(_closing_text_marks(), interruption_processed_at=103.2)
    assert [m["content"] for m in history.messages if m["role"] == "assistant"] == ["First sentence."]


async def test_a_closing_chunk_that_carries_speech_is_still_credited_by_its_share():
    tm, history = _sync_target()
    await tm.sync_history(_closing_text_marks(closing_pause=False), interruption_processed_at=103.2)
    assert [m["content"] for m in history.messages if m["role"] == "assistant"] == ["First"]
