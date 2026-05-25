"""Tests for the post-call observability primitives on MarkEventMetaData.

Covers get_chunk_marks(), ack persistence on fetch_data, cleared_on_interrupt
flagging, ordering by sent_ts, and pre_mark_message exclusion.
"""

import time

import pytest

from bolna.helpers.mark_event_meta_data import MarkEventMetaData


def _audio_chunk(seq_id: int, text: str, sent_ts: float, duration: float = 0.5) -> dict:
    return {
        "type": "",
        "text_synthesized": text,
        "is_first_chunk": False,
        "is_final_chunk": False,
        "sequence_id": seq_id,
        "duration": duration,
        "sent_ts": sent_ts,
    }


class TestGetChunkMarks:
    def test_excludes_pre_mark_messages(self):
        m = MarkEventMetaData()
        m.update_data("pre-1", {"type": "pre_mark_message"})
        m.update_data("audio-1", _audio_chunk(1, "hello", 100.0))

        marks = m.get_chunk_marks()
        assert len(marks) == 1
        assert marks[0]["mark_id"] == "audio-1"
        assert marks[0]["text_synthesized"] == "hello"

    def test_marks_sorted_by_sent_ts(self):
        m = MarkEventMetaData()
        m.update_data("c", _audio_chunk(1, "third", 300.0))
        m.update_data("a", _audio_chunk(1, "first", 100.0))
        m.update_data("b", _audio_chunk(1, "second", 200.0))

        marks = m.get_chunk_marks()
        assert [mark["text_synthesized"] for mark in marks] == ["first", "second", "third"]

    def test_initial_state_unacked(self):
        m = MarkEventMetaData()
        m.update_data("audio-1", _audio_chunk(1, "hi", 100.0))

        marks = m.get_chunk_marks()
        assert marks[0]["acked"] is False
        assert marks[0]["ack_ts"] is None

    def test_fetch_data_marks_as_acked_and_persists_in_history(self):
        m = MarkEventMetaData()
        m.update_data("audio-1", _audio_chunk(1, "hi", 100.0))

        before = time.time()
        popped = m.fetch_data("audio-1")
        after = time.time()

        assert popped["acked"] is True
        assert before <= popped["ack_ts"] <= after
        assert "audio-1" not in m.mark_event_meta_data

        marks = m.get_chunk_marks()
        assert len(marks) == 1
        assert marks[0]["acked"] is True
        assert marks[0]["ack_ts"] == popped["ack_ts"]

    def test_clear_data_flags_unacked_marks(self):
        m = MarkEventMetaData()
        m.update_data("acked-mark", _audio_chunk(1, "got it", 100.0))
        m.update_data("dropped-mark", _audio_chunk(1, "interrupted tail", 150.0))

        m.fetch_data("acked-mark")
        m.clear_data()

        by_id = {mark["mark_id"]: mark for mark in m.get_chunk_marks()}
        assert by_id["acked-mark"]["acked"] is True
        assert by_id["acked-mark"]["cleared_on_interrupt"] is False
        assert by_id["dropped-mark"]["acked"] is False
        assert by_id["dropped-mark"]["cleared_on_interrupt"] is True

    def test_history_survives_repeated_clears(self):
        m = MarkEventMetaData()
        m.update_data("turn1-chunk", _audio_chunk(1, "first turn", 100.0))
        m.fetch_data("turn1-chunk")
        m.clear_data()

        m.update_data("turn2-chunk", _audio_chunk(2, "second turn", 200.0))
        m.fetch_data("turn2-chunk")

        marks = m.get_chunk_marks()
        assert len(marks) == 2
        assert {mark["sequence_id"] for mark in marks} == {1, 2}

    def test_multi_chunk_sequence_kept_in_send_order(self):
        m = MarkEventMetaData()
        for i, ts in enumerate([100.0, 100.5, 101.0]):
            m.update_data(f"chunk-{i}", _audio_chunk(7, f"part-{i}", ts))

        marks = m.get_chunk_marks()
        assert [mark["text_synthesized"] for mark in marks] == ["part-0", "part-1", "part-2"]
        assert all(mark["sequence_id"] == 7 for mark in marks)
