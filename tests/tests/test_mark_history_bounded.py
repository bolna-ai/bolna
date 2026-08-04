"""Tests for the bounds on MarkEventMetaData's per-call accumulators.

Covers the mark-history record cap and its eviction reporting, the calibration anchor
surviving eviction, end-of-call buffer release, and raw chunk-mark sampling.
"""

import pytest

from bolna.helpers import mark_event_meta_data as med
from bolna.helpers.mark_event_meta_data import MarkEventMetaData, should_persist_chunk_marks


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


def _fill(mark: MarkEventMetaData, count: int, start_ts: float = 100.0):
    for i in range(count):
        mark.update_data(f"chunk-{i}", _audio_chunk(1, f"part-{i}", start_ts + i))


class TestHistoryCap:
    def test_history_stops_growing_at_cap(self):
        m = MarkEventMetaData(max_history=5)
        _fill(m, 50)

        assert len(m.get_chunk_marks()) == 5

    def test_cap_keeps_the_most_recent_chunks(self):
        m = MarkEventMetaData(max_history=3)
        _fill(m, 6)

        assert [mark["text_synthesized"] for mark in m.get_chunk_marks()] == ["part-3", "part-4", "part-5"]

    def test_summary_reports_retained_and_dropped(self):
        m = MarkEventMetaData(max_history=4)
        _fill(m, 10)

        summary = m.get_mark_tracking_summary()
        assert summary["history_retained"] == 4
        assert summary["history_dropped"] == 6
        # The aggregate still counts every chunk that was sent, capped history or not.
        assert summary["total_sent"] == 10

    def test_nothing_dropped_below_the_cap(self):
        m = MarkEventMetaData(max_history=100)
        _fill(m, 10)

        summary = m.get_mark_tracking_summary()
        assert summary["history_dropped"] == 0
        assert summary["history_retained"] == 10

    def test_first_sent_ts_survives_eviction(self):
        m = MarkEventMetaData(max_history=2)
        _fill(m, 20, start_ts=500.0)

        summary = m.get_mark_tracking_summary()
        assert summary["first_mark_sent_ts"] == 500.0
        assert all(mark["sent_ts"] > 500.0 for mark in m.get_chunk_marks())

    def test_pre_mark_messages_do_not_consume_the_cap(self):
        m = MarkEventMetaData(max_history=2)
        for i in range(10):
            m.update_data(f"pre-{i}", {"type": "pre_mark_message"})
        _fill(m, 2)

        assert len(m.get_chunk_marks()) == 2
        assert m.get_mark_tracking_summary()["history_dropped"] == 0

    def test_retained_chunks_still_pick_up_acks_and_interrupt_flags(self):
        m = MarkEventMetaData(max_history=2)
        _fill(m, 3)  # chunk-0 evicted, chunk-1 and chunk-2 retained

        m.fetch_data("chunk-1")
        m.clear_data()

        by_id = {mark["mark_id"]: mark for mark in m.get_chunk_marks()}
        assert by_id["chunk-1"]["acked"] is True
        assert by_id["chunk-2"]["acked"] is False
        assert by_id["chunk-2"]["cleared_on_interrupt"] is True

    def test_default_cap_comes_from_the_module_constant(self):
        assert MarkEventMetaData()._max_history == med.MAX_MARK_HISTORY


class TestReleaseCallBuffers:
    def test_release_frees_history_and_heard_text(self):
        m = MarkEventMetaData()
        _fill(m, 5)
        m.record_heard_text({"turn_id": 1, "response_uid": "r1"}, "hello there")
        m.clear_data()

        m.release_call_buffers()

        assert m.get_chunk_marks() == []
        assert m.get_heard_text_for_turn(1) == ""
        assert m.get_heard_text_for_response("r1") == ""
        assert m.fetch_cleared_mark_event_data() == {}

    def test_release_keeps_the_aggregate_intact(self):
        m = MarkEventMetaData()
        _fill(m, 5)
        summary_before = m.get_mark_tracking_summary()

        m.release_call_buffers()
        summary_after = m.get_mark_tracking_summary()

        assert summary_after["total_sent"] == summary_before["total_sent"] == 5
        assert summary_after["first_mark_sent_ts"] == summary_before["first_mark_sent_ts"]


class TestShouldPersistChunkMarks:
    def test_disabled_by_default(self):
        assert med.PERSIST_CHUNK_MARKS_PCT == 0
        assert should_persist_chunk_marks("any-run-id") is False

    def test_full_rollout_persists_every_call(self, monkeypatch):
        monkeypatch.setattr(med, "PERSIST_CHUNK_MARKS_PCT", 100)
        assert should_persist_chunk_marks("any-run-id") is True

    def test_decision_is_stable_for_a_run_id(self, monkeypatch):
        monkeypatch.setattr(med, "PERSIST_CHUNK_MARKS_PCT", 50)
        run_id = "766d0374-2355-447e-95dd-0ff872e815cc"
        assert should_persist_chunk_marks(run_id) == should_persist_chunk_marks(run_id)

    def test_sample_size_tracks_the_percentage(self, monkeypatch):
        monkeypatch.setattr(med, "PERSIST_CHUNK_MARKS_PCT", 10)
        run_ids = [f"run-{i}" for i in range(2000)]
        sampled = sum(1 for r in run_ids if should_persist_chunk_marks(r))
        assert 0.07 < sampled / len(run_ids) < 0.13

    @pytest.mark.parametrize("run_id", [None, ""])
    def test_missing_run_id_is_not_sampled(self, run_id, monkeypatch):
        monkeypatch.setattr(med, "PERSIST_CHUNK_MARKS_PCT", 50)
        assert should_persist_chunk_marks(run_id) is False
