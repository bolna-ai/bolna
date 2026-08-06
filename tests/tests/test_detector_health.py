"""detector_health telemetry: emitted ONLY when the LID tap produced nothing while the
caller demonstrably spoke.

Topaz 389b16aa/df3479eb/5b063ddd: tap connected, socket alive, caller spoke several turns,
yet zero detector segments — so zero decides, so zero telemetry. The call was indistinguishable
from one where nobody switched. Presence of this record is the signal.
"""

from unittest.mock import MagicMock

from bolna.transcriber.transcriber_pool import TranscriberPool

RECORD = TranscriberPool._record_detector_health


def _pool(segments_received, user_turns, provider="sarvam", **lid_extra):
    pool = MagicMock(spec=TranscriberPool)
    pool.lid_detection_events = []
    pool._lid_provider_name = provider
    pool.active_label = "mr"
    lid = MagicMock()
    lid.segments_received = segments_received
    lid.chunks_fed = lid_extra.get("chunks_fed", 100)
    lid.chunks_dropped = lid_extra.get("chunks_dropped", 0)
    lid.unknown_frames = lid_extra.get("unknown_frames", 0)
    lid._dead = lid_extra.get("dead", False)
    lid._reconnect_attempts = lid_extra.get("reconnects", 0)
    pool._lid = lid
    transcriber = MagicMock()
    transcriber.turn_counter = user_turns
    pool.transcribers = {"mr": transcriber}
    return pool


def test_silent_detector_with_user_turns_is_recorded():
    pool = _pool(segments_received=0, user_turns=4, chunks_fed=5842, unknown_frames=3)
    RECORD(pool)
    assert len(pool.lid_detection_events) == 1
    rec = pool.lid_detection_events[0]
    assert rec["type"] == "detector_health"
    assert rec["segments_received"] == 0
    assert rec["user_turns"] == 4
    assert rec["chunks_fed"] == 5842
    assert rec["unknown_frames"] == 3
    assert rec["provider"] == "sarvam"
    assert rec["ts"] > 0


def test_healthy_detector_writes_nothing():
    pool = _pool(segments_received=9, user_turns=4)
    RECORD(pool)
    assert pool.lid_detection_events == []


def test_silent_call_is_not_reported_as_a_broken_detector():
    # Nobody spoke: zero segments is correct, not a failure.
    pool = _pool(segments_received=0, user_turns=0)
    RECORD(pool)
    assert pool.lid_detection_events == []


def test_no_lid_tap_writes_nothing():
    pool = _pool(segments_received=0, user_turns=3)
    pool._lid = None
    RECORD(pool)
    assert pool.lid_detection_events == []


def test_backend_without_counters_is_skipped():
    # Older/segment-less backends expose no counter — never guess at their health.
    pool = _pool(segments_received=0, user_turns=3)
    del pool._lid.segments_received
    RECORD(pool)
    assert pool.lid_detection_events == []


def test_record_type_does_not_collide_with_switch_records():
    # Every existing query filters flow == 'llm_switch'; this record must not carry it.
    pool = _pool(segments_received=0, user_turns=2)
    RECORD(pool)
    assert "flow" not in pool.lid_detection_events[0]
