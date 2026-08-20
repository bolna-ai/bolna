"""Recent-turns evidence quality + playback-gate lifecycle + detector_health persistence.

- A turn whose segments never carried the judge's detected language must not borrow another
  language's duration into RECENT TURNS (fake `en(2.5)` built from hi-tagged audio).
- Switch firings are marked `→xx` so pre-switch drift reads as stale.
- The switch path records gate telemetry immediately but keeps HOLDING until the sequence is
  invalidated — clearing early let a 50ms output-loop poll ship the held old-language audio.
- detector_health must be flushed into the task_output snapshot (pool cleanup runs only at the
  tasks_to_cancel gather, after the snapshot — a record written there never persisted).
- The undrained detector buffer is age-bounded so a skipped generation cannot lend its
  duration or text to a later turn.
"""

import time
from unittest.mock import AsyncMock, MagicMock


from bolna.agent_manager.task_manager import TaskManager
from bolna.helpers.language_switcher import LanguageSwitcher
from bolna.lid.base import LIDBackend
from bolna.transcriber.transcriber_pool import TranscriberPool

RECENT = TaskManager._TaskManager__recent_detected_turns


def test_no_matching_segment_contributes_zero_duration():
    pool = MagicMock(spec=TranscriberPool)
    pool.lid_detection_events = [
        {
            "flow": "llm_switch",
            "detected_language": "en",  # judge said en...
            "buffered_max_segment_s": 2.5,
            "detector_segments": [{"lang": "hi", "audio_s": 2.5}],  # ...but only hi audio exists
        }
    ]
    assert RECENT(pool) == [("en", 0.0, None)]


def test_matching_segment_duration_still_used():
    pool = MagicMock(spec=TranscriberPool)
    pool.lid_detection_events = [
        {
            "flow": "llm_switch",
            "detected_language": "hi",
            "detector_segments": [{"lang": "hi", "audio_s": 1.4}, {"lang": "en", "audio_s": 2.0}],
        }
    ]
    assert RECENT(pool) == [("hi", 1.4, None)]


def test_switch_marker_travels_and_renders():
    pool = MagicMock(spec=TranscriberPool)
    pool.lid_detection_events = [
        {"flow": "llm_switch", "detected_language": "hi", "detector_segments": [{"lang": "hi", "audio_s": 2.2}]},
        {
            "flow": "llm_switch",
            "detected_language": "en",
            "switched_to": "en",
            "detector_segments": [{"lang": "en", "audio_s": 1.0}],
        },
    ]
    turns = RECENT(pool)
    assert turns == [("hi", 2.2, None), ("en", 1.0, "en")]
    rendered = LanguageSwitcher._format_recent_turns(turns)
    assert rendered == "hi(2.2), en(1.0)→en"
    # Old 2-tuple shape (tests, replay harness) still renders.
    assert LanguageSwitcher._format_recent_turns([("hi", 1.4)]) == "hi(1.4)"


async def test_switch_path_holds_gate_until_cleanup(language_switch_tm):
    class GateWatcher:
        def __init__(self, tm):
            self.tm = tm
            self.gate_at_cleanup = "unset"

        async def cleanup(self):
            self.gate_at_cleanup = self.tm.lid_playback_gate

    tm = language_switch_tm(audio_playing=True)
    live_task = MagicMock()
    live_task.done.return_value = False  # a live decide — must not be retired as stale
    gate = {"sequence_id": 1, "task": live_task, "armed_at": time.monotonic(), "language": "hi", "deadline": 1e18}
    tm.lid_playback_gate = gate
    tm._TaskManager__release_lid_playback_gate = TaskManager._TaskManager__release_lid_playback_gate.__get__(
        tm, TaskManager
    )
    tm._TaskManager__record_lid_event = MagicMock()
    watcher = GateWatcher(tm)
    tm._TaskManager__cleanup_downstream_tasks = AsyncMock(side_effect=watcher.cleanup)
    run = TaskManager._TaskManager__run_language_switch.__get__(tm, TaskManager)
    await run("garbled hi", {"sequence_id": 1}, "hi")
    # HELD through cleanup (the race window), cleared after.
    assert watcher.gate_at_cleanup is gate
    assert tm.lid_playback_gate is None
    # Telemetry recorded exactly once despite the split release.
    gate_records = [c for c in tm._TaskManager__record_lid_event.call_args_list if "playback_gate" in str(c)]
    assert len(gate_records) == 1


def test_snapshot_flushes_detector_health():
    tm = MagicMock()
    pool = MagicMock(spec=TranscriberPool)
    pool.lid_detection_events = []

    def record():
        pool.lid_detection_events.append({"type": "detector_health"})

    pool._record_detector_health = record
    tm.tools = {"transcriber": pool}
    snap = TaskManager._TaskManager__snapshot_lid_events(tm)
    assert {"type": "detector_health"} in snap


def test_detector_health_records_only_once():
    pool = MagicMock(spec=TranscriberPool)
    pool.lid_detection_events = []
    pool._detector_health_recorded = False
    pool._lid_provider_name = "sarvam"
    pool.active_label = "mr"
    lid = MagicMock()
    lid.segments_received = 0
    lid.chunks_fed = 10
    lid.chunks_dropped = 0
    lid.unknown_frames = 0
    lid._dead = False
    lid._reconnect_attempts = 0
    pool._lid = lid
    transcriber = MagicMock()
    transcriber.turn_counter = 3
    pool.transcribers = {"mr": transcriber}
    record = TranscriberPool._record_detector_health
    record(pool)
    record(pool)  # cleanup-time second call must be a no-op
    assert len(pool.lid_detection_events) == 1


def test_buffer_evicts_segments_past_age_bound():
    lid = LIDBackend(on_language=None, config={})
    lid._accumulate("purana", "hi", 2.5)
    # Age the first segment past the bound.
    lid._buffer_segments[0]["ts"] = time.time() - 31.0
    lid._accumulate("naya", "mr", 0.6)
    assert [s["text"] for s in lid._buffer_segments] == ["naya"]
    assert lid.buffer_max_segment_seconds() == 0.6  # old 2.5s no longer lends its duration
    text, lang = lid.take_turn_transcript()
    assert text == "naya"
