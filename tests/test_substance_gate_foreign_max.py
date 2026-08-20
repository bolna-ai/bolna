"""The decide's substance gate measures FOREIGN segments, not the buffer-lifetime max.

The idle-flush skip deliberately does not drain, and the buffer max only resets on a drain —
so a long stale active-language segment could carry a short mis-tagged foreign fragment past
the gate that __detector_language_mismatch already measures per-foreign-segment.

Also pins the eager call site: it must pass eager_meta_info (the meta carrying the
sequence_id the eager reply's audio plays under) — the raw message meta has none, and
__arm_lid_playback_gate silently refuses to arm without one.
"""

import inspect
import re
from unittest.mock import AsyncMock, MagicMock


from bolna.agent_manager.task_manager import TaskManager
from bolna.synthesizer.synthesizer_pool import SynthesizerPool
from bolna.transcriber.transcriber_pool import TranscriberPool


def _tm(monkeypatch, segments, buffer_max):
    monkeypatch.setenv("LANGUAGE_SWITCH_SETTLE_MS", "0")
    tm = MagicMock()
    tm.task_config = {"tools_config": {"llm_agent": {"agent_type": "graph_agent"}}}
    tm.language = "hi"
    tm.conversation_ended = False
    tm.hangup_triggered = False
    tm.function_call_in_flight = False
    tm.multilingual_prompts = {"hi": "p", "mr": "p"}
    tm._should_ignore_transcriber_input = MagicMock(return_value=False)
    pool = MagicMock(spec=TranscriberPool)
    pool.labels = ["hi", "mr"]
    pool.lid_detection_events = []
    pool.lid_buffer_max_segment_seconds.return_value = buffer_max
    pool.lid_buffer_language_confidence.return_value = 0.9
    pool.lid_buffer_segments.return_value = segments
    pool.take_lid_transcript.return_value = ("kahi tari", "mr")
    synth = MagicMock(spec=SynthesizerPool)
    synth.labels = ["hi", "mr"]
    tm.tools = {"transcriber": pool, "synthesizer": synth, "input": MagicMock()}
    tm.language_switcher = MagicMock()
    tm.language_switcher.decide = AsyncMock(
        return_value={"target_language": "mr", "target_confidence": 0.95, "reasoning": "r"}
    )
    tm._inflight_response_activity = MagicMock(return_value={"audio_playing": False})
    tm._TaskManager__cleanup_downstream_tasks = AsyncMock()
    tm.switch_language = AsyncMock()
    tm._TaskManager__switch_context_note = MagicMock(return_value="note")
    tm._TaskManager__play_switch_handoff = AsyncMock()
    tm._TaskManager__prepare_followup_generation = MagicMock(return_value=None)
    tm.conversation_history = MagicMock()
    tm.conversation_history.replace_last_user.return_value = True
    for name in ("switch_audio_gap_s", "switch_settle_ms", "switch_decide_timeout_s", "record_lid_event"):
        attr = f"_TaskManager__{name}"
        setattr(tm, attr, getattr(TaskManager, attr).__get__(tm, TaskManager))
    tm._TaskManager__detector_corroborates = TaskManager._TaskManager__detector_corroborates
    return tm


async def _run(tm):
    run = TaskManager._TaskManager__run_language_switch.__get__(tm, TaskManager)
    return await run("garbled", {"sequence_id": 1}, "hi")


def _outcomes(tm):
    return [e.get("outcome") for e in tm.tools["transcriber"].lid_detection_events]


async def test_stale_active_max_cannot_carry_a_short_foreign_fragment(monkeypatch):
    # 2.5s ACTIVE-language segment sits undrained; the foreign evidence is a 0.3s fragment.
    tm = _tm(
        monkeypatch,
        segments=[
            {"lang": "hi", "prob": 0.95, "audio_s": 2.5},
            {"lang": "mr", "prob": 0.9, "audio_s": 0.3},
        ],
        buffer_max=2.5,
    )
    await _run(tm)
    tm.switch_language.assert_not_awaited()
    assert "gated:short_audio" in _outcomes(tm)


async def test_genuine_long_foreign_segment_still_passes(monkeypatch):
    tm = _tm(
        monkeypatch,
        segments=[{"lang": "mr", "prob": 0.9, "audio_s": 1.4}],
        buffer_max=1.4,
    )
    await _run(tm)
    tm.switch_language.assert_awaited_once()
    assert "switched" in _outcomes(tm)


async def test_explicit_request_still_bypasses_the_gate(monkeypatch):
    # A by-name request is legitimately short — the bypass must survive the measure change.
    tm = _tm(
        monkeypatch,
        segments=[{"lang": "mr", "prob": 0.9, "audio_s": 0.4}],
        buffer_max=0.4,
    )
    tm.language_switcher.decide = AsyncMock(
        return_value={
            "target_language": "mr",
            "target_confidence": 0.95,
            "explicit_request": True,
            "reasoning": "asked by name",
        }
    )
    await _run(tm)
    tm.switch_language.assert_awaited_once()


def test_eager_call_site_passes_eager_meta_info():
    # The raw message meta has no sequence_id, so the playback gate would never arm on
    # eager turns; the call site must hand over eager_meta_info instead.
    src = inspect.getsource(TaskManager)
    eager_block = src[src.find("using speculative LLM") :]
    call = re.search(r"self\._spawn_language_switch_decision\(([^)]*)\)", eager_block)
    assert call is not None
    assert "self.eager_meta_info" in call.group(1)


async def test_gate_armed_late_from_drained_evidence(monkeypatch):
    # Spawn-time arming can miss (idle-flush drain emptied the buffer at that instant);
    # the decide must arm from what it drained so the old-language reply can't play.
    tm = _tm(
        monkeypatch,
        segments=[{"lang": "mr", "prob": 0.9, "audio_s": 1.4}],
        buffer_max=1.4,
    )
    tm.lid_playback_gate = None
    tm._TaskManager__arm_lid_playback_gate = TaskManager._TaskManager__arm_lid_playback_gate.__get__(tm, TaskManager)
    tm.language_switcher.decide = AsyncMock(
        return_value={"target_language": None, "target_confidence": 0.0, "reasoning": "stay"}
    )
    run = TaskManager._TaskManager__run_language_switch.__get__(tm, TaskManager)
    await run("garbled", {"sequence_id": 7}, "hi")
    assert tm.lid_playback_gate is not None
    assert tm.lid_playback_gate["sequence_id"] == 7  # keyed to the reply that must wait


async def test_no_late_arm_without_substantive_foreign_evidence(monkeypatch):
    tm = _tm(
        monkeypatch,
        segments=[{"lang": "mr", "prob": 0.9, "audio_s": 0.3}],
        buffer_max=0.3,
    )
    tm.lid_playback_gate = None
    tm._TaskManager__arm_lid_playback_gate = TaskManager._TaskManager__arm_lid_playback_gate.__get__(tm, TaskManager)
    tm.language_switcher.decide = AsyncMock(
        return_value={"target_language": None, "target_confidence": 0.0, "reasoning": "stay"}
    )
    run = TaskManager._TaskManager__run_language_switch.__get__(tm, TaskManager)
    await run("garbled", {"sequence_id": 7}, "hi")
    assert tm.lid_playback_gate is None


async def test_live_gate_is_not_clobbered_by_late_arm(monkeypatch):
    # A gate whose decide is still running (the spawn-time arm for this very turn) must win.
    tm = _tm(
        monkeypatch,
        segments=[{"lang": "mr", "prob": 0.9, "audio_s": 1.4}],
        buffer_max=1.4,
    )
    live_task = MagicMock()
    live_task.done.return_value = False
    sentinel = {"sequence_id": 3, "task": live_task, "armed_at": 0.0, "language": "hi", "deadline": 1e18}
    tm.lid_playback_gate = sentinel
    tm._TaskManager__arm_lid_playback_gate = TaskManager._TaskManager__arm_lid_playback_gate.__get__(tm, TaskManager)
    tm.language_switcher.decide = AsyncMock(
        return_value={"target_language": None, "target_confidence": 0.0, "reasoning": "stay"}
    )
    run = TaskManager._TaskManager__run_language_switch.__get__(tm, TaskManager)
    await run("garbled", {"sequence_id": 7}, "hi")
    assert tm.lid_playback_gate is sentinel  # spawn-time gate wins


async def test_stale_done_gate_is_retired_and_rearmed(monkeypatch):
    # A finished decide's gate that no chunk ever polled (its audio had already played) must
    # not block late arming forever — only chunk polls release gates otherwise.
    tm = _tm(
        monkeypatch,
        segments=[{"lang": "mr", "prob": 0.9, "audio_s": 1.4}],
        buffer_max=1.4,
    )
    done_task = MagicMock()
    done_task.done.return_value = True
    tm.lid_playback_gate = {"sequence_id": 3, "task": done_task, "armed_at": 0.0, "language": "hi", "deadline": 1e18}
    for name in ("arm_lid_playback_gate", "release_lid_playback_gate"):
        attr = f"_TaskManager__{name}"
        setattr(tm, attr, getattr(TaskManager, attr).__get__(tm, TaskManager))
    tm.language_switcher.decide = AsyncMock(
        return_value={"target_language": None, "target_confidence": 0.0, "reasoning": "stay"}
    )
    run = TaskManager._TaskManager__run_language_switch.__get__(tm, TaskManager)
    await run("garbled", {"sequence_id": 7}, "hi")
    assert tm.lid_playback_gate is not None
    assert tm.lid_playback_gate["sequence_id"] == 7  # stale gate retired, new one armed
    outcomes = [
        e.get("outcome") for e in tm.tools["transcriber"].lid_detection_events if e.get("type") == "playback_gate"
    ]
    assert "decided" in outcomes  # the stale gate was released with telemetry, not dropped
