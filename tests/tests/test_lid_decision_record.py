"""build_lid_decision_record — the per-firing LID telemetry record (persisted to lid_detection_events)."""

from bolna.agent_manager.task_manager import build_lid_decision_record

DECISION = {
    "target_language": "en",
    "target_confidence": 0.92,
    "explicit_request": False,
    "reasoning": "Clear English matrix.",
}


def _record(**overrides):
    base = dict(
        outcome="switched",
        fired_at=1000.0,
        now=1001.5,
        speculate=False,
        active_transcript="garbled hi text",
        active="hi",
        detector_transcript="how much money can I make",
        detector_lang_tag="en",
        decision=DECISION,
        buffered_max_segment_s=2.336,
        speculation_started=False,
    )
    base.update(overrides)
    return build_lid_decision_record(**base)


def test_captures_both_transcripts_and_decision():
    r = _record()
    # Both ASR transcripts must be present — the whole point for accuracy analysis.
    assert r["detector_transcript"] == "how much money can I make"  # unbiased / LID ASR
    assert r["active_transcript"] == "garbled hi text"  # locked main ASR
    assert r["detector_lang_tag"] == "en"
    assert r["active_language"] == "hi"
    assert r["target_language"] == "en"
    assert r["target_confidence"] == 0.92
    assert r["explicit_request"] is False
    assert r["reasoning"] == "Clear English matrix."
    assert r["flow"] == "llm_switch"  # discriminator vs legacy heuristic records


def test_latency_and_fired_at():
    r = _record(fired_at=1000.0, now=1001.5)
    assert r["fired_at"] == 1000.0
    assert r["decide_latency_ms"] == 1500.0


def test_path_derivation():
    # streak (speculate) wins regardless of transcript.
    assert _record(speculate=True, active_transcript="x")["path"] == "streak"
    # non-streak with a live transcript = turn-boundary.
    assert _record(speculate=False, active_transcript="x")["path"] == "turn_boundary"
    # non-streak with no live transcript = idle-flush.
    assert _record(speculate=False, active_transcript="")["path"] == "idle_flush"


def test_switched_record_carries_context_note_and_timestamp():
    r = _record(outcome="switched", switched_to="en", context_note="## Language note: respond in English")
    assert r["outcome"] == "switched"
    assert r["switched_to"] == "en"
    assert r["context_note_sent"] == "## Language note: respond in English"
    assert r["context_note_sent_at"] == r["fired_at"] + r["decide_latency_ms"] / 1000  # == now


def test_non_switch_outcomes_have_no_context_note():
    for outcome in ("stay", "gated:low_confidence", "gated:short_audio", "timeout"):
        r = _record(outcome=outcome, switched_to=None, context_note=None)
        assert r["outcome"] == outcome
        assert r["switched_to"] is None
        assert r["context_note_sent"] is None
        assert r["context_note_sent_at"] is None


def test_timeout_record_tolerates_missing_decision():
    # decide() timed out → no decision dict; the record must still build without KeyErrors.
    r = _record(outcome="timeout", decision=None)
    assert r["outcome"] == "timeout"
    assert r["target_language"] is None
    assert r["target_confidence"] is None
    assert r["reasoning"] == ""


def test_captures_sarvam_confidence_and_per_segment_languages():
    segs = [
        {"lang": "en", "prob": 0.97, "text": "hello", "audio_s": 1.2},
        {"lang": "ta", "prob": 0.91, "text": "நன்றி", "audio_s": 0.8},
    ]
    r = _record(detector_lang_confidence=0.91, detector_segments=segs)
    assert r["detector_lang_confidence"] == 0.91
    assert r["detector_segments"] == segs  # all languages in the turn, with confidence


def test_captures_llm_detection_independent_of_support():
    # Unsupported language: detected_language/detection_confidence still recorded; target stays null/0.
    r = _record(
        outcome="gated:unsupported",
        decision={
            "detected_language": "pa",
            "detection_confidence": 0.95,
            "target_language": None,
            "target_confidence": 0.0,
            "explicit_request": False,
            "reasoning": "Punjabi spoken; unsupported.",
        },
    )
    assert r["detected_language"] == "pa"
    assert r["detection_confidence"] == 0.95
    assert r["target_language"] is None
    assert r["target_confidence"] == 0.0


def test_detector_and_llm_extras_default_when_absent():
    r = _record(decision={})
    assert r["detector_lang_confidence"] is None
    assert r["detector_segments"] == []
    assert r["detected_language"] is None
    assert r["detection_confidence"] is None


def test_inflight_activity_captures_truncation_state():
    # Captured pre-truncate; audio_playing tells whether the old reply was cut mid-speech.
    activity = {"response_in_pipeline": True, "audio_playing": True, "pending_marks": True}
    r = _record(outcome="switched", switched_to="en", inflight_activity=activity)
    assert r["inflight_activity"] == activity
    assert r["inflight_activity"]["audio_playing"] is True


def test_inflight_activity_defaults_to_empty():
    assert _record()["inflight_activity"] == {}
