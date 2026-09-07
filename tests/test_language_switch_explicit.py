"""Explicit-only judge mode (per-agent toggle): switches only on an explicit
request/selection/confirmation; speaking another language alone never switches."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

from bolna.prompts import (
    EXPLICIT_LANGUAGE_SWITCH_SYSTEM_PROMPT,
    LANGUAGE_SWITCH_SYSTEM_PROMPT,
)

MOD = "bolna.helpers.language_switcher"


def make_switcher(generate_return, labels=("en", "hi", "te"), explicit_only=True):
    from bolna.helpers.language_switcher import LanguageSwitcher

    fake_llm = MagicMock()
    fake_llm.generate = AsyncMock(return_value=(generate_return, {}))
    with patch(f"{MOD}.LiteLLM", return_value=fake_llm):
        switcher = LanguageSwitcher(available_labels=list(labels), explicit_only=explicit_only)
    return switcher, fake_llm


def system_text(fake_llm):
    return fake_llm.generate.await_args.args[0][0]["content"][0]["text"]


def turn_text(fake_llm):
    return fake_llm.generate.await_args.args[0][-1]["content"]


# ── prompt selection ──────────────────────────────────────────────────────────────


async def test_explicit_mode_uses_explicit_prompts_and_last_agent_turn():
    payload = json.dumps({"target_language": None, "request_status": "no_request"})
    switcher, fake_llm = make_switcher(payload)
    await switcher.decide(
        "मेरा order कहाँ है?",
        "mera order kahan",
        active_label="en",
        last_agent_turn="How can I help you today?",
    )
    assert system_text(fake_llm) == EXPLICIT_LANGUAGE_SWITCH_SYSTEM_PROMPT
    turn = turn_text(fake_llm)
    assert "last_agent_turn: How can I help you today?" in turn
    assert "unbiased_user_transcript" in turn
    assert "RECENT TURNS" not in turn


async def test_ambient_mode_prompts_unchanged_and_ignores_last_agent_turn():
    payload = json.dumps({"languages": [], "target_language": None, "reasoning": "stay"})
    switcher, fake_llm = make_switcher(payload, explicit_only=False)
    await switcher.decide(
        "hello",
        "hello",
        active_label="en",
        recent_turns=[("hi", 2.1)],
        last_agent_turn="Would you like Hindi?",
    )
    assert system_text(fake_llm) == LANGUAGE_SWITCH_SYSTEM_PROMPT
    turn = turn_text(fake_llm)
    assert "Would you like Hindi?" not in turn
    assert "RECENT TURNS" in turn


async def test_explicit_mode_empty_last_agent_turn_is_none_marker():
    payload = json.dumps({"target_language": None})
    switcher, fake_llm = make_switcher(payload)
    await switcher.decide("hello", "hello", active_label="en", last_agent_turn=None)
    assert "last_agent_turn: (none)" in turn_text(fake_llm)


async def test_explicit_mode_empty_live_stays_empty_not_marker():
    # The explicit prompt handles empty LIVE itself; the ambient marker must not leak in.
    payload = json.dumps({"target_language": None})
    switcher, fake_llm = make_switcher(payload)
    await switcher.decide("नमस्ते", "", active_label="en")
    turn = turn_text(fake_llm)
    assert 'live_user_transcript: ""' in turn
    assert "idle flush" not in turn


# ── the user's canonical examples, end to end through decide() ────────────────────


async def test_canonical_no_request_stays():
    payload = json.dumps(
        {
            "detected_language": "hi",
            "detection_confidence": 0.98,
            "explicit_request": False,
            "requested_language": None,
            "target_language": None,
            "target_confidence": 0,
            "request_status": "no_request",
            "request_source": "none",
            "reasoning": "Caller spoke Hindi but made no language request",
        }
    )
    switcher, _ = make_switcher(payload)
    result = await switcher.decide("मेरा order कहाँ है?", "", active_label="en", last_agent_turn=None)
    assert result["target_language"] is None
    assert result["request_status"] == "no_request"


async def test_canonical_one_word_answer_switches():
    payload = json.dumps(
        {
            "detected_language": "en",
            "detection_confidence": 0.85,
            "explicit_request": True,
            "requested_language": "te",
            "target_language": "te",
            "target_confidence": 0.99,
            "request_status": "switch",
            "request_source": "agent_prompted_selection",
            "reasoning": "Caller selected Telugu from the offered languages",
        }
    )
    switcher, fake_llm = make_switcher(payload)
    result = await switcher.decide(
        "Telugu.",
        "Telugu.",
        active_label="en",
        last_agent_turn="Would you prefer Hindi or Telugu?",
    )
    assert result["target_language"] == "te"
    assert result["explicit_request"] is True
    assert "Would you prefer Hindi or Telugu?" in turn_text(fake_llm)


async def test_canonical_yes_to_two_options_is_ambiguous_stay():
    payload = json.dumps(
        {
            "detected_language": "en",
            "detection_confidence": 0.95,
            "explicit_request": False,
            "requested_language": None,
            "target_language": None,
            "target_confidence": 0,
            "request_status": "ambiguous",
            "request_source": "none",
            "reasoning": "Affirmation does not choose between two offered languages",
        }
    )
    switcher, _ = make_switcher(payload)
    result = await switcher.decide(
        "Yes.", "Yes.", active_label="en", last_agent_turn="Would you prefer Hindi or Tamil?"
    )
    assert result["target_language"] is None
    assert result["request_status"] == "ambiguous"


# ── task_manager: explicit mode bypasses detection gates, keeps structural ones ───

from bolna.agent_manager.task_manager import TaskManager
from bolna.synthesizer.synthesizer_pool import SynthesizerPool
from bolna.transcriber.transcriber_pool import TranscriberPool


def make_tm(monkeypatch, decision, explicit_only=True, synth_labels=("en", "hi")):
    monkeypatch.setenv("LANGUAGE_SWITCH_SETTLE_MS", "0")
    tm = MagicMock()

    tm.task_config = {"tools_config": {"llm_agent": {"agent_type": "graph_agent"}}}
    tm.language = "en"
    tm.conversation_ended = False
    tm.hangup_triggered = False
    tm.function_call_in_flight = False
    tm.multilingual_prompts = {"en": "p", "hi": "p"}
    tm._should_ignore_transcriber_input = MagicMock(return_value=False)
    pool = MagicMock(spec=TranscriberPool)
    pool.labels = ["en", "hi"]
    pool.lid_detection_events = []
    pool.lid_buffer_max_segment_seconds.return_value = 0.3
    pool.lid_buffer_language_confidence.return_value = 0.4
    # A one-word answer: far below every ambient substance/confidence bar.
    pool.lid_buffer_segments.return_value = [{"lang": "en", "prob": 0.4, "audio_s": 0.3}]
    pool.take_lid_transcript.return_value = ("Hindi.", "en")
    synth = MagicMock(spec=SynthesizerPool)
    synth.labels = list(synth_labels)
    tm.tools = {"transcriber": pool, "synthesizer": synth, "input": MagicMock()}
    tm.language_switcher = MagicMock()
    tm.language_switcher.explicit_only = explicit_only
    tm.language_switcher.decide = AsyncMock(return_value=decision)
    tm._inflight_response_activity = MagicMock(return_value={"audio_playing": False})
    tm._TaskManager__cleanup_downstream_tasks = AsyncMock()
    tm.switch_language = AsyncMock()
    tm._TaskManager__language_directive = MagicMock(return_value="note")
    tm._TaskManager__play_switch_handoff = AsyncMock()
    tm._TaskManager__prepare_followup_generation = MagicMock(return_value=None)
    tm.conversation_history = MagicMock()
    tm.conversation_history.replace_last_user.return_value = True
    tm.conversation_history.last_assistant_content.return_value = "Would you like Hindi?"
    for name in ("switch_audio_gap_s", "switch_settle_ms", "switch_decide_timeout_s", "record_lid_event"):
        attr = f"_TaskManager__{name}"
        setattr(tm, attr, getattr(TaskManager, attr).__get__(tm, TaskManager))
    tm._TaskManager__detector_corroborates = TaskManager._TaskManager__detector_corroborates
    return tm


async def run_switch(tm):
    run = TaskManager._TaskManager__run_language_switch.__get__(tm, TaskManager)
    return await run("Hindi.", {"sequence_id": 1}, "en")


def outcomes(tm):
    return [e.get("outcome") for e in tm.tools["transcriber"].lid_detection_events]


EXPLICIT_SWITCH = {
    "detected_language": "en",
    "detection_confidence": 0.85,
    "explicit_request": True,
    "requested_language": "hi",
    "target_language": "hi",
    "target_confidence": 0.99,
    "request_status": "switch",
    "request_source": "agent_prompted_selection",
    "reasoning": "Caller selected Hindi",
}


async def test_explicit_mode_switches_past_all_detection_gates(monkeypatch):
    # 0.3s low-prob segment: ambient mode would gate this on confidence corroboration
    # or substance; explicit mode must switch.
    tm = make_tm(monkeypatch, dict(EXPLICIT_SWITCH))
    await run_switch(tm)
    tm.switch_language.assert_awaited_once()
    assert "switched" in outcomes(tm)


async def test_explicit_mode_low_self_confidence_still_switches(monkeypatch):
    decision = dict(EXPLICIT_SWITCH, target_confidence=0.2)
    tm = make_tm(monkeypatch, decision)
    await run_switch(tm)
    tm.switch_language.assert_awaited_once()


async def test_ambient_mode_same_evidence_is_gated(monkeypatch):
    # Same weak evidence, ambient prompt shape: must NOT switch (confidence 0.2 < bar).
    decision = {"target_language": "hi", "target_confidence": 0.2, "reasoning": "r", "languages": []}
    tm = make_tm(monkeypatch, decision, explicit_only=False)
    await run_switch(tm)
    tm.switch_language.assert_not_awaited()
    assert "gated:low_confidence" in outcomes(tm)


async def test_explicit_mode_unsupported_target_still_blocked(monkeypatch):
    decision = dict(EXPLICIT_SWITCH, target_language="ta", requested_language="ta")
    tm = make_tm(monkeypatch, decision)
    await run_switch(tm)
    tm.switch_language.assert_not_awaited()
    assert "gated:unsupported" in outcomes(tm)


async def test_explicit_mode_missing_synth_voice_still_blocked(monkeypatch):
    tm = make_tm(monkeypatch, dict(EXPLICIT_SWITCH), synth_labels=("en",))
    await run_switch(tm)
    tm.switch_language.assert_not_awaited()
    assert "gated:no_synth" in outcomes(tm)


async def test_explicit_mode_null_target_stays(monkeypatch):
    decision = {
        "detected_language": "hi",
        "explicit_request": False,
        "requested_language": None,
        "target_language": None,
        "target_confidence": 0,
        "request_status": "no_request",
        "request_source": "none",
        "reasoning": "no request",
    }
    tm = make_tm(monkeypatch, decision)
    await run_switch(tm)
    tm.switch_language.assert_not_awaited()
    assert outcomes(tm) == ["stay"]


async def test_telemetry_record_carries_request_fields(monkeypatch):
    tm = make_tm(monkeypatch, dict(EXPLICIT_SWITCH))
    await run_switch(tm)
    record = tm.tools["transcriber"].lid_detection_events[-1]
    assert record["request_status"] == "switch"
    assert record["request_source"] == "agent_prompted_selection"


async def test_explicit_mode_target_without_verdict_is_gated(monkeypatch):
    # The exact shape from review: a target with status=ambiguous / explicit_request=false
    # must not switch — rule 18 of the prompt is not the only guard anymore.
    decision = {
        "detected_language": "hi",
        "explicit_request": False,
        "requested_language": None,
        "target_language": "hi",
        "target_confidence": 0.0,
        "request_status": "ambiguous",
        "request_source": "none",
        "reasoning": "drift",
    }
    tm = make_tm(monkeypatch, decision)
    await run_switch(tm)
    tm.switch_language.assert_not_awaited()
    assert "gated:not_explicit" in outcomes(tm)


async def test_explicit_mode_status_switch_but_no_explicit_flag_is_gated(monkeypatch):
    decision = dict(EXPLICIT_SWITCH, explicit_request=False)
    tm = make_tm(monkeypatch, decision)
    await run_switch(tm)
    tm.switch_language.assert_not_awaited()
    assert "gated:not_explicit" in outcomes(tm)
