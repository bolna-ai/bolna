"""The truncate-before-switch block in __run_language_switch, driven through the real method.

Truncating wipes the mark dict, so the final-chunk ack that clears is_audio_being_played_to_user
never arrives — if the switch path doesn't clear it, the flag latches True and disables both the
silence prompt and the stall-hangup backstop for the rest of the call. The gap that follows is an
await like any other, so teardown starting during it must abandon the switch.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from bolna.agent_manager.task_manager import TaskManager
from bolna.synthesizer.synthesizer_pool import SynthesizerPool
from bolna.transcriber.transcriber_pool import TranscriberPool

DECISION = {"target_language": "mr", "target_confidence": 0.95, "reasoning": "clear Marathi"}


def _tm(monkeypatch, gap=0.0, audio_playing=True):
    monkeypatch.setenv("LANGUAGE_SWITCH_SETTLE_MS", "0")  # skip the detector-tail settle
    tm = MagicMock()
    tm.task_config = {
        "tools_config": {
            "llm_agent": {"agent_type": "graph_agent"},  # suppress speculation
            "language_switch_audio_gap_s": gap,
        }
    }
    tm.language = "hi"
    tm.conversation_ended = False
    tm.hangup_triggered = False
    tm.function_call_in_flight = False
    tm.multilingual_prompts = {"hi": "p", "mr": "p"}
    tm._should_ignore_transcriber_input = MagicMock(return_value=False)

    pool = MagicMock(spec=TranscriberPool)
    pool.labels = ["hi", "mr"]
    pool.lid_detection_events = []
    pool.lid_buffer_max_segment_seconds.return_value = 2.0
    pool.lid_buffer_language_confidence.return_value = 0.9
    # Corroboration is per-segment now: prob and duration must describe the SAME utterance.
    pool.lid_buffer_segments.return_value = [{"lang": "mr", "prob": 0.9, "audio_s": 2.0}]
    pool.take_lid_transcript.return_value = ("mala samajla nahi", "mr")
    synth = MagicMock(spec=SynthesizerPool)
    synth.labels = ["hi", "mr"]
    tm.tools = {"transcriber": pool, "synthesizer": synth, "input": MagicMock()}

    tm.language_switcher = MagicMock()
    tm.language_switcher.decide = AsyncMock(return_value=DECISION)
    tm._inflight_response_activity = MagicMock(
        return_value={"audio_playing": audio_playing, "response_in_pipeline": True}
    )
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


async def _run(tm, active_transcript="garbled hi"):
    run = TaskManager._TaskManager__run_language_switch.__get__(tm, TaskManager)
    return await run(active_transcript, {"sequence_id": 1}, "hi")


def _outcomes(tm):
    return [e.get("outcome") for e in tm.tools["transcriber"].lid_detection_events]


@pytest.mark.asyncio
async def test_audio_flag_cleared_on_truncate(monkeypatch):
    tm = _tm(monkeypatch)
    await _run(tm)
    tm.tools["input"].update_is_audio_being_played.assert_called_once_with(False)
    tm.switch_language.assert_awaited_once()
    assert "switched" in _outcomes(tm)


@pytest.mark.asyncio
async def test_gap_skipped_when_no_audio_was_playing(monkeypatch):
    # Nothing reached the caller, so a gap would be pure dead air — a 5s gap must not be paid.
    tm = _tm(monkeypatch, gap=5.0, audio_playing=False)
    await asyncio.wait_for(_run(tm), timeout=2.0)
    tm.switch_language.assert_awaited_once()


@pytest.mark.asyncio
async def test_gap_sleeps_when_audio_was_playing(monkeypatch):
    tm = _tm(monkeypatch, gap=0.2)
    started = asyncio.get_event_loop().time()
    await _run(tm)
    assert asyncio.get_event_loop().time() - started >= 0.2
    tm.switch_language.assert_awaited_once()


@pytest.mark.asyncio
async def test_zero_gap_does_not_sleep(monkeypatch):
    tm = _tm(monkeypatch, gap=0)
    await asyncio.wait_for(_run(tm), timeout=1.0)
    tm.switch_language.assert_awaited_once()


@pytest.mark.asyncio
async def test_switch_abandoned_when_hangup_starts_during_gap(monkeypatch):
    tm = _tm(monkeypatch, gap=0.2)

    async def hangup_midway():
        await asyncio.sleep(0.05)
        tm.conversation_ended = True

    asyncio.create_task(hangup_midway())
    assert await _run(tm) is None
    tm.switch_language.assert_not_awaited()  # pools must NOT flip under a goodbye
    tm._TaskManager__play_switch_handoff.assert_not_awaited()
    assert "gated:hangup" in _outcomes(tm)


@pytest.mark.asyncio
async def test_switch_abandoned_on_transfer_during_gap(monkeypatch):
    # _should_ignore_transcriber_input covers _end_call_in_progress / has_transfer, which the
    # downstream handoff and follow-up guards do NOT check — so only this re-check catches them.
    tm = _tm(monkeypatch, gap=0.2)

    async def transfer_midway():
        await asyncio.sleep(0.05)
        tm._should_ignore_transcriber_input.return_value = True

    asyncio.create_task(transfer_midway())
    assert await _run(tm) is None
    tm.switch_language.assert_not_awaited()
    assert "gated:hangup" in _outcomes(tm)


@pytest.mark.asyncio
async def test_detector_corroboration_admits_a_lower_llm_confidence(monkeypatch):
    # Detector independently agrees on the target at high prob with substantive audio, so a 0.6
    # self-report switches where the bare 0.7 gate would have refused.
    tm = _tm(monkeypatch)
    tm.language_switcher.decide = AsyncMock(
        return_value={"target_language": "mr", "target_confidence": 0.6, "reasoning": "leaning Marathi"}
    )
    await _run(tm)
    tm.switch_language.assert_awaited_once()
    assert "switched" in _outcomes(tm)


@pytest.mark.asyncio
async def test_no_corroboration_keeps_the_full_bar(monkeypatch):
    # Detector disagrees with the target, so the 0.7 bar stands and 0.6 is refused.
    tm = _tm(monkeypatch)
    tm.tools["transcriber"].take_lid_transcript.return_value = ("text", "hi")
    tm.tools["transcriber"].lid_buffer_segments.return_value = [{"lang": "hi", "prob": 0.95, "audio_s": 2.0}]
    tm.language_switcher.decide = AsyncMock(
        return_value={"target_language": "mr", "target_confidence": 0.6, "reasoning": "unsure"}
    )
    await _run(tm)
    tm.switch_language.assert_not_awaited()
    assert "gated:low_confidence" in _outcomes(tm)


@pytest.mark.asyncio
async def test_short_audio_is_not_corroborating_evidence(monkeypatch):
    # A sub-second fragment cannot lend its confidence to a switch: corroboration requires ONE
    # segment to carry the target tag, the prob AND the duration, so the aggregates can no longer
    # be mixed across segments (a 0.3s "okay" at token-share 1.0 borrowing a 3s turn's substance).
    tm = _tm(monkeypatch)
    tm.tools["transcriber"].lid_buffer_max_segment_seconds.return_value = 0.5
    tm.tools["transcriber"].lid_buffer_segments.return_value = [{"lang": "mr", "prob": 1.0, "audio_s": 0.5}]
    tm.language_switcher.decide = AsyncMock(
        return_value={"target_language": "mr", "target_confidence": 0.6, "reasoning": "short"}
    )
    await _run(tm)
    tm.switch_language.assert_not_awaited()
    assert "gated:low_confidence" in _outcomes(tm)  # refused before the substance gate is reached


@pytest.mark.asyncio
async def test_corroboration_ignored_on_low_detector_prob(monkeypatch):
    tm = _tm(monkeypatch)
    tm.tools["transcriber"].lid_buffer_segments.return_value = [{"lang": "mr", "prob": 0.4, "audio_s": 2.0}]
    tm.language_switcher.decide = AsyncMock(
        return_value={"target_language": "mr", "target_confidence": 0.6, "reasoning": "weak tag"}
    )
    await _run(tm)
    tm.switch_language.assert_not_awaited()
    assert "gated:low_confidence" in _outcomes(tm)


@pytest.mark.asyncio
async def test_settle_skipped_when_detector_already_quiet(monkeypatch):
    # The settle exists to let the detector's socket deliver this turn's tail. If it has been
    # quiet longer than the settle window nothing is in flight, so waiting only holds the lock.
    tm = _tm(monkeypatch)
    monkeypatch.setenv("LANGUAGE_SWITCH_SETTLE_MS", "400")  # after _tm, which zeroes it
    tm.tools["transcriber"].lid_buffer_age.return_value = 1.5  # quiet well past 0.4s
    started = asyncio.get_event_loop().time()
    await _run(tm)
    assert asyncio.get_event_loop().time() - started < 0.3
    tm.switch_language.assert_awaited_once()


@pytest.mark.asyncio
async def test_settle_still_paid_when_a_segment_just_landed(monkeypatch):
    # A segment arrived <settle ago, so more of this turn may still be in flight — wait for it.
    tm = _tm(monkeypatch)
    monkeypatch.setenv("LANGUAGE_SWITCH_SETTLE_MS", "250")  # after _tm, which zeroes it
    tm.tools["transcriber"].lid_buffer_age.return_value = 0.05
    started = asyncio.get_event_loop().time()
    await _run(tm)
    assert asyncio.get_event_loop().time() - started >= 0.25
