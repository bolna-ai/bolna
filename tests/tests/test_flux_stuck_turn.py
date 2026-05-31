"""Tests for the Flux stuck-turn watchdog (Layer 1).

Repro: prod run 906511cf — a Flux turn opened from an Update interim, the server then
went application-silent (no further Flux events) so EndOfTurn never arrived, callee_speaking
stayed True, and the agent's audio was held forever. The watchdog must release such turns by
emitting speech_ended, without preempting healthy turns or injecting a phantom transcript.
"""

import asyncio

import pytest

from bolna.transcriber.deepgram_transcriber import DeepgramTranscriber


def _make_flux(output_queue=None, **kwargs):
    return DeepgramTranscriber(
        telephony_provider="plivo",
        model="flux-general-en",
        language="en",
        stream=True,
        output_queue=output_queue,
        **kwargs,
    )


def _make_nova(output_queue=None, **kwargs):
    return DeepgramTranscriber(
        telephony_provider="plivo",
        model="nova-3",
        language="en",
        stream=True,
        output_queue=output_queue,
        **kwargs,
    )


def _open_turn(t, last_msg_age_s):
    """Simulate a turn that has been opened (interim seen) with the last Flux message
    received `last_msg_age_s` seconds ago."""
    now = 1_000_000.0
    t.last_interim_time = now - last_msg_age_s
    t._last_flux_msg_time = now - last_msg_age_s
    t.meta_info = {"request_id": "test"}
    return now


def test_threshold_derived_from_eot_timeout():
    # Default eot_timeout_ms=500 -> max(3.0, 0.5*4=2.0) == 3.0
    assert _make_flux().flux_turn_stall_timeout_s == 3.0
    # Larger eot_timeout_ms scales the stall window: max(3.0, 4*4=16) == 16
    assert _make_flux(eot_timeout_ms=4000).flux_turn_stall_timeout_s == 16.0


def test_stalled_when_no_event_past_window():
    t = _make_flux()
    now = _open_turn(t, last_msg_age_s=t.flux_turn_stall_timeout_s + 1)
    assert t._flux_turn_is_stalled(now) is True


def test_not_stalled_within_window():
    t = _make_flux()
    now = _open_turn(t, last_msg_age_s=t.flux_turn_stall_timeout_s - 1)
    assert t._flux_turn_is_stalled(now) is False


def test_not_stalled_when_no_turn_open():
    t = _make_flux()
    # No interim seen yet -> not armed, even if _last_flux_msg_time is old.
    t.last_interim_time = None
    t._last_flux_msg_time = 1.0
    assert t._flux_turn_is_stalled(1_000_000.0) is False


def test_release_emits_single_speech_ended_and_disarms():
    q = asyncio.Queue()
    t = _make_flux(output_queue=q)
    _open_turn(t, last_msg_age_s=t.flux_turn_stall_timeout_s + 1)
    t.is_transcript_sent_for_processing = False  # mid-turn

    asyncio.run(t._release_stuck_flux_turn())

    # Exactly one speech_ended packet, no phantom transcript.
    assert q.qsize() == 1
    packet = q.get_nowait()
    assert packet["data"] == {"type": "speech_ended"}
    # Disarmed so the watchdog won't re-fire and a late EndOfTurn is suppressed.
    assert t.last_interim_time is None
    assert t.is_transcript_sent_for_processing is True
    assert t._flux_turn_is_stalled(1_000_000.0) is False


def test_nova_does_not_arm_flux_watchdog_state():
    # Nova path must not depend on Flux watchdog; predicate stays inert (no flux msgs).
    t = _make_nova()
    t.last_interim_time = 1.0
    assert t._last_flux_msg_time is None
    assert t._flux_turn_is_stalled(1_000_000.0) is False
