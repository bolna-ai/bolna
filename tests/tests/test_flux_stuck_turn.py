"""Tests for the Flux stuck-turn watchdog (Layer 1).

Repros:
- prod run 906511cf — a Flux turn opened from an Update interim, the server then went
  application-silent (no further Flux events) so EndOfTurn never arrived, callee_speaking
  stayed True, and the agent's audio was held forever.
- prod run 9c9dc030 — same stuck turn, but Deepgram kept the socket chatty with
  empty-transcript Updates, so message-arrival liveness never detected the stall; the
  buffered interims ('Ok', 'तो') were real user speech that never reached the LLM.

The watchdog must therefore key on transcript progress (last_interim_time), and on release
must force-finalize buffered words so they reach the LLM — falling back to a bare
speech_ended (no phantom transcript) only when the stuck turn produced no text.
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


def _open_turn(t, last_interim_age_s):
    """Simulate a turn that has been opened (interim seen) with the last transcript-bearing
    Flux event received `last_interim_age_s` seconds ago."""
    now = 1_000_000.0
    t.last_interim_time = now - last_interim_age_s
    t.meta_info = {"request_id": "test"}
    return now


def test_threshold_derived_from_eot_timeout():
    # Default eot_timeout_ms=500 -> max(3.0, 0.5*4=2.0) == 3.0
    assert _make_flux().flux_turn_stall_timeout_s == 3.0
    # Larger eot_timeout_ms scales the stall window: max(3.0, 4*4=16) == 16
    assert _make_flux(eot_timeout_ms=4000).flux_turn_stall_timeout_s == 16.0


def test_stalled_when_no_transcript_progress_past_window():
    t = _make_flux()
    now = _open_turn(t, last_interim_age_s=t.flux_turn_stall_timeout_s + 1)
    assert t._flux_turn_is_stalled(now) is True


def test_not_stalled_within_window():
    t = _make_flux()
    now = _open_turn(t, last_interim_age_s=t.flux_turn_stall_timeout_s - 1)
    assert t._flux_turn_is_stalled(now) is False


def test_not_stalled_when_no_turn_open():
    t = _make_flux()
    # No interim seen yet -> not armed.
    t.last_interim_time = None
    assert t._flux_turn_is_stalled(1_000_000.0) is False


def test_release_with_no_text_emits_single_speech_ended_and_disarms():
    q = asyncio.Queue()
    t = _make_flux(output_queue=q)
    _open_turn(t, last_interim_age_s=t.flux_turn_stall_timeout_s + 1)
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


def test_release_with_buffered_interims_force_finalizes_transcript():
    q = asyncio.Queue()
    t = _make_flux(output_queue=q)
    now = _open_turn(t, last_interim_age_s=t.flux_turn_stall_timeout_s + 1)
    t.is_transcript_sent_for_processing = False
    t.current_turn_interim_details = [
        {"transcript": "Ok", "latency_ms": None, "is_final": False, "received_at": now - 5},
        {"transcript": "Ok तो", "latency_ms": None, "is_final": False, "received_at": now - 4},
    ]

    asyncio.run(t._release_stuck_flux_turn())

    # The buffered words are delivered to the LLM as a force-finalized transcript.
    assert q.qsize() == 1
    packet = q.get_nowait()
    assert packet["data"]["type"] == "transcript"
    assert packet["data"]["content"] == "Ok तो"
    assert packet["data"]["force_finalized"] is True
    # Disarmed so the watchdog won't re-fire and a late EndOfTurn is suppressed.
    assert t.last_interim_time is None
    assert t.is_transcript_sent_for_processing is True
    assert t._flux_turn_is_stalled(1_000_000.0) is False


def test_nova_does_not_arm_flux_watchdog_state():
    # Nova sets last_interim_time too (its own monitor uses it); the flux stall predicate
    # must stay inert for non-flux models.
    t = _make_nova()
    t.last_interim_time = 1.0
    assert t._flux_turn_is_stalled(1_000_000.0) is False
