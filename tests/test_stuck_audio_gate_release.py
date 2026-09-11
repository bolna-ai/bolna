"""A held audio gate must release once the ASR stops producing new words.

Incident 0c64ec07: Deepgram re-emitted an identical non-final "hello" every ~0.6s for 2.4s while a
fully synthesized reply sat behind callee_speaking. Each duplicate refreshed liveness, so the
staleness escape never fired and the reply was discarded without playing a frame. Only a *changed*
transcript counts as new speech.
"""

import pytest

from unittest.mock import patch

from bolna.agent_manager.interruption_manager import InterruptionManager
from bolna.constants import STUCK_AUDIO_GATE_RELEASE_S

CLOCK = "bolna.agent_manager.interruption_manager.time.time"
T0 = 1_000.0


def _speaking_at(t0=T0):
    """A manager with the caller mid-utterance, as the output loop sees it while holding audio."""
    manager = InterruptionManager()
    with patch(CLOCK, return_value=t0):
        manager.on_user_speech_started()
    return manager


def _feed(manager, offsets_and_text):
    for offset, text in offsets_and_text:
        with patch(CLOCK, return_value=T0 + offset):
            manager.note_user_liveness(text)


def _staleness_at(manager, offset):
    with patch(CLOCK, return_value=T0 + offset):
        return manager.user_speech_staleness_s()


def test_repeated_identical_interims_let_the_gate_go_stale():
    # The 0c64ec07 shape: one word, then the ASR re-emitting it while endpointing lags.
    manager = _speaking_at()
    _feed(manager, [(0.0, "hello"), (1.0, "hello"), (1.6, "hello"), (2.2, "hello")])

    assert _staleness_at(manager, 2.4) == pytest.approx(2.4)
    assert _staleness_at(manager, 2.4) > STUCK_AUDIO_GATE_RELEASE_S


def test_new_words_keep_the_gate_held():
    # A caller genuinely still talking must never be spoken over.
    manager = _speaking_at()
    _feed(manager, [(0.0, "hello"), (1.0, "hello there"), (1.6, "hello there I"), (2.2, "hello there I need")])

    assert _staleness_at(manager, 2.4) == pytest.approx(0.2)
    assert _staleness_at(manager, 2.4) < STUCK_AUDIO_GATE_RELEASE_S


def test_the_gate_survives_a_long_utterance_of_new_words():
    # Staleness must not accumulate across a genuinely long turn.
    manager = _speaking_at()
    _feed(manager, [(i * 0.5, f"word {i}") for i in range(20)])

    assert _staleness_at(manager, 9.6) == pytest.approx(0.1)
    assert _staleness_at(manager, 9.6) < STUCK_AUDIO_GATE_RELEASE_S


def test_a_caller_who_goes_silent_still_goes_stale():
    # The original purpose of the escape: no interims at all after speech started.
    manager = _speaking_at()

    assert _staleness_at(manager, 2.5) == pytest.approx(2.5)
    assert _staleness_at(manager, 2.5) > STUCK_AUDIO_GATE_RELEASE_S


def test_liveness_without_a_transcript_still_refreshes():
    # Back-compat: any caller not passing content keeps the old unconditional behaviour.
    manager = _speaking_at()
    _feed(manager, [(0.0, ""), (2.2, "")])

    assert _staleness_at(manager, 2.4) == pytest.approx(0.2)


def test_speech_ended_clears_the_remembered_transcript():
    # Otherwise a next utterance opening with the same word would not refresh liveness.
    manager = _speaking_at()
    _feed(manager, [(0.0, "hello")])

    with patch(CLOCK, return_value=T0 + 1.0):
        manager.on_user_speech_ended()
    assert manager.last_interim_text == ""

    with patch(CLOCK, return_value=T0 + 5.0):
        manager.on_user_speech_started()
    _feed(manager, [(5.0, "hello")])
    assert _staleness_at(manager, 5.2) == pytest.approx(0.2)


def test_staleness_is_negative_when_the_caller_is_not_speaking():
    manager = InterruptionManager()
    assert manager.user_speech_staleness_s() == -1


def test_the_0c64ec07_timeline_releases_before_the_asr_finalizes():
    """Replay of the incident, offsets relative to callee_speaking_start_time (12:07:51.118).

    Reply was synthesized and waiting at +0.179s; Deepgram finalized "hello" only at +2.601s and the
    reply was then discarded unplayed. The gate must go stale before that final arrives.
    """
    manager = _speaking_at()
    interims = [(-0.001, "hello"), (1.000, "hello"), (1.601, "hello"), (2.201, "hello")]
    _feed(manager, interims)

    release_at = next(t / 1000 for t in range(0, 3000) if _staleness_at(manager, t / 1000) > STUCK_AUDIO_GATE_RELEASE_S)
    asr_final_at = 2.601
    assert release_at < asr_final_at, f"gate released at +{release_at}s, ASR final at +{asr_final_at}s"
    assert asr_final_at - release_at == pytest.approx(0.6, abs=0.05)


def test_playback_starting_is_what_spares_the_reply_from_the_overlapped_discard():
    """Why releasing the gate is sufficient for 0c64ec07, with no change to the overlap policy.

    The short final is only routed to the discard path because no audio was playing. Once the gate
    releases and playback starts, the same final is a false interruption, is ignored, and the guard
    returns before _handle_transcriber_output (and its overlapped/discard branch) is ever reached.
    """
    manager = InterruptionManager(number_of_words_for_interruption=3)
    incident = dict(word_count=1, transcript="hello", welcome_played=True)

    # What happened: reply stuck in WAIT, so is_audio_playing was False -> not a false interruption.
    assert manager.is_false_interruption(is_audio_playing=False, **incident) is False
    # With the gate released the reply is audible, so the same final is correctly ignored.
    assert manager.is_false_interruption(is_audio_playing=True, **incident) is True
