"""Tests for the __check_for_completion stall backstop (Layer 2).

Guarantees that a wedged pipeline (e.g. a stuck transcriber turn that holds audio and never
clears response_in_pipeline) cannot produce an endless call, while never firing during healthy
playback, in-flight tool/LLM generation, or normal short silences.
"""

from types import SimpleNamespace

from bolna.agent_manager.task_manager import TaskManager
from bolna.constants import STALL_HANGUP_FLOOR_S


# _should_stall_hangup only reads self.hang_conversation_after, so we can exercise the pure
# predicate on a lightweight stand-in via the unbound method.
def _decide(hang_conversation_after, *, audio_playing, has_pending_generation, ai_silent_s, user_silent_s):
    fake = SimpleNamespace(hang_conversation_after=hang_conversation_after)
    return TaskManager._should_stall_hangup(
        fake,
        audio_playing=audio_playing,
        has_pending_generation=has_pending_generation,
        time_since_last_spoken_ai_word=ai_silent_s,
        time_since_user_last_spoke=user_silent_s,
    )


AGED = STALL_HANGUP_FLOOR_S + 5  # comfortably past the floor


def test_fires_on_wedged_pipeline_stall():
    # The bug: audio held (not playing), nothing in flight, both sides long silent.
    assert _decide(15, audio_playing=False, has_pending_generation=False, ai_silent_s=AGED, user_silent_s=AGED) is True


def test_does_not_fire_during_audio_playback():
    # A long healthy TTS response: audio is playing -> never force-hangup.
    assert _decide(15, audio_playing=True, has_pending_generation=False, ai_silent_s=AGED, user_silent_s=AGED) is False


def test_does_not_fire_during_pending_generation():
    # A slow tool/LLM call in flight with no audio yet -> must not hang up.
    assert _decide(15, audio_playing=False, has_pending_generation=True, ai_silent_s=AGED, user_silent_s=AGED) is False


def test_does_not_fire_when_timers_fresh():
    assert _decide(15, audio_playing=False, has_pending_generation=False, ai_silent_s=1.0, user_silent_s=1.0) is False


def test_requires_both_sides_silent():
    # User just spoke (fresh) even though agent has been silent -> not a stall.
    assert _decide(15, audio_playing=False, has_pending_generation=False, ai_silent_s=AGED, user_silent_s=1.0) is False


def test_disabled_when_hang_conversation_after_zero():
    # hangup_after_silence disabled -> backstop also disabled.
    assert _decide(0, audio_playing=False, has_pending_generation=False, ai_silent_s=AGED, user_silent_s=AGED) is False


def test_floor_applies_when_hangup_after_silence_small():
    # hang_conversation_after=5, but floor is STALL_HANGUP_FLOOR_S: a 10s stall must NOT fire
    # (would be past the 5s normal timeout, but the backstop deliberately waits longer so the
    # normal inactivity path wins whenever it is reachable).
    short = STALL_HANGUP_FLOOR_S - 5
    assert short > 0
    assert _decide(5, audio_playing=False, has_pending_generation=False, ai_silent_s=short, user_silent_s=short) is False
    # Past the floor it does fire.
    assert _decide(5, audio_playing=False, has_pending_generation=False, ai_silent_s=AGED, user_silent_s=AGED) is True


def test_uses_larger_of_floor_and_configured_timeout():
    # A long configured timeout (60s) must be respected over the floor: a 30s stall must not fire.
    assert _decide(60, audio_playing=False, has_pending_generation=False, ai_silent_s=30, user_silent_s=30) is False
    assert _decide(60, audio_playing=False, has_pending_generation=False, ai_silent_s=65, user_silent_s=65) is True
