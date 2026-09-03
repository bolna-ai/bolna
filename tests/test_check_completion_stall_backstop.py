"""Tests for the __check_for_completion stall backstop (Layer 2).

Guarantees that a wedged pipeline (e.g. a stuck transcriber turn that holds audio and never
clears response_in_pipeline) cannot produce an endless call, while never firing during healthy
playback or normal short silences.
"""

from types import SimpleNamespace

from bolna.agent_manager.task_manager import TaskManager
from bolna.constants import STALL_HANGUP_HARD_CAP_S


# _should_stall_hangup only reads self.hang_conversation_after, so we can exercise the pure
# predicate on a lightweight stand-in via the unbound method.
def _decide(hang_conversation_after, *, audio_playing, ai_silent_s, user_silent_s):
    fake = SimpleNamespace(hang_conversation_after=hang_conversation_after)
    return TaskManager._should_stall_hangup(
        fake,
        audio_playing=audio_playing,
        time_since_last_spoken_ai_word=ai_silent_s,
        time_since_user_last_spoke=user_silent_s,
    )


AGED = STALL_HANGUP_HARD_CAP_S + 5  # comfortably past the hard cap


def test_fires_on_wedged_pipeline_stall():
    # The bug: audio held (not playing), both sides long silent past the hard cap.
    assert _decide(15, audio_playing=False, ai_silent_s=AGED, user_silent_s=AGED) is True


def test_does_not_fire_during_audio_playback():
    # A long healthy TTS response: audio is playing -> never force-hangup.
    assert _decide(15, audio_playing=True, ai_silent_s=AGED, user_silent_s=AGED) is False


def test_does_not_fire_when_timers_fresh():
    assert _decide(15, audio_playing=False, ai_silent_s=1.0, user_silent_s=1.0) is False


def test_requires_both_sides_silent():
    # User just spoke (fresh) even though agent has been silent -> not a stall.
    assert _decide(15, audio_playing=False, ai_silent_s=AGED, user_silent_s=1.0) is False


def test_disabled_when_hang_conversation_after_zero():
    # hangup_after_silence disabled -> backstop also disabled.
    assert _decide(0, audio_playing=False, ai_silent_s=AGED, user_silent_s=AGED) is False


def test_does_not_fire_below_the_hard_cap():
    below_cap = STALL_HANGUP_HARD_CAP_S - 5
    assert below_cap > 0
    assert _decide(15, audio_playing=False, ai_silent_s=below_cap, user_silent_s=below_cap) is False


def test_configured_patience_above_the_hard_cap_wins():
    # An agent that asked for more patience than the hard cap must get it - the cap is a
    # floor/insurance policy, not a ceiling that overrides what the agent configured.
    above_cap = STALL_HANGUP_HARD_CAP_S + 30
    assert _decide(above_cap, audio_playing=False, ai_silent_s=STALL_HANGUP_HARD_CAP_S + 5, user_silent_s=STALL_HANGUP_HARD_CAP_S + 5) is False
    assert _decide(above_cap, audio_playing=False, ai_silent_s=above_cap + 5, user_silent_s=above_cap + 5) is True
