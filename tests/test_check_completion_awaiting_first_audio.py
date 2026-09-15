"""Tests for the __check_for_completion idle gate (_pipeline_busy).

Regression for the inactivity prompt ("are you still there?") being synthesized onto a
response that had already been pushed to the synthesizer but whose first audio chunk had not
yet reached the SEND path. In that gap response_in_pipeline can read False (e.g. right after a
tool-call follow-up), so without _synthesis_awaiting_first_audio the checker saw the whole
pipeline idle and merged the prompt into the open synth stream.
"""

from types import SimpleNamespace

from bolna.agent_manager.task_manager import TaskManager


# _pipeline_busy only reads three flags, so we exercise the pure predicate on a lightweight
# stand-in via the unbound method (same pattern as test_check_completion_stall_backstop).
def _busy(*, audio_playing, response_in_pipeline, synthesis_awaiting_first_audio):
    fake = SimpleNamespace(
        response_in_pipeline=response_in_pipeline,
        _synthesis_awaiting_first_audio=synthesis_awaiting_first_audio,
    )
    return TaskManager._pipeline_busy(fake, audio_playing)


def test_busy_when_synthesis_pushed_but_no_audio_yet():
    # The bug window: response text pushed to the synthesizer, response_in_pipeline already
    # flipped False, first audio chunk not yet in the SEND path. Must read busy so the checker
    # does not emit the inactivity prompt onto the still-open stream.
    assert _busy(audio_playing=False, response_in_pipeline=False, synthesis_awaiting_first_audio=True) is True


def test_idle_when_nothing_in_flight():
    # Genuine idle: no audio, no response, no synthesis pending -> checker may act.
    assert _busy(audio_playing=False, response_in_pipeline=False, synthesis_awaiting_first_audio=False) is False


def test_busy_during_audio_playback():
    assert _busy(audio_playing=True, response_in_pipeline=False, synthesis_awaiting_first_audio=False) is True


def test_busy_while_response_generating():
    assert _busy(audio_playing=False, response_in_pipeline=True, synthesis_awaiting_first_audio=False) is True
