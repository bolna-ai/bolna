"""user_bot_latencies: the user's start and end must describe the same utterance on the same clock.

The start used to be when the first interim transcript arrived, an ASR delay after the caller began,
while the end is the provider's measured speech end — so short replies ended before they started.
"""

from unittest.mock import patch

from bolna.agent_manager import interruption_manager as module
from bolna.agent_manager.interruption_manager import InterruptionManager


def at(seconds):
    return patch.object(module.time, "time", return_value=seconds)


def test_the_provider_onset_is_the_user_start():
    im = InterruptionManager()
    with at(101.0):
        im.on_user_speech_started(user_start_ts_wall=100.0)  # first interim lands 1 s after onset
    with at(101.4):
        im.on_user_speech_ended(user_stop_ts_wall=100.4)
    with at(102.0):
        im.on_agent_speech_started(sequence_id=1)

    (entry,) = im.user_bot_latencies
    assert entry["user_start_s"] == 100.0
    assert entry["user_start_s"] < entry["user_end_s"] == 100.4


def test_without_an_onset_the_interim_arrival_is_kept():
    im = InterruptionManager()
    with at(101.0):
        im.on_user_speech_started()
    with at(101.4):
        im.on_user_speech_ended(user_stop_ts_wall=101.3)
    with at(102.0):
        im.on_agent_speech_started(sequence_id=1)

    assert im.user_bot_latencies[0]["user_start_s"] == 101.0


def test_the_previous_utterances_onset_is_not_reused():
    im = InterruptionManager()
    with at(101.0):
        im.on_user_speech_started(user_start_ts_wall=100.0)
    with at(101.4):
        im.on_user_speech_ended(user_stop_ts_wall=100.4)
    with at(105.0):
        im.on_user_speech_started(user_start_ts_wall=100.0)  # stale: the previous turn's onset

    assert im._last_user_start_ts == 105.0


def test_an_onset_in_the_future_is_ignored():
    im = InterruptionManager()
    with at(101.0):
        im.on_user_speech_started(user_start_ts_wall=101.5)

    assert im._last_user_start_ts == 101.0


def test_speech_begun_after_the_utterance_ended_does_not_move_its_start():
    im = InterruptionManager()
    with at(101.0):
        im.on_user_speech_started(user_start_ts_wall=100.0)
    with at(101.4):
        im.on_user_speech_ended(user_stop_ts_wall=100.4)
    with at(101.8):
        im.on_user_speech_started(user_start_ts_wall=101.6)  # caller starts again, never finishes
    with at(102.0):
        im.on_agent_speech_started(sequence_id=1)

    (entry,) = im.user_bot_latencies
    assert entry["user_start_s"] == 100.0
    assert entry["user_end_s"] == 100.4


def test_a_barge_in_event_records_the_onset():
    im = InterruptionManager()
    with at(101.0):
        im.on_user_speech_started(user_start_ts_wall=100.2)
        im.on_interruption_triggered(asr_turn_id=3)

    assert im.interruption_events[-1]["user_start_s"] == 100.2


def test_a_quick_resume_keeps_its_onset_even_when_the_previous_end_was_estimated_late():
    im = InterruptionManager()
    with at(101.0):
        im.on_user_speech_started(user_start_ts_wall=100.0)
    with at(102.0):
        im.on_user_speech_ended(stop_offset_ms=0)  # no provider stop: estimated at 102.0, after the real end
    with at(102.5):
        im.on_user_speech_started(user_start_ts_wall=101.8)

    assert im._last_user_start_ts == 101.8
