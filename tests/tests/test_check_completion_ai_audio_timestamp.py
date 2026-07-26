"""Regression guard for the mid-utterance inactivity hangup: the playout estimate on
MarkEventMetaData and compute_last_ai_audio_timestamp, which the watchdog reads as "AI last
spoke". last_transmitted_timestamp alone stays frozen through a long agent turn, so the watchdog
scored a still-speaking agent as silent and cut the call.
"""

import time
from types import SimpleNamespace

from bolna.agent_manager.task_manager import TaskManager
from bolna.helpers.mark_event_meta_data import MarkEventMetaData


# Exercised on a stand-in via the unbound method, as in test_check_completion_stall_backstop.
def _resolve(last_transmitted, audio_playing_until):
    fake = SimpleNamespace(
        last_transmitted_timestamp=last_transmitted,
        mark_event_meta_data=SimpleNamespace(get_audio_playing_until=lambda: audio_playing_until),
    )
    return TaskManager.compute_last_ai_audio_timestamp(fake)


def _content_mark(duration, mark_type="agent_response"):
    return {"type": mark_type, "sequence_id": 4, "duration": duration}


def test_audio_still_playing_reports_now():
    # Stamp frozen 90s back while audio still has 30s to play: measured silence must be ~0.
    now = time.time()
    assert _resolve(now - 90, now + 30) >= now - 0.5


def test_finished_playback_reports_when_audio_stopped():
    now = time.time()
    assert _resolve(now - 90, now - 12) == now - 12


def test_final_chunk_stamp_wins_when_more_recent():
    now = time.time()
    assert _resolve(now - 2, now - 12) == now - 2


def test_no_estimate_falls_back_to_final_chunk_stamp():
    # Marks registered without a duration, and calls before any audio, leave the estimate at 0.
    now = time.time()
    assert _resolve(now - 30, 0.0) == now - 30


def test_never_reports_older_than_final_chunk_stamp():
    # Silence can only shrink versus the stamp alone, so this can never hang up earlier.
    now = time.time()
    for playing_until in (0.0, now - 60, now, now + 60):
        assert _resolve(now - 30, playing_until) >= now - 30


def test_queued_audio_accumulates_beyond_send_time():
    # Three 10s chunks sent back to back must push the estimate ~30s out, not ~10s.
    mark = MarkEventMetaData()
    for i in range(3):
        mark.update_data(f"m{i}", _content_mark(10.0))
    assert 29 < mark.get_audio_playing_until() - time.time() <= 30


def test_estimate_resumes_from_now_after_playback_finished():
    # A turn after a gap starts its tail from now, not from the stale previous deadline.
    mark = MarkEventMetaData()
    mark.audio_playing_until = time.time() - 60
    mark.update_data("m0", _content_mark(5.0))
    assert 4 < mark.get_audio_playing_until() - time.time() <= 5


def test_interruption_drops_the_estimate():
    # Keeps a barge-in from leaving the watchdog permanently blocked.
    mark = MarkEventMetaData()
    mark.update_data("m0", _content_mark(30.0))
    mark.clear_data()
    assert mark.get_audio_playing_until() == 0.0


def test_online_check_prompt_does_not_extend_the_estimate():
    # The "are you still there" prompt must not postpone the inactivity hangup.
    mark = MarkEventMetaData()
    mark.update_data("m0", _content_mark(3.0, mark_type="is_user_online_message"))
    assert mark.get_audio_playing_until() == 0.0


def test_pre_mark_does_not_extend_the_estimate():
    mark = MarkEventMetaData()
    mark.update_data("m0", {"type": "pre_mark_message", "duration": 5.0})
    assert mark.get_audio_playing_until() == 0.0


def test_zero_duration_mark_does_not_reset_a_stale_estimate():
    # A mark for which no audio was queued (control packet, unmeasurable format) must not pull
    # the estimate forward to now, which would discard the accumulated silence.
    mark = MarkEventMetaData()
    mark.audio_playing_until = time.time() - 60
    mark.update_data("m0", _content_mark(0))
    assert mark.get_audio_playing_until() < time.time() - 59


def test_none_duration_does_not_raise():
    mark = MarkEventMetaData()
    mark.update_data("m0", _content_mark(None))
    assert mark.get_audio_playing_until() == 0.0


def test_decision_flips_for_long_turn_but_not_for_real_silence():
    # Replay with hangup_after_silence = 10s: stale stamp at now-17.6 either way.
    now = time.time()
    threshold = 10.0
    assert now - _resolve(now - 17.6, now + 4) < threshold  # 4s of audio left, must not cut
    assert now - _resolve(now - 17.6, now - 17.6) > threshold  # real silence, must still cut
