"""Tests for compute_last_ai_audio_timestamp — the inactivity watchdog's "AI last spoke"
reference (task_manager.__check_for_completion).

Regression guard for the mid-utterance inactivity hangup (calls 89ce0a14, 57956d2b): during a
long single agent turn, last_transmitted_timestamp (set only on the FINAL-chunk mark ack) stays
frozen while the input handler's per-chunk playback stamp (get_current_mark_started_time) keeps
advancing. The watchdog must key off the more recent of the two, so a still-speaking agent is not
mis-scored as silent, while genuine silence is still measured from the real last-audio moment.
"""

from types import SimpleNamespace

from bolna.agent_manager.task_manager import TaskManager


# compute_last_ai_audio_timestamp only reads self.last_transmitted_timestamp and
# self.tools["input"].get_current_mark_started_time(), so we exercise it on a lightweight
# stand-in via the unbound method (same approach as test_check_completion_stall_backstop).
def _resolve(last_transmitted, mark_started, *, with_input=True):
    tools = {}
    if with_input:
        tools["input"] = SimpleNamespace(get_current_mark_started_time=lambda: mark_started)
    fake = SimpleNamespace(last_transmitted_timestamp=last_transmitted, tools=tools)
    return TaskManager.compute_last_ai_audio_timestamp(fake)


def test_in_progress_playback_wins_over_stale_final_chunk():
    # The bug: final-chunk stamp frozen at the previous turn (100.0) while audio is still
    # playing now (195.0) -> must report the recent playback moment.
    assert _resolve(100.0, 195.0) == 195.0


def test_final_chunk_stamp_wins_when_more_recent():
    # After a turn fully completes, the final-chunk ack (200.0) is newer than the last
    # pre-mark stamp (150.0) -> report the final-chunk moment.
    assert _resolve(200.0, 150.0) == 200.0


def test_genuine_silence_keeps_the_frozen_stamp():
    # Audio ended; both stamps are frozen at the real last-audio moment, so silence is
    # still measured from it and a legitimate inactivity hangup can still fire.
    assert _resolve(120.0, 120.0) == 120.0


def test_missing_mark_time_falls_back_to_final_chunk():
    # get_current_mark_started_time() could be None -> must not crash or poison max().
    assert _resolve(150.0, None) == 150.0


def test_zero_mark_time_falls_back_to_final_chunk():
    assert _resolve(150.0, 0) == 150.0


def test_no_input_handler_uses_final_chunk_only():
    # Defensive: if the input tool is absent, fall back to last_transmitted_timestamp.
    assert _resolve(150.0, 999.0, with_input=False) == 150.0


def test_never_reports_older_than_final_chunk():
    # Invariant: the result is >= last_transmitted_timestamp, so measured AI-silence can only
    # shrink (never grow) vs the old behaviour -> the change can never cause an EARLIER hangup.
    for last_transmitted, mark in [(100.0, 90.0), (100.0, 100.0), (100.0, 110.0), (100.0, None)]:
        assert _resolve(last_transmitted, mark) >= last_transmitted


def test_decision_flips_for_long_turn_but_not_for_real_silence():
    # Concrete replay of call 57956d2b with hangup_after_silence = 10s.
    now = 1000.0
    threshold = 10.0
    # Long turn: agent finished speaking ~2s ago (playback stamp now-2), but the final-chunk
    # stamp is stale at now-17.6 -> old code would hang up mid-utterance; fixed code must not.
    long_turn_silence = now - _resolve(now - 17.6, now - 2.0)
    assert long_turn_silence < threshold
    # Real silence: no audio for 17.6s on either stamp -> still exceeds the threshold.
    real_silence = now - _resolve(now - 17.6, now - 17.6)
    assert real_silence > threshold
