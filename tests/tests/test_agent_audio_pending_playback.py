"""Tests for InterruptionManager.is_agent_audio_pending_playback — the completion watchdog's
"is the caller still hearing the agent?" guard (task_manager.__check_for_completion).

Regression guard for the mid-utterance trigger_user_online nudge (staging call 1170a1c5): the
final chunk of a long turn was a single ~16s audio segment whose is_final_chunk mark ack never
arrived before the watchdog acted, so agent_end_s stayed unset while the caller was still hearing
the closing line. compute_last_ai_audio_timestamp keeps the AI-silence clock fresh but cannot
cover a long single chunk (no intermediate marks), so the watchdog must also treat an un-acked
latest turn as busy and not nudge/hang up over it.
"""

from bolna.agent_manager.interruption_manager import InterruptionManager


def _mgr(user_bot_latencies):
    mgr = InterruptionManager()
    mgr.user_bot_latencies = user_bot_latencies
    return mgr


def test_no_turns_is_not_pending():
    assert _mgr([]).is_agent_audio_pending_playback() is False


def test_started_but_final_chunk_not_acked_is_pending():
    # agent_start_s set (turn began playing), agent_end_s absent (final-chunk ack not yet in).
    assert _mgr([{"sequence_id": 4, "agent_start_s": 82.4}]).is_agent_audio_pending_playback() is True


def test_fully_played_turn_is_not_pending():
    # agent_end_s set from the is_final_chunk mark ack -> playback confirmed done.
    assert _mgr([{"sequence_id": 4, "agent_start_s": 82.4, "agent_end_s": 100.1}]).is_agent_audio_pending_playback() is False


def test_only_latest_turn_matters():
    # An older completed turn does not keep the call "busy"; the newest turn (still playing) does.
    latencies = [
        {"sequence_id": 3, "agent_start_s": 42.6, "agent_end_s": 60.2},
        {"sequence_id": 4, "agent_start_s": 82.4},
    ]
    assert _mgr(latencies).is_agent_audio_pending_playback() is True


def test_latest_turn_completed_after_older_pending_is_not_pending():
    latencies = [
        {"sequence_id": 3, "agent_start_s": 42.6},
        {"sequence_id": 4, "agent_start_s": 82.4, "agent_end_s": 100.1},
    ]
    assert _mgr(latencies).is_agent_audio_pending_playback() is False
