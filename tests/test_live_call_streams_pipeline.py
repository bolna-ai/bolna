"""Live calls always run the streaming pipeline; turn-based chats stream only when enforced."""

from bolna.agent_manager.task_manager import streams_pipeline


def test_live_calls_always_stream():
    assert streams_pipeline(turn_based_conversation=False, enforce_streaming=False)
    assert streams_pipeline(turn_based_conversation=False, enforce_streaming=True)


def test_turn_based_chats_stream_only_when_enforced():
    assert streams_pipeline(turn_based_conversation=True, enforce_streaming=True)
    assert not streams_pipeline(turn_based_conversation=True, enforce_streaming=False)
