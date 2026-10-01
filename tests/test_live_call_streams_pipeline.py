"""A live call must run the streaming pipeline even when the synthesizer's stream flag is off.

The non-streaming branches are built for turn-based chats: they hand raw streaming-ASR events
to _handle_transcriber_output (which hangs up on the first interim) and never record the
agent's replies in history.
"""

from bolna.agent_manager.task_manager import streams_pipeline


def test_live_call_streams_when_synthesizer_stream_is_off():
    assert streams_pipeline({"stream": False}, turn_based_conversation=False, enforce_streaming=False)


def test_live_call_streams_when_synthesizer_stream_is_missing():
    assert streams_pipeline({}, turn_based_conversation=False, enforce_streaming=False)


def test_turn_based_chat_streams_only_when_enforced_and_synthesizer_streams():
    assert streams_pipeline({"stream": True}, turn_based_conversation=True, enforce_streaming=True)
    assert not streams_pipeline({"stream": True}, turn_based_conversation=True, enforce_streaming=False)
    assert not streams_pipeline({"stream": False}, turn_based_conversation=True, enforce_streaming=True)


def test_no_synthesizer_never_streams():
    assert not streams_pipeline(None, turn_based_conversation=False, enforce_streaming=True)
