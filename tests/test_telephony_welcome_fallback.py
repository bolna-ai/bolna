"""A text greeting should not be silently skipped when cached audio is absent."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from bolna.agent_manager.task_manager import TaskManager


def make_manager(text, audio):
    manager = TaskManager.__new__(TaskManager)
    manager.welcome_message_delay = 0
    manager._TaskManager__await_stream_sid = AsyncMock(return_value=True)
    manager.kwargs = {"agent_welcome_message": text}
    manager.tools = {
        "input": SimpleNamespace(is_welcome_message_played=False),
        "output": SimpleNamespace(get_provider=Mock(return_value="twilio")),
    }
    manager.task_config = {"tools_config": {"output": {"format": "pcm"}}}
    manager.preloaded_welcome_audio = audio
    manager.stream_sid = "stream-1"
    manager.conversation_history = SimpleNamespace(append_welcome_message=Mock())
    manager._synthesize = AsyncMock()
    return manager


@pytest.mark.parametrize("audio", [None, b""])
async def test_uncached_telephony_welcome_is_synthesized(audio):
    manager = make_manager("Hello there", audio)

    await manager._TaskManager__forced_first_message()

    manager._synthesize.assert_awaited_once()
    packet = manager._synthesize.await_args.args[0]
    assert packet["data"] == "Hello there"
    assert packet["meta_info"]["message_category"] == "agent_welcome_message"
    assert packet["meta_info"]["cached"] is False
    assert manager.tools["input"].is_welcome_message_played is False
    manager.conversation_history.append_welcome_message.assert_called_once_with("Hello there")


@pytest.mark.parametrize("text", [None, "", "   "])
async def test_no_welcome_text_does_not_synthesize(text):
    manager = make_manager(text, None)

    await manager._TaskManager__forced_first_message()

    manager._synthesize.assert_not_awaited()
    assert manager.tools["input"].is_welcome_message_played is True
    manager.conversation_history.append_welcome_message.assert_not_called()


async def test_uncached_welcome_reaches_synthesizer(monkeypatch):
    import bolna.agent_manager.task_manager as task_manager

    manager = make_manager("Hello there", None)
    manager._synthesize = TaskManager._synthesize.__get__(manager)
    manager.turn_based_conversation = False
    manager.conversation_start_init_ts = 0
    manager.conversation_ended = False
    manager.interruption_manager = SimpleNamespace(is_valid_sequence=lambda sequence: sequence == -1)
    manager.synthesizer_provider = next(iter(task_manager.SUPPORTED_SYNTHESIZER_MODELS))
    manager.synthesizer_characters = 0
    manager.run_id = "test"
    manager.tools["synthesizer"] = SimpleNamespace(get_engine=lambda: "test", push=AsyncMock())
    monkeypatch.setattr(task_manager, "convert_to_request_log", lambda **kwargs: None)

    await manager._TaskManager__forced_first_message()

    manager.tools["synthesizer"].push.assert_awaited_once()
    packet = manager.tools["synthesizer"].push.await_args.args[0]
    assert packet["data"] == "Hello there"
    assert packet["meta_info"]["message_category"] == "agent_welcome_message"
    assert manager._synthesis_awaiting_first_audio is True


async def test_cached_audio_still_uses_direct_playback(monkeypatch):
    import bolna.agent_manager.task_manager as task_manager

    manager = make_manager("Hello there", b"\0" * 160)
    manager.tools["input"].update_is_audio_being_played = Mock()
    manager.tools["output"].handle = AsyncMock()
    manager.tools["synthesizer"] = SimpleNamespace(get_engine=lambda: "test")
    manager.synthesizer_provider = "test"
    manager.run_id = "test"
    manager.sampling_rate = 8000
    manager.should_record = False
    monkeypatch.setattr(task_manager, "convert_to_request_log", lambda **kwargs: None)

    await manager._TaskManager__forced_first_message()

    manager._synthesize.assert_not_awaited()
    manager.tools["output"].handle.assert_awaited_once()
    assert manager.tools["output"].handle.await_args.args[0]["data"] == b"\0" * 160
    assert manager.tools["input"].is_welcome_message_played is False


async def test_stream_timeout_does_not_start_greeting():
    manager = make_manager("Hello there", None)
    manager._TaskManager__await_stream_sid.return_value = False

    await manager._TaskManager__forced_first_message()

    manager._synthesize.assert_not_awaited()
    manager.conversation_history.append_welcome_message.assert_not_called()
