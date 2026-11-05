"""The telephony welcome plays even when its audio wasn't pre-generated before the call."""

from unittest.mock import AsyncMock, MagicMock, patch

from bolna.agent_manager.task_manager import TaskManager


def task_manager(welcome_text, preloaded_audio):
    tm = TaskManager.__new__(TaskManager)
    tm.welcome_message_delay = 0
    tm.kwargs = {"agent_welcome_message": welcome_text}
    tm.preloaded_welcome_audio = preloaded_audio
    tm.task_config = {"tools_config": {"output": {"format": "wav"}}}
    tm.tools = {"input": MagicMock(), "output": MagicMock(), "synthesizer": MagicMock()}
    tm.tools["input"].is_welcome_message_played = False
    tm.tools["input"].welcome_message_played = lambda: tm.tools["input"].is_welcome_message_played
    tm.tools["output"].get_welcome_message_sent_ts = MagicMock(return_value=None)
    tm.tools["output"].get_provider = MagicMock(return_value="plivo")
    tm.tools["output"].handle = AsyncMock()
    tm.conversation_history = MagicMock()
    tm.synthesizer_provider = "elevenlabs"
    tm.run_id = "run-1"
    tm.sampling_rate = 8000
    tm.should_record = False
    tm.stream_sid = "stream-1"
    tm._TaskManager__await_stream_sid = AsyncMock(return_value=True)
    tm._synthesize = AsyncMock()
    return tm


async def run_forced_first_message(tm):
    with (
        patch("bolna.agent_manager.task_manager.convert_to_request_log"),
        patch("bolna.agent_manager.task_manager.LIVE_WELCOME_AUDIO_TIMEOUT", 0),
    ):
        await tm._TaskManager__forced_first_message()


async def test_missing_welcome_audio_is_synthesized_live():
    tm = task_manager("Hi Asha, this is Sonia.", None)
    tm.tools["output"].get_welcome_message_sent_ts = MagicMock(return_value=1700000000000)

    await run_forced_first_message(tm)

    tm._synthesize.assert_awaited_once()
    packet = tm._synthesize.await_args.args[0]
    assert packet["data"] == "Hi Asha, this is Sonia."
    assert packet["meta_info"]["cached"] is False
    assert packet["meta_info"]["message_category"] == "agent_welcome_message"
    assert packet["meta_info"]["sequence_id"] == -1
    tm.conversation_history.append_welcome_message.assert_called_once_with("Hi Asha, this is Sonia.")
    tm.tools["output"].handle.assert_not_awaited()
    # Played is flipped by the synthesized audio's final mark, not up front.
    assert tm.tools["input"].is_welcome_message_played is False


async def test_live_welcome_that_sends_no_audio_is_marked_played_after_the_timeout():
    tm = task_manager("Hi Asha, this is Sonia.", None)

    await run_forced_first_message(tm)

    tm._synthesize.assert_awaited_once()
    assert tm.tools["input"].is_welcome_message_played is True


async def test_empty_welcome_is_marked_played_without_synthesis():
    tm = task_manager("", None)

    await run_forced_first_message(tm)

    tm._synthesize.assert_not_awaited()
    tm.conversation_history.append_welcome_message.assert_not_called()
    assert tm.tools["input"].is_welcome_message_played is True


async def test_preloaded_welcome_audio_is_sent_as_before():
    tm = task_manager("Hi Asha, this is Sonia.", b"\x01\x02" * 400)

    await run_forced_first_message(tm)

    tm._synthesize.assert_not_awaited()
    tm.tools["output"].handle.assert_awaited_once()
    sent = tm.tools["output"].handle.await_args.args[0]
    assert sent["data"] == b"\x01\x02" * 400
    tm.conversation_history.append_welcome_message.assert_called_once_with("Hi Asha, this is Sonia.")
