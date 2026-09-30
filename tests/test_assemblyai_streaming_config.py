import asyncio
import json
from unittest.mock import AsyncMock, MagicMock
from urllib.parse import parse_qs, urlparse

import pytest
from pydantic import ValidationError

from bolna.agent_manager.task_manager import TaskManager
from bolna.models import AssemblyAIStreamingConfig, Transcriber
from bolna.transcriber.assemblyai_transcriber import AssemblyAITranscriber
from bolna.transcriber.base_transcriber import BaseTranscriber
from bolna.transcriber.transcriber_pool import TranscriberPool

FULL_CONFIG = {
    "mode": "balanced",
    "min_turn_silence": 200,
    "max_turn_silence": 1500,
    "interruption_delay": 300,
    "vad_threshold": 0.4,
    "voice_focus": "near-field",
    "voice_focus_threshold": 0.6,
}


def make_transcriber(model="universal-3-6-pro", **kwargs):
    return AssemblyAITranscriber("twilio", model=model, language="en", **kwargs)


def query(transcriber):
    return {key: values[0] for key, values in parse_qs(urlparse(transcriber.get_assemblyai_ws_url()).query).items()}


def test_universal_3_sends_every_configured_streaming_param():
    params = query(make_transcriber(assemblyai_config=FULL_CONFIG))
    assert {name: params[name] for name in FULL_CONFIG} == {name: str(value) for name, value in FULL_CONFIG.items()}
    assert params["speech_model"] == "universal-3-6-pro"
    assert params["sample_rate"] == "8000"


def test_unset_params_are_left_to_assemblyai_defaults():
    config = AssemblyAIStreamingConfig(min_turn_silence=300).model_dump()
    params = query(make_transcriber(assemblyai_config=config))
    assert params["min_turn_silence"] == "300"
    assert not {"mode", "max_turn_silence", "interruption_delay", "vad_threshold", "voice_focus"} & params.keys()


def test_no_config_sends_no_streaming_params():
    params = query(make_transcriber())
    assert not set(FULL_CONFIG) & params.keys()


def test_legacy_model_keeps_turn_params_and_drops_universal_3_only_ones():
    params = query(make_transcriber(model="universal", assemblyai_config=FULL_CONFIG))
    assert params["speech_model"] == "universal-streaming-english"
    assert (params["min_turn_silence"], params["max_turn_silence"], params["vad_threshold"]) == ("200", "1500", "0.4")
    assert not {"mode", "interruption_delay", "voice_focus", "voice_focus_threshold"} & params.keys()


def test_transcriber_model_validates_assemblyai_config():
    transcriber = Transcriber(provider="assembly", model="universal-3-6-pro", assemblyai_config=FULL_CONFIG)
    assert transcriber.assemblyai_config.voice_focus == "near-field"
    assert Transcriber(provider="assembly").assemblyai_config is None
    with pytest.raises(ValidationError):
        AssemblyAIStreamingConfig(voice_focus_threshold=0.5)
    with pytest.raises(ValidationError):
        AssemblyAIStreamingConfig(interruption_delay=1001)
    with pytest.raises(ValidationError):
        AssemblyAIStreamingConfig(mode="fast")
    with pytest.raises(ValidationError):
        AssemblyAIStreamingConfig(voice_focus="headset")


def test_agent_context_spoken_before_connecting_seeds_the_session():
    transcriber = make_transcriber()
    transcriber.set_agent_context("  Hi, this is Bolna. How can I help?  ")
    assert query(transcriber)["agent_context"] == "Hi, this is Bolna. How can I help?"


def test_agent_context_keeps_the_end_of_a_long_reply():
    transcriber = make_transcriber()
    transcriber.set_agent_context("x" * 2000 + " What's your email address?")
    assert len(transcriber.agent_context) == 1750
    assert transcriber.agent_context.endswith("What's your email address?")


@pytest.mark.parametrize("model", ["universal-3-5-pro", "universal"])
def test_agent_context_is_only_sent_to_models_that_accept_it(model):
    transcriber = make_transcriber(model=model)
    transcriber.websocket_connection = MagicMock(send=AsyncMock())
    transcriber.set_agent_context("What's your email address?")
    assert "agent_context" not in query(transcriber)
    assert not transcriber.configuration_update_tasks


async def test_agent_reply_mid_call_sends_update_configuration_once():
    transcriber = make_transcriber()
    transcriber.websocket_connection = MagicMock(send=AsyncMock())
    transcriber.set_agent_context("What's your email address?")
    transcriber.set_agent_context("What's your email address?")
    await asyncio.gather(*transcriber.configuration_update_tasks)
    transcriber.websocket_connection.send.assert_awaited_once_with(
        json.dumps({"type": "UpdateConfiguration", "agent_context": "What's your email address?"})
    )


async def test_failed_update_configuration_does_not_raise():
    transcriber = make_transcriber()
    transcriber.websocket_connection = MagicMock(send=AsyncMock(side_effect=ConnectionError("closed")))
    transcriber.set_agent_context("Anything else?")
    await asyncio.gather(*transcriber.configuration_update_tasks)


def test_other_transcribers_ignore_agent_context():
    BaseTranscriber().set_agent_context("Anything else?")


def test_pool_shares_agent_context_with_every_leg():
    legs = {"en": MagicMock(), "hi": MagicMock()}
    pool = TranscriberPool.__new__(TranscriberPool)
    pool.transcribers = legs
    pool.set_agent_context("Anything else?")
    for leg in legs.values():
        leg.set_agent_context.assert_called_once_with("Anything else?")


def test_committed_assistant_reply_is_shared_with_the_transcriber():
    tm = TaskManager.__new__(TaskManager)
    tm.tools = {"transcriber": MagicMock()}
    tm._committed_assistant_sequences = set()
    tm._pending_assistant_history = {
        7: {"content": "What's your email address?", "turn_id": 2, "response_uid": "r1", "user_input": None}
    }
    tm._turn_msg_map = {}
    tm.conversation_history = MagicMock(messages=[{}])
    tm._commit_staged_assistant_history(7)
    tm._commit_staged_assistant_history(7)
    tm.tools["transcriber"].set_agent_context.assert_called_once_with("What's your email address?")


def test_text_chat_without_a_transcriber_skips_agent_context():
    tm = TaskManager.__new__(TaskManager)
    tm.tools = {}
    tm._share_agent_reply_with_transcriber("Hello")
