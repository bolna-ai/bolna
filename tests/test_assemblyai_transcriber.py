"""AssemblyAI session parameters and agent context carryover."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock
from urllib.parse import parse_qs, urlparse

import pytest
from pydantic import ValidationError

from bolna.agent_manager.task_manager import TaskManager
from bolna.enums import ChatRole
from bolna.helpers.conversation_history import ConversationHistory
from bolna.models import AssemblyAITranscriberConfig, Transcriber
from bolna.transcriber.assemblyai_transcriber import AssemblyAITranscriber
from bolna.transcriber.transcriber_pool import TranscriberPool

SESSION_CONFIG = {
    "mode": "balanced",
    "min_turn_silence": 200,
    "max_turn_silence": 1500,
    "interruption_delay": 300,
    "vad_threshold": 0.4,
    "voice_focus": "near-field",
    "voice_focus_threshold": 0.6,
}


def _make_transcriber(model="universal-3-6-pro", language="en", **kwargs):
    return AssemblyAITranscriber("twilio", model=model, language=language, input_queue=asyncio.Queue(), **kwargs)


def _params(transcriber):
    return {key: values[0] for key, values in parse_qs(urlparse(transcriber.get_assemblyai_ws_url()).query).items()}


async def _run_sender(transcriber, *packets):
    ws = MagicMock(send=AsyncMock())
    for packet in packets:
        transcriber.input_queue.put_nowait(packet)
    transcriber.input_queue.put_nowait({"data": None, "meta_info": {"eos": True}})
    await transcriber.sender_stream(ws)
    return [call.args[0] for call in ws.send.await_args_list]


def _audio():
    return {"data": b"\xff" * 160, "meta_info": {}}


def test_universal_3_sends_every_configured_session_param():
    params = _params(_make_transcriber(assemblyai_config=SESSION_CONFIG))
    assert {name: params[name] for name in SESSION_CONFIG} == {k: str(v) for k, v in SESSION_CONFIG.items()}


def test_unset_session_params_are_left_to_assemblyai_defaults():
    config = AssemblyAITranscriberConfig(max_turn_silence=2000).model_dump()
    params = _params(_make_transcriber(assemblyai_config=config))
    assert params["max_turn_silence"] == "2000"
    assert not (set(SESSION_CONFIG) - {"max_turn_silence"}) & params.keys()


def test_legacy_model_drops_universal_3_only_params():
    params = _params(_make_transcriber(model="universal", assemblyai_config=SESSION_CONFIG))
    assert (params["min_turn_silence"], params["max_turn_silence"], params["vad_threshold"]) == ("200", "1500", "0.4")
    assert not {"mode", "interruption_delay", "voice_focus", "voice_focus_threshold"} & params.keys()


def test_transcriber_carries_validated_assemblyai_config():
    transcriber = Transcriber(provider="assembly", model="universal-3-6-pro", assemblyai_config=SESSION_CONFIG)
    assert transcriber.model_dump()["assemblyai_config"] == SESSION_CONFIG
    assert Transcriber(provider="assembly").assemblyai_config is None


@pytest.mark.parametrize(
    "config",
    [
        {"mode": "fast"},
        {"interruption_delay": 1001},
        {"vad_threshold": 1.5},
        {"min_turn_silence": -1},
        {"voice_focus": "headset"},
        {"voice_focus_threshold": 0.6},
    ],
)
def test_invalid_assemblyai_config_is_rejected(config):
    with pytest.raises(ValidationError):
        AssemblyAITranscriberConfig(**config)


async def _connect(transcriber, monkeypatch):
    monkeypatch.setattr(
        "bolna.transcriber.assemblyai_transcriber.websockets.connect", AsyncMock(return_value=MagicMock())
    )
    await transcriber.assemblyai_connect()


async def test_reply_spoken_before_connecting_goes_out_ahead_of_the_first_audio(monkeypatch):
    transcriber = _make_transcriber()
    transcriber.set_agent_context("  Hi, this is Bolna. Can I get your order number?  ")
    assert "agent_context" not in _params(transcriber)
    await _connect(transcriber, monkeypatch)
    sent = await _run_sender(transcriber, _audio())
    assert json.loads(sent[0]) == {
        "type": "UpdateConfiguration",
        "agent_context": "Hi, this is Bolna. Can I get your order number?",
    }
    assert isinstance(sent[1], bytes)


async def test_each_new_reply_goes_out_once_ahead_of_the_next_audio(monkeypatch):
    transcriber = _make_transcriber()
    await _connect(transcriber, monkeypatch)
    transcriber.set_agent_context("What's your email address?")
    sent = await _run_sender(transcriber, _audio(), _audio())
    assert json.loads(sent[0]) == {"type": "UpdateConfiguration", "agent_context": "What's your email address?"}
    assert all(isinstance(message, bytes) for message in sent[1:3])

    transcriber.set_agent_context("And your date of birth?")
    sent = await _run_sender(transcriber, _audio())
    assert json.loads(sent[0]) == {"type": "UpdateConfiguration", "agent_context": "And your date of birth?"}


async def test_reconnected_session_receives_the_current_reply_again(monkeypatch):
    transcriber = _make_transcriber()
    transcriber.set_agent_context("What's your email address?")
    await _connect(transcriber, monkeypatch)
    await _run_sender(transcriber, _audio())

    await _connect(transcriber, monkeypatch)
    sent = await _run_sender(transcriber, _audio())
    assert json.loads(sent[0]) == {"type": "UpdateConfiguration", "agent_context": "What's your email address?"}


async def test_prompting_goes_out_once_per_session_ahead_of_the_first_audio(monkeypatch):
    transcriber = _make_transcriber(keywords="Saanvi Iyer, Kia Syros", context="The caller is booking a test drive.")
    assert not {"keyterms_prompt", "prompt"} & _params(transcriber).keys()
    transcriber.set_agent_context("Which car would you like to drive?")
    await _connect(transcriber, monkeypatch)
    sent = await _run_sender(transcriber, _audio(), _audio())
    assert json.loads(sent[0]) == {
        "type": "UpdateConfiguration",
        "keyterms_prompt": ["Saanvi Iyer", "Kia Syros"],
        "prompt": "The caller is booking a test drive.",
        "agent_context": "Which car would you like to drive?",
    }
    assert all(isinstance(message, bytes) for message in sent[1:3])

    transcriber.set_agent_context("And which day suits you?")
    sent = await _run_sender(transcriber, _audio())
    assert json.loads(sent[0]) == {"type": "UpdateConfiguration", "agent_context": "And which day suits you?"}

    await _connect(transcriber, monkeypatch)
    sent = await _run_sender(transcriber, _audio())
    assert set(json.loads(sent[0])) == {"type", "keyterms_prompt", "prompt", "agent_context"}


async def test_legacy_model_gets_keyterms_but_no_prompt(monkeypatch):
    transcriber = _make_transcriber(model="universal", keywords="Bajaj Allianz", context="Insurance call.")
    await _connect(transcriber, monkeypatch)
    sent = await _run_sender(transcriber, _audio())
    assert json.loads(sent[0]) == {"type": "UpdateConfiguration", "keyterms_prompt": ["Bajaj Allianz"]}


def test_non_latin_prompting_stays_out_of_the_connect_url():
    hindi = "क्या आप अपना पिनकोड बता सकते हैं? " * 60
    transcriber = _make_transcriber(language="hi", keywords="पिनकोड, आधार", context=hindi)
    transcriber.set_agent_context(hindi)
    assert len(transcriber.get_assemblyai_ws_url()) < 500


def test_long_reply_keeps_its_closing_question_within_the_limit():
    transcriber = _make_transcriber()
    transcriber.set_agent_context("x" * 2000 + " What's your email address?")
    assert len(transcriber.agent_context) == 1750
    assert transcriber.agent_context.endswith("What's your email address?")


async def test_legacy_model_never_receives_agent_context(monkeypatch):
    transcriber = _make_transcriber(model="universal")
    transcriber.set_agent_context("What's your email address?")
    await _connect(transcriber, monkeypatch)
    sent = await _run_sender(transcriber, _audio())
    assert not any("UpdateConfiguration" in str(message) for message in sent)


def test_history_reports_each_spoken_assistant_message():
    spoken = []
    history = ConversationHistory(on_assistant_message=spoken.append)
    history.append_welcome_message("Hi, how can I help?")
    history.append_user("I want to change my address")
    history.append_assistant("Sure, what's the new pincode?", turn_id=1)
    history.append_assistant("Sure, what's the new pincode?", turn_id=1)
    history.append_assistant(None, tool_calls=[{"id": "call_1"}])
    assert spoken == ["Hi, how can I help?", "Sure, what's the new pincode?"]


def test_history_without_a_listener_appends_as_before():
    history = ConversationHistory()
    history.append_assistant("Anything else?")
    assert history.messages[-1] == {"role": ChatRole.ASSISTANT, "content": "Anything else?"}


async def test_task_manager_wires_config_and_spoken_replies_into_the_transcriber(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "env-key")
    task_config = {
        "task_type": "conversation",
        "toolchain": {"execution": "sequential", "pipelines": [["transcriber", "llm", "synthesizer"]]},
        "tools_config": {
            "llm_agent": {
                "agent_type": "simple_llm_agent",
                "agent_flow_type": "streaming",
                "llm_config": {"model": "gpt-5.4-mini", "provider": "openai", "max_tokens": 150, "temperature": 1},
            },
            "synthesizer": {
                "provider": "elevenlabs",
                "provider_config": {"voice": "Nila", "voice_id": "test", "model": "eleven_turbo_v2_5"},
                "stream": True,
                "buffer_size": 100,
            },
            "transcriber": {
                "provider": "assembly",
                "model": "universal-3-6-pro",
                "language": "en",
                "stream": True,
                "assemblyai_config": SESSION_CONFIG,
            },
            "input": {"provider": "default"},
            "output": {"provider": "default"},
        },
        "task_config": {},
    }
    tm = TaskManager("agent", 0, task_config, MagicMock())
    transcriber = tm.tools["transcriber"]
    assert isinstance(transcriber, AssemblyAITranscriber)
    assert transcriber.session_params == SESSION_CONFIG

    tm.conversation_history.append_assistant("Can you spell your last name?", turn_id=1)
    assert transcriber.agent_context == "Can you spell your last name?"


def test_text_conversation_without_a_transcriber_ignores_spoken_replies():
    tm = TaskManager.__new__(TaskManager)
    tm.tools = {}
    tm._share_agent_reply_with_transcriber("Anything else?")


def test_pool_shares_agent_context_with_every_leg():
    legs = {"en": MagicMock(), "hi": MagicMock()}
    pool = TranscriberPool.__new__(TranscriberPool)
    pool.transcribers = legs
    pool.set_agent_context("Anything else?")
    for leg in legs.values():
        leg.set_agent_context.assert_called_once_with("Anything else?")
