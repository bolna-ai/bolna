"""AssemblyAI session parameters and agent context carryover."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock
from urllib.parse import parse_qs, urlparse

import pytest
from pydantic import ValidationError

from bolna.agent_manager.task_manager import TaskManager
from bolna.models import AssemblyAITranscriberConfig
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


async def _connect(transcriber, monkeypatch):
    monkeypatch.setattr(
        "bolna.transcriber.assemblyai_transcriber.websockets.connect", AsyncMock(return_value=MagicMock())
    )
    await transcriber.assemblyai_connect()


async def _sent_for_audio(transcriber, frames=1):
    """What the sender puts on the socket for `frames` audio packets."""
    ws = MagicMock(send=AsyncMock())
    for _ in range(frames):
        transcriber.input_queue.put_nowait({"data": b"\xff" * 160, "meta_info": {}})
    transcriber.input_queue.put_nowait({"data": None, "meta_info": {"eos": True}})
    await transcriber.sender_stream(ws)
    return [json.loads(m) if isinstance(m, str) else m for m in (call.args[0] for call in ws.send.await_args_list)]


def test_universal_3_sends_only_the_session_params_that_are_set():
    params = _params(_make_transcriber(assemblyai_config={**dict.fromkeys(SESSION_CONFIG), "max_turn_silence": 2000}))
    assert params["max_turn_silence"] == "2000"
    assert not (set(SESSION_CONFIG) - {"max_turn_silence"}) & params.keys()

    params = _params(_make_transcriber(assemblyai_config=SESSION_CONFIG))
    assert {name: params[name] for name in SESSION_CONFIG} == {k: str(v) for k, v in SESSION_CONFIG.items()}


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


def _reply_source(transcriber, reply=None):
    """Give the transcriber a mutable "latest spoken reply", as conversation history would."""
    latest = {"text": reply}
    transcriber.set_agent_context_source(lambda: latest["text"])
    return latest


async def test_session_gets_prompting_once_and_each_reply_once_ahead_of_the_audio(monkeypatch):
    transcriber = _make_transcriber(keywords="Vireo, Kestrel", context="A caller booking a test drive.")
    reply = _reply_source(transcriber, "  Which car would you like to drive?  ")
    assert not {"keyterms_prompt", "prompt", "agent_context"} & _params(transcriber).keys()

    await _connect(transcriber, monkeypatch)
    first = await _sent_for_audio(transcriber, frames=2)
    assert first[0] == {
        "type": "UpdateConfiguration",
        "keyterms_prompt": ["Vireo", "Kestrel"],
        "prompt": "A caller booking a test drive.",
        "agent_context": "Which car would you like to drive?",
    }
    assert all(isinstance(message, bytes) for message in first[1:3])

    reply["text"] = "And which day suits you?"
    assert (await _sent_for_audio(transcriber))[0] == {
        "type": "UpdateConfiguration",
        "agent_context": "And which day suits you?",
    }

    await _connect(transcriber, monkeypatch)
    assert set((await _sent_for_audio(transcriber))[0]) == {"type", "keyterms_prompt", "prompt", "agent_context"}


async def test_context_follows_a_barge_in_trim_and_clears_when_nothing_was_heard(monkeypatch):
    transcriber = _make_transcriber()
    reply = _reply_source(transcriber, "Your total is 500. Pay by UPI or card?")
    await _connect(transcriber, monkeypatch)
    await _sent_for_audio(transcriber)

    reply["text"] = "Your total"
    assert (await _sent_for_audio(transcriber))[0] == {"type": "UpdateConfiguration", "agent_context": "Your total"}

    reply["text"] = None
    assert (await _sent_for_audio(transcriber))[0] == {"type": "UpdateConfiguration", "agent_context": ""}


@pytest.mark.parametrize("model", ["universal", "universal-streaming-multilingual", "whisper-rt"])
async def test_other_models_get_keyterms_but_no_universal_3_only_settings(model, monkeypatch):
    transcriber = _make_transcriber(
        model=model, keywords="Aster Insure", context="Insurance call.", assemblyai_config=SESSION_CONFIG
    )
    _reply_source(transcriber, "Which insurer is your policy with?")
    params = _params(transcriber)
    assert (params["min_turn_silence"], params["max_turn_silence"], params["vad_threshold"]) == ("200", "1500", "0.4")
    assert not {"mode", "interruption_delay", "voice_focus", "voice_focus_threshold"} & params.keys()

    await _connect(transcriber, monkeypatch)
    assert (await _sent_for_audio(transcriber))[0] == {
        "type": "UpdateConfiguration",
        "keyterms_prompt": ["Aster Insure"],
    }


def test_missing_model_does_not_break_construction():
    assert not _make_transcriber(model=None).is_universal_3_model


def test_non_latin_prompting_stays_out_of_the_connect_url():
    hindi = "क्या आप अपना पिनकोड बता सकते हैं? " * 60
    transcriber = _make_transcriber(language="hi", keywords="पिनकोड, आधार", context=hindi)
    _reply_source(transcriber, hindi)
    assert len(transcriber.get_assemblyai_ws_url()) < 500


def test_long_reply_keeps_its_closing_question_within_the_limit():
    transcriber = _make_transcriber()
    _reply_source(transcriber, "x" * 2000 + " What's your email address?")
    context = transcriber._agent_context()
    assert len(context) == 1750
    assert context.endswith("What's your email address?")


async def test_task_manager_points_the_transcriber_at_the_conversation(monkeypatch):
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

    tm.conversation_history.setup_system_prompt({"role": "system", "content": "You book test drives."})
    tm.conversation_history.append_welcome_message("Hi {name}, how can I help?")
    tm.conversation_history.update_welcome_message("Hi Asha, how can I help?")
    assert transcriber._agent_context() == "Hi Asha, how can I help?"
    tm.conversation_history.append_assistant("Can you spell your last name?", turn_id=1)
    assert transcriber._agent_context() == "Can you spell your last name?"


def test_pool_points_every_leg_at_the_conversation():
    legs = {"en": MagicMock(), "hi": MagicMock()}
    pool = TranscriberPool.__new__(TranscriberPool)
    pool.transcribers = legs
    source = MagicMock()
    pool.set_agent_context_source(source)
    for leg in legs.values():
        leg.set_agent_context_source.assert_called_once_with(source)
