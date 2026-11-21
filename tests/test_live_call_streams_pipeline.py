"""Live calls always run the streaming pipeline; turn-based chats stream only when enforced."""

from unittest.mock import MagicMock

import pytest

import bolna.agent_manager.task_manager as tmmod
from bolna.agent_manager.task_manager import TaskManager, streams_pipeline

SARVAM_HTTP = {
    "provider": "sarvam",
    "provider_config": {"voice": "Suhani", "voice_id": "suhani", "model": "bulbul:v3"},
    "stream": False,
}


def test_live_calls_always_stream():
    assert streams_pipeline(turn_based_conversation=False, enforce_streaming=False)
    assert streams_pipeline(turn_based_conversation=False, enforce_streaming=True)


def test_turn_based_chats_stream_only_when_enforced():
    assert streams_pipeline(turn_based_conversation=True, enforce_streaming=True)
    assert not streams_pipeline(turn_based_conversation=True, enforce_streaming=False)


def _live_call_synth_kwargs(monkeypatch, synthesizer):
    built = {}
    monkeypatch.setattr(tmmod, "SUPPORTED_SYNTHESIZER_MODELS", {"sarvam": lambda **kwargs: built.update(kwargs)})
    monkeypatch.setattr(tmmod, "SynthesizerPool", MagicMock())

    tm = MagicMock()
    tm.turn_based_conversation = False
    tm.enforce_streaming = False
    tm.is_web_based_call = False
    tm.language_switcher = None
    tm.kwargs = {}
    tm.provider_api_keys = {}
    tm._TaskManager__pool_leg_kwargs = TaskManager._TaskManager__pool_leg_kwargs.__get__(tm, TaskManager)
    tm.task_config = {
        "tools_config": {
            "transcriber": {"language": "hi"},
            "synthesizer": synthesizer,
            "output": {"provider": "plivo"},
            "llm_agent": None,
        }
    }
    TaskManager._TaskManager__setup_synthesizer.__get__(tm, TaskManager)()
    return built


@pytest.mark.parametrize(
    "synthesizer",
    [dict(SARVAM_HTTP), {"provider": "sarvam", "multilingual": {"hi": dict(SARVAM_HTTP)}}],
    ids=["single", "pool_leg"],
)
def test_live_call_streams_the_synthesizer_even_when_its_config_says_not_to(monkeypatch, synthesizer):
    assert _live_call_synth_kwargs(monkeypatch, synthesizer)["stream"] is True
