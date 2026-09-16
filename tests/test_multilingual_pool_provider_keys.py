"""Each pool leg authenticates with its own provider's key, not the base provider's (BOLNA-2697)."""

from unittest.mock import MagicMock

import bolna.agent_manager.task_manager as tmmod
from bolna.agent_manager.task_manager import TaskManager


def _recorder(calls, provider):
    def factory(*args, **kwargs):
        calls[provider] = kwargs
        return MagicMock()

    return factory


def _mock_task_manager(kwargs, provider_api_keys):
    tm = MagicMock()
    tm.turn_based_conversation = True
    tm.is_web_based_call = False
    tm.enforce_streaming = True
    tm.kwargs = kwargs
    tm.provider_api_keys = provider_api_keys
    tm._TaskManager__pool_leg_kwargs = TaskManager._TaskManager__pool_leg_kwargs.__get__(tm, TaskManager)
    return tm


def _setup_transcribers(monkeypatch, multilingual, kwargs, provider_api_keys):
    calls = {}
    monkeypatch.setattr(
        tmmod,
        "SUPPORTED_TRANSCRIBER_PROVIDERS",
        {p: _recorder(calls, p) for p in ("deepgram", "sarvam", "azure")},
    )
    monkeypatch.setattr(tmmod, "TranscriberPool", MagicMock())

    tm = _mock_task_manager(kwargs, provider_api_keys)
    tm.task_config = {"tools_config": {"transcriber": {"provider": "deepgram", "multilingual": multilingual}}}
    tm._TaskManager__language_switch_enabled = MagicMock(return_value=False)
    TaskManager._TaskManager__setup_transcriber.__get__(tm, TaskManager)()
    return calls


def _setup_synthesizers(monkeypatch, multilingual, kwargs, provider_api_keys):
    calls = {}
    monkeypatch.setattr(
        tmmod,
        "SUPPORTED_SYNTHESIZER_MODELS",
        {p: _recorder(calls, p) for p in ("elevenlabs", "cartesia", "sarvam")},
    )
    monkeypatch.setattr(tmmod, "SynthesizerPool", MagicMock())

    tm = _mock_task_manager(kwargs, provider_api_keys)
    tm.task_config = {
        "tools_config": {
            "transcriber": {"language": "en"},
            "synthesizer": {"provider": "elevenlabs", "multilingual": multilingual},
            "output": {"provider": "plivo"},
            "llm_agent": None,
        }
    }
    TaskManager._TaskManager__setup_synthesizer.__get__(tm, TaskManager)()
    return calls


def test_transcriber_leg_does_not_inherit_the_base_provider_key(monkeypatch):
    # The Kannada leg used to receive the customer's Deepgram key and get a 403 from Sarvam.
    calls = _setup_transcribers(
        monkeypatch,
        {
            "en": {"provider": "deepgram", "model": "nova-3"},
            "kn": {"provider": "sarvam", "model": "saaras:v3"},
        },
        kwargs={"transcriber_key": "deepgram-byok"},
        provider_api_keys={"deepgram": "deepgram-byok", "sarvam": "sarvam-byok"},
    )

    assert calls["deepgram"]["transcriber_key"] == "deepgram-byok"
    assert calls["sarvam"]["transcriber_key"] == "sarvam-byok"


def test_transcriber_leg_without_a_stored_key_falls_back_to_env(monkeypatch):
    calls = _setup_transcribers(
        monkeypatch,
        {
            "en": {"provider": "deepgram", "model": "nova-3"},
            "kn": {"provider": "sarvam", "model": "saaras:v3"},
        },
        kwargs={"transcriber_key": "deepgram-byok"},
        provider_api_keys={"deepgram": "deepgram-byok"},
    )

    assert calls["deepgram"]["transcriber_key"] == "deepgram-byok"
    assert "transcriber_key" not in calls["sarvam"]


def test_transcriber_legs_keep_the_base_key_when_no_map_is_sent(monkeypatch):
    # Version skew: an older caller sends no map, and dropping the key here would silently move a
    # BYOK customer onto the platform's own credentials.
    calls = _setup_transcribers(
        monkeypatch,
        {
            "en": {"provider": "deepgram", "model": "nova-3"},
            "hi": {"provider": "deepgram", "model": "nova-2"},
        },
        kwargs={"transcriber_key": "deepgram-byok"},
        provider_api_keys={},
    )

    assert calls["deepgram"]["transcriber_key"] == "deepgram-byok"


def test_transcriber_leg_identified_only_by_model_keeps_the_base_key(monkeypatch):
    calls = _setup_transcribers(
        monkeypatch,
        {"en": {"provider": "deepgram", "model": "nova-3"}, "hi": {"model": "nova-2"}},
        kwargs={"transcriber_key": "deepgram-byok"},
        provider_api_keys={"deepgram": "deepgram-byok"},
    )

    assert calls["deepgram"]["transcriber_key"] == "deepgram-byok"


def test_synthesizer_leg_does_not_inherit_the_base_provider_key(monkeypatch):
    calls = _setup_synthesizers(
        monkeypatch,
        {
            "en": {"provider": "elevenlabs", "provider_config": {"voice": "Anika", "voice_id": "v1"}},
            "kn": {"provider": "cartesia", "provider_config": {"voice": "Kavya", "voice_id": "v2"}},
        },
        kwargs={"synthesizer_key": "elevenlabs-byok"},
        provider_api_keys={"elevenlabs": "elevenlabs-byok", "cartesia": "cartesia-byok"},
    )

    assert calls["elevenlabs"]["synthesizer_key"] == "elevenlabs-byok"
    assert calls["cartesia"]["synthesizer_key"] == "cartesia-byok"


def test_synthesizer_leg_without_a_stored_key_falls_back_to_env(monkeypatch):
    calls = _setup_synthesizers(
        monkeypatch,
        {
            "en": {"provider": "elevenlabs", "provider_config": {"voice": "Anika", "voice_id": "v1"}},
            "kn": {"provider": "cartesia", "provider_config": {"voice": "Kavya", "voice_id": "v2"}},
        },
        kwargs={"synthesizer_key": "elevenlabs-byok"},
        provider_api_keys={"elevenlabs": "elevenlabs-byok"},
    )

    assert calls["elevenlabs"]["synthesizer_key"] == "elevenlabs-byok"
    assert "synthesizer_key" not in calls["cartesia"]


async def test_provider_key_map_is_captured_and_kept_out_of_the_kwargs_splat(monkeypatch):
    # self.kwargs is splatted into every provider constructor, so the map must not linger there.
    monkeypatch.setenv("ELEVENLABS_API_KEY", "env-key")
    task_config = {
        "task_type": "conversation",
        "toolchain": {"execution": "sequential", "pipelines": [["llm"]]},
        "tools_config": {
            "llm_agent": {
                "agent_type": "simple_llm_agent",
                "agent_flow_type": "streaming",
                "llm_config": {
                    "model": "gpt-5.4-mini",
                    "provider": "openai",
                    "max_tokens": 150,
                    "temperature": 1,
                },
            },
            "synthesizer": {
                "provider": "elevenlabs",
                "provider_config": {"voice": "Nila", "voice_id": "test", "model": "eleven_turbo_v2_5"},
                "stream": True,
                "buffer_size": 100,
            },
            "transcriber": {
                "provider": "deepgram",
                "model": "nova-3",
                "language": "en",
                "stream": True,
                "encoding": "linear16",
                "sampling_rate": 16000,
                "endpointing": 250,
            },
            "input": {"provider": "default"},
            "output": {"provider": "default"},
        },
        "task_config": {},
    }

    tm = TaskManager("agent", 0, task_config, MagicMock(), provider_api_keys={"sarvam": "sarvam-byok"})

    assert tm.provider_api_keys == {"sarvam": "sarvam-byok"}
    assert "provider_api_keys" not in tm.kwargs
