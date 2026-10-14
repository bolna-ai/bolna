"""ElevenLabs is told the call's language, and only on models that accept language_code."""

from unittest.mock import MagicMock

import pytest

import bolna.agent_manager.task_manager as tmmod
import bolna.synthesizer.elevenlabs_synthesizer as elmod
from bolna.agent_manager.task_manager import TaskManager
from bolna.synthesizer.elevenlabs_synthesizer import ElevenlabsSynthesizer, elevenlabs_language_code


@pytest.mark.parametrize(
    "model, language, expected",
    [
        ("eleven_flash_v2_5", "hi", "hi"),
        ("eleven_turbo_v2_5", "hi-IN", "hi"),
        ("eleven_turbo_v2_5", "multi-hi", "hi"),
        ("eleven_turbo_v2_5", "en-IN", "en"),
        ("eleven_flash_v2_5", "ta", "ta"),
        ("eleven_multilingual_v2", "hi", None),
        ("eleven_flash_v2_5", "unknown", None),
        ("eleven_flash_v2_5", "mr", None),
        ("eleven_flash_v2_5", None, None),
    ],
)
def test_language_code_only_for_supported_models_and_languages(model, language, expected):
    assert elevenlabs_language_code(model, language) == expected


def _synth(model, language):
    return ElevenlabsSynthesizer(
        voice="voice", voice_id="vid", model=model, synthesizer_key="test", caching=False, language=language
    )


def test_socket_url_carries_the_language():
    assert _synth("eleven_flash_v2_5", "hi").ws_url.endswith("&language_code=hi")


def test_socket_url_omits_the_language_when_the_model_cannot_take_it():
    assert "language_code" not in _synth("eleven_multilingual_v2", "hi").ws_url
    assert "language_code" not in _synth("eleven_flash_v2_5", None).ws_url


class _FakeResponse:
    status = 200

    async def read(self):
        return b"audio"

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeSession:
    def __init__(self, payloads):
        self.payloads = payloads

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def post(self, url, headers=None, json=None):
        self.payloads.append(json)
        return _FakeResponse()


@pytest.mark.parametrize("language, expected", [("hi", "hi"), (None, None)])
async def test_http_render_carries_the_language(monkeypatch, language, expected):
    payloads = []
    monkeypatch.setattr(elmod.aiohttp, "ClientSession", lambda: _FakeSession(payloads))

    await _synth("eleven_turbo_v2_5", language).synthesize("नमस्ते")

    assert payloads[0].get("language_code") == expected


def _setup_synthesizer(monkeypatch, tools_config):
    calls = {}

    def recorder(provider):
        def factory(*args, **kwargs):
            calls.setdefault(provider, []).append(kwargs)
            return MagicMock()

        return factory

    monkeypatch.setattr(
        tmmod, "SUPPORTED_SYNTHESIZER_MODELS", {p: recorder(p) for p in ("elevenlabs", "sarvam", "cartesia")}
    )
    monkeypatch.setattr(tmmod, "SynthesizerPool", MagicMock())
    tm = MagicMock()
    tm.turn_based_conversation = True
    tm.is_web_based_call = False
    tm.enforce_streaming = True
    tm.kwargs = {}
    tm.provider_api_keys = None
    tm._TaskManager__pool_leg_kwargs = TaskManager._TaskManager__pool_leg_kwargs.__get__(tm, TaskManager)
    tm.task_config = {"tools_config": {"output": {"provider": "plivo"}, "llm_agent": None, **tools_config}}
    TaskManager._TaskManager__setup_synthesizer.__get__(tm, TaskManager)()
    return calls


def test_single_synthesizer_speaks_the_agent_language(monkeypatch):
    calls = _setup_synthesizer(
        monkeypatch,
        {
            "transcriber": {"language": "hi"},
            "synthesizer": {"provider": "elevenlabs", "provider_config": {"voice": "voice", "voice_id": "v1"}},
        },
    )

    assert calls["elevenlabs"][0]["language"] == "hi"


def test_single_synthesizer_keeps_its_configured_language(monkeypatch):
    calls = _setup_synthesizer(
        monkeypatch,
        {
            "transcriber": {"language": "en"},
            "synthesizer": {
                "provider": "elevenlabs",
                "provider_config": {"voice": "voice", "voice_id": "v1", "language": "hi"},
            },
        },
    )

    assert calls["elevenlabs"][0]["language"] == "hi"


def test_each_pool_leg_speaks_its_own_language(monkeypatch):
    # Legs are copies of the base voice, so a stale language from the base must not leak into another leg.
    leg = {"provider": "elevenlabs", "provider_config": {"voice": "voice", "voice_id": "v1", "language": "hi"}}
    calls = _setup_synthesizer(
        monkeypatch,
        {
            "transcriber": {"language": "hi"},
            "synthesizer": {"provider": "elevenlabs", "multilingual": {"hi": dict(leg), "en": dict(leg)}},
        },
    )

    assert sorted(kwargs["language"] for kwargs in calls["elevenlabs"]) == ["en", "hi"]


def test_other_providers_keep_their_own_language(monkeypatch):
    calls = _setup_synthesizer(
        monkeypatch,
        {
            "transcriber": {"language": "hi"},
            "synthesizer": {"provider": "cartesia", "provider_config": {"voice": "voice", "voice_id": "v2"}},
        },
    )

    assert "language" not in calls["cartesia"][0]
