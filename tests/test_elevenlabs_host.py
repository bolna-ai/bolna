"""The ElevenLabs host a caller passes for one call wins over the host this deployment is configured for."""

import asyncio

import pytest

from bolna.providers import SUPPORTED_SYNTHESIZER_MODELS
from bolna.transcriber.elevenlabs_transcriber import ElevenLabsTranscriber

CONFIGURED_HOST = "api.in.residency.elevenlabs.io"
CALL_HOST = "api.elevenlabs.io"


@pytest.fixture(autouse=True)
def configured_host(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_HOST", CONFIGURED_HOST)


def _synthesizer(**kwargs):
    return SUPPORTED_SYNTHESIZER_MODELS["elevenlabs"](
        voice="George",
        voice_id="JBFqnCBsd6RMkjVDRZzb",
        model="eleven_turbo_v2_5",
        caching=False,
        synthesizer_key="k",
        **kwargs,
    )


def _transcriber_url(**kwargs):
    return ElevenLabsTranscriber(
        telephony_provider="plivo",
        input_queue=asyncio.Queue(),
        output_queue=asyncio.Queue(),
        transcriber_key="k",
        **kwargs,
    ).get_elevenlabs_ws_url()


def test_a_per_call_host_wins_over_the_configured_one():
    synth = _synthesizer(elevenlabs_host=CALL_HOST)
    assert synth.ws_url.startswith(f"wss://{CALL_HOST}/")
    assert synth.api_url.startswith(f"https://{CALL_HOST}/")
    assert _transcriber_url(elevenlabs_host=CALL_HOST).startswith(f"wss://{CALL_HOST}/")


def test_without_a_per_call_host_the_configured_one_is_used():
    assert _synthesizer().ws_url.startswith(f"wss://{CONFIGURED_HOST}/")
    assert _transcriber_url().startswith(f"wss://{CONFIGURED_HOST}/")
