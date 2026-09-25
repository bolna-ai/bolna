"""Payload-level parity tests for provider speed and volume controls. No network."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from bolna.synthesizer.cartesia_synthesizer import CartesiaSynthesizer
from bolna.synthesizer.deepgram_synthesizer import DeepgramSynthesizer
from bolna.synthesizer.openai_synthesizer import OPENAISynthesizer
from bolna.synthesizer.rime_synthesizer import RimeSynthesizer
from bolna.synthesizer.sarvam_synthesizer import SarvamSynthesizer
from bolna.synthesizer.smallest_synthesizer import SmallestSynthesizer


def _task_manager():
    return MagicMock()


def test_deepgram_aura_2_speed_is_on_http_and_websocket_urls():
    synth = DeepgramSynthesizer(
        voice="Thalia",
        voice_id="thalia-en",
        model="aura-2",
        speed=1.25,
        transcriber_key="test-key",
        task_manager_instance=_task_manager(),
    )

    assert "speed=1.25" in synth.ws_url
    assert "speed=1.25" in synth._http_url()


def test_deepgram_does_not_send_aura_2_controls_to_legacy_models():
    synth = DeepgramSynthesizer(
        voice="Zeus",
        voice_id="zeus-en",
        model="aura",
        speed=1.0,
        transcriber_key="test-key",
        task_manager_instance=_task_manager(),
    )

    assert "speed=" not in synth.ws_url
    assert "speed=" not in synth._http_url()


async def test_openai_passes_speed_to_the_sdk(monkeypatch):
    response = MagicMock()
    response.iter_bytes.return_value = [b"MP3"]
    create = AsyncMock(return_value=response)
    client = MagicMock()
    client.audio.speech.create = create
    monkeypatch.setattr("bolna.synthesizer.openai_synthesizer.AsyncOpenAI", lambda **_: client)

    synth = OPENAISynthesizer(
        voice="alloy",
        model="tts-1",
        speed=1.75,
        synthesizer_key="test-key",
        task_manager_instance=_task_manager(),
    )

    assert await synth._generate_http("hello") == b"MP3"
    create.assert_awaited_once_with(
        model="tts-1",
        voice="alloy",
        response_format="mp3",
        input="hello",
        speed=1.75,
    )


def test_smallest_speed_is_on_http_and_websocket_payloads():
    synth = SmallestSynthesizer(
        voice_id="meher",
        model="lightning_v3.1",
        language="en",
        speed=1.4,
        synthesizer_key="test-key",
        task_manager_instance=_task_manager(),
    )

    assert synth.form_payload("hello")["speed"] == 1.4
    assert synth._http_payload("hello")["speed"] == 1.4


def test_rime_time_scale_is_on_coda_http_and_websocket_requests():
    synth = RimeSynthesizer(
        voice="Astra",
        voice_id="astra",
        model="coda",
        time_scale_factor=0.8,
        synthesizer_key="test-key",
        task_manager_instance=_task_manager(),
    )

    assert "timeScaleFactor=0.8" in synth.ws_url
    assert synth._http_payload("hello")["timeScaleFactor"] == 0.8


def test_rime_does_not_send_time_scale_to_unsupported_models():
    synth = RimeSynthesizer(
        voice="Legacy",
        voice_id="legacy",
        model="mistv2",
        time_scale_factor=1.0,
        synthesizer_key="test-key",
        task_manager_instance=_task_manager(),
    )

    assert "timeScaleFactor" not in synth.ws_url
    assert "timeScaleFactor" not in synth._http_payload("hello")


@pytest.mark.parametrize("model", ["sonic-3", "sonic-3.5", "sonic-3.6", "sonic-preview"])
def test_cartesia_sonic_3_series_volume_is_on_streaming_and_http_generation_config(model):
    synth = CartesiaSynthesizer(
        voice="Sonic",
        voice_id="voice-id",
        model=model,
        language="en",
        volume=1.6,
        synthesizer_key="test-key",
        task_manager_instance=_task_manager(),
    )

    assert synth.form_payload("hello")["generation_config"]["volume"] == 1.6
    assert synth._generation_config()["volume"] == 1.6


def test_cartesia_legacy_models_do_not_receive_volume():
    synth = CartesiaSynthesizer(
        voice="Legacy",
        voice_id="voice-id",
        model="sonic-english",
        language="en",
        volume=1.0,
        synthesizer_key="test-key",
        task_manager_instance=_task_manager(),
    )

    assert "volume" not in synth.form_payload("hello")["generation_config"]


def test_sarvam_v2_loudness_is_on_http_and_websocket_payloads():
    synth = SarvamSynthesizer(
        voice="Ritu",
        voice_id="ritu",
        model="bulbul:v2",
        language="hi-IN",
        loudness=1.8,
        synthesizer_key="test-key",
        task_manager_instance=_task_manager(),
    )

    assert synth._config_message()["data"]["loudness"] == 1.8
    assert synth._http_payload("hello")["loudness"] == 1.8


def test_sarvam_v2_accepts_documented_loudness_minimum():
    synth = SarvamSynthesizer(
        voice="Ritu",
        voice_id="ritu",
        model="bulbul:v2",
        language="hi-IN",
        loudness=0.3,
        synthesizer_key="test-key",
        task_manager_instance=_task_manager(),
    )

    assert synth._config_message()["data"]["loudness"] == 0.3


def test_sarvam_v3_omits_unsupported_loudness_on_both_transports():
    synth = SarvamSynthesizer(
        voice="Shubh",
        voice_id="shubh",
        model="bulbul:v3",
        language="hi-IN",
        loudness=1.0,
        synthesizer_key="test-key",
        task_manager_instance=_task_manager(),
    )

    assert "loudness" not in synth._config_message()["data"]
    assert "loudness" not in synth._http_payload("hello")


@pytest.mark.parametrize(
    ("factory", "kwargs"),
    [
        (DeepgramSynthesizer, {"voice": "T", "voice_id": "t-en", "model": "aura-2", "speed": 1.6, "transcriber_key": "k"}),
        (OPENAISynthesizer, {"voice": "alloy", "speed": 4.1, "synthesizer_key": "k"}),
        (SmallestSynthesizer, {"voice_id": "meher", "speed": 2.1, "synthesizer_key": "k"}),
        (RimeSynthesizer, {"voice": "A", "voice_id": "a", "model": "coda", "time_scale_factor": 2.6, "synthesizer_key": "k"}),
        (CartesiaSynthesizer, {"voice": "C", "voice_id": "c", "volume": 2.1, "synthesizer_key": "k"}),
        (SarvamSynthesizer, {"voice_id": "ritu", "model": "bulbul:v2", "language": "hi-IN", "loudness": 0.29, "synthesizer_key": "k"}),
        (SarvamSynthesizer, {"voice_id": "ritu", "model": "bulbul:v2", "language": "hi-IN", "loudness": 3.1, "synthesizer_key": "k"}),
    ],
)
def test_out_of_range_controls_fail_at_setup(monkeypatch, factory, kwargs):
    if factory is OPENAISynthesizer:
        monkeypatch.setattr("bolna.synthesizer.openai_synthesizer.AsyncOpenAI", MagicMock())
    with pytest.raises(ValueError):
        factory(task_manager_instance=_task_manager(), **kwargs)
