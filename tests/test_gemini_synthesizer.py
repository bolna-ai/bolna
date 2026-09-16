"""Gemini TTS: the generateContent payload, inlineData parsing, 24 kHz -> telephony
conversion, and the one-shot clip paths. No network, no ADC."""

import asyncio
import audioop
import base64
from unittest.mock import AsyncMock, MagicMock

import pytest

from bolna.models import Synthesizer
from bolna.providers import SUPPORTED_SYNTHESIZER_MODELS
from bolna.synthesizer.gemini_synthesizer import GeminiSynthesizer


def _synth(**kwargs):
    kwargs.setdefault("voice", "Puck")
    kwargs.setdefault("voice_id", "Puck")
    kwargs.setdefault("use_mulaw", True)
    kwargs.setdefault("caching", False)
    kwargs.setdefault("project", "test-project")
    s = GeminiSynthesizer(task_manager_instance=MagicMock(), **kwargs)
    s.task_manager_instance.is_sequence_id_in_current_ids.return_value = True
    return s


def _pcm(seconds, rate=24000):
    """A silent-but-nonzero 16-bit mono buffer of a known length."""
    return (b"\x10\x00") * int(seconds * rate)


def _response(pcm, mime="audio/L16;codec=pcm;rate=24000"):
    return {
        "candidates": [
            {"content": {"parts": [{"inlineData": {"mimeType": mime, "data": base64.b64encode(pcm).decode()}}]}}
        ]
    }


# ----------------------------------------------------------------------
# Construction
# ----------------------------------------------------------------------


def test_voice_id_is_the_gemini_voice_name_and_wins_over_the_label():
    assert _synth(voice="Label", voice_id="Kore").voice == "Kore"
    assert GeminiSynthesizer(voice="Puck", voice_id=None, project="p").voice == "Puck"


def test_telephony_pins_8k_mulaw_and_web_keeps_native_24k():
    tel = _synth(use_mulaw=True)
    assert (tel.target_sample_rate, tel._get_http_audio_format()) == (8000, "mulaw")

    web = _synth(use_mulaw=False, sampling_rate="24000")
    assert (web.target_sample_rate, web._get_http_audio_format()) == (24000, "pcm")


def test_endpoint_carries_project_location_and_model():
    s = _synth(model="gemini-tts-x")
    assert s._endpoint("proj-123") == (
        "https://aiplatform.googleapis.com/v1beta1/projects/proj-123/locations/global"
        "/publishers/google/models/gemini-tts-x:generateContent"
    )


def test_voice_id_survives_the_config_model_and_reaches_the_synthesizer():
    cfg = Synthesizer(
        provider="gemini",
        provider_config={"voice_id": "Kore", "voice": "Kore", "model": "gemini-tts", "language": "en"},
        stream=False,
    ).model_dump()
    cfg.pop("caching", None)
    cfg.pop("provider")
    provider_config = cfg.pop("provider_config")
    assert provider_config["voice_id"] == "Kore"

    s = SUPPORTED_SYNTHESIZER_MODELS["gemini"](
        **cfg, **provider_config, caching=False, project="p", task_manager_instance=MagicMock()
    )
    assert s.voice == "Kore"


# ----------------------------------------------------------------------
# Payload
# ----------------------------------------------------------------------


def test_payload_puts_the_voice_in_prebuilt_voice_config():
    payload = _synth(voice_id="Puck")._build_payload("Hello there.")
    # role is mandatory: the endpoint 400s without it.
    assert payload["contents"][0]["role"] == "user"
    assert payload["contents"][0]["parts"][0] == {"text": "Hello there."}
    voice = payload["generation_config"]["speech_config"]["voice_config"]["prebuilt_voice_config"]
    assert voice == {"voice_name": "Puck"}
    assert payload["generation_config"]["response_modalities"] == ["AUDIO"]


def test_style_goes_into_structured_speech_metadata_not_the_transcript():
    # The model speaks inline direction verbatim; style must ride in metadata instead.
    part = _synth(style="calm and reassuring")._build_payload("Welcome to the flight deck.")["contents"][0]["parts"][0]
    assert part == {"text": "Welcome to the flight deck.", "speech_metadata": {"style": "calm and reassuring"}}


def test_no_style_means_no_speech_metadata_key():
    part = _synth(style=None)._build_payload("Hi <breath> there!")["contents"][0]["parts"][0]
    assert part == {"text": "Hi <breath> there!"}  # inline tags stay in the text


# ----------------------------------------------------------------------
# Response parsing
# ----------------------------------------------------------------------


def test_extract_audio_decodes_base64_pcm_and_reads_the_rate_from_the_mime_type():
    pcm, rate = GeminiSynthesizer._extract_audio(_response(b"\x01\x02\x03\x04", mime="audio/L16;codec=pcm;rate=16000"))
    assert pcm == b"\x01\x02\x03\x04"
    assert rate == 16000


def test_extract_audio_falls_back_to_native_rate_when_the_mime_type_omits_it():
    _, rate = GeminiSynthesizer._extract_audio(_response(b"\x01\x02", mime="audio/pcm"))
    assert rate == 24000


def test_extract_audio_accepts_snake_case_inline_data():
    resp = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"inline_data": {"mime_type": "rate=24000", "data": base64.b64encode(b"\x05\x06").decode()}}
                    ]
                }
            }
        ]
    }
    pcm, rate = GeminiSynthesizer._extract_audio(resp)
    assert (pcm, rate) == (b"\x05\x06", 24000)


def test_extract_audio_returns_none_when_there_is_no_audio_part():
    assert GeminiSynthesizer._extract_audio({"candidates": []}) == (None, None)
    assert GeminiSynthesizer._extract_audio({"candidates": [{"content": {"parts": [{"text": "oops"}]}}]}) == (
        None,
        None,
    )


# ----------------------------------------------------------------------
# Audio conversion
# ----------------------------------------------------------------------


def test_telephony_audio_is_resampled_to_8k_and_mulaw_encoded():
    s = _synth(use_mulaw=True)
    out = s._to_target(_pcm(1.0), 24000)
    # 24k 16-bit in -> 8k 8-bit out: one sixth of the bytes, and decodable as mu-law.
    assert len(out) == 8000
    assert len(audioop.ulaw2lin(out, 2)) == 16000


def test_web_audio_stays_pcm_at_the_target_rate():
    s = _synth(use_mulaw=False, sampling_rate="24000")
    chunk = _pcm(0.5)
    assert s._to_target(chunk, 24000) == chunk


def test_empty_audio_is_dropped_rather_than_converted():
    assert _synth()._to_target(None, 24000) is None
    assert _synth()._to_target(b"", 24000) is None


# ----------------------------------------------------------------------
# Turn render + one-shot
# ----------------------------------------------------------------------


def test_generate_http_converts_the_fetched_pcm_to_the_target_format():
    s = _synth(use_mulaw=True)
    s._fetch_pcm = AsyncMock(return_value=(_pcm(1.0), 24000))
    assert len(asyncio.run(s._generate_http("hi"))) == 8000


def test_synthesize_wraps_pcm_in_wav_so_the_rate_is_self_describing():
    from bolna.helpers.utils import audio_to_mulaw8k

    s = _synth(use_mulaw=True)
    s._fetch_pcm = AsyncMock(return_value=(_pcm(1.0), 24000))
    out = asyncio.run(s.synthesize("hi"))
    assert out[:4] == b"RIFF"
    # One second in stays one second out (8000 mu-law bytes) even with a wrong hint.
    assert len(audio_to_mulaw8k(out, rate_hint=8000, format_hint="")) == 8000


def test_synthesize_returns_none_when_the_api_returns_nothing():
    s = _synth()
    s._fetch_pcm = AsyncMock(return_value=(None, None))
    assert asyncio.run(s.synthesize("hi")) is None


def test_fetch_pcm_bails_when_no_project_resolves(monkeypatch):
    """No configured project and ADC resolves none: bail before posting a malformed URL."""
    s = GeminiSynthesizer(voice="Puck", project=None, caching=False, task_manager_instance=MagicMock())
    s.project = None
    monkeypatch.setattr(
        "bolna.synthesizer.gemini_synthesizer.get_gcp_credentials", AsyncMock(return_value=("tok", None))
    )
    assert asyncio.run(s._fetch_pcm("hi")) == (None, None)


def test_fetch_pcm_bails_when_credentials_cannot_be_obtained(monkeypatch):
    """A stalled or failing token mint ends the turn cleanly instead of raising into the loop."""
    s = _synth(project="p")
    monkeypatch.setattr(
        "bolna.synthesizer.gemini_synthesizer.get_gcp_credentials", AsyncMock(side_effect=asyncio.TimeoutError)
    )
    assert asyncio.run(s._fetch_pcm("hi")) == (None, None)
