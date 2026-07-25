"""Speechify HTTP synthesizer: request shape (auth, Speechify-Caller attribution,
output_format selection) and the mulaw/pcm telephony-clip split. Speechify's public
API is plain HTTP (POST /v1/audio/stream, chunked-bytes response) rather than a
duplex WebSocket, so this synthesizer is HTTP-only unlike ElevenlabsSynthesizer."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bolna.enums import SynthesizerProvider
from bolna.providers import SUPPORTED_SYNTHESIZER_MODELS
from bolna.synthesizer.speechify_synthesizer import CALLER_HEADER_NAME, CALLER_HEADER_VALUE, SpeechifySynthesizer


def _make_synth(**overrides):
    kwargs = {"voice_id": "voice-1", "synthesizer_key": "test-key", "caching": False}
    kwargs.update(overrides)
    return SpeechifySynthesizer(**kwargs)


def test_registered_under_speechify_provider():
    assert SynthesizerProvider.SPEECHIFY.value == "speechify"
    assert SUPPORTED_SYNTHESIZER_MODELS[SynthesizerProvider.SPEECHIFY.value] is SpeechifySynthesizer


def test_default_model_is_simba_3_2():
    assert _make_synth().model == "simba-3.2"


def test_api_key_prefers_explicit_synthesizer_key():
    synth = _make_synth(synthesizer_key="explicit-key")
    assert synth.api_key == "explicit-key"


def test_api_key_falls_back_to_env_var():
    with patch.dict(os.environ, {"SPEECHIFY_API_KEY": "env-key"}, clear=False):
        synth = SpeechifySynthesizer(voice_id="voice-1", caching=False)
    assert synth.api_key == "env-key"


def test_declares_no_websocket_support():
    assert _make_synth().supports_websocket() is False


def test_mulaw_config_selects_ulaw_8000_wire_format():
    synth = _make_synth(audio_format="mulaw")
    assert synth.use_mulaw is True
    assert synth.wire_output_format == "ulaw_8000"
    assert synth._get_http_audio_format() == "mulaw"


def test_pcm_config_selects_supported_rate_and_falls_back_to_24000():
    synth = _make_synth(audio_format="pcm", sampling_rate="16000")
    assert synth.wire_output_format == "pcm_16000"

    synth_odd_rate = _make_synth(audio_format="pcm", sampling_rate="12345")
    assert synth_odd_rate.wire_output_format == "pcm_24000"


def _mock_post_response(status=200, body=b"\x01\x02\x03"):
    response = MagicMock()
    response.status = status
    response.read = AsyncMock(return_value=body)
    response.text = AsyncMock(return_value="error detail")

    session = MagicMock()
    post_ctx = MagicMock()
    post_ctx.__aenter__ = AsyncMock(return_value=response)
    post_ctx.__aexit__ = AsyncMock(return_value=False)
    session.post = MagicMock(return_value=post_ctx)

    session_ctx = MagicMock()
    session_ctx.__aenter__ = AsyncMock(return_value=session)
    session_ctx.__aexit__ = AsyncMock(return_value=False)
    return session_ctx, session


@pytest.mark.asyncio
async def test_generate_http_sends_caller_header_and_bearer_auth():
    synth = _make_synth(audio_format="mulaw")
    session_ctx, session = _mock_post_response()

    with patch("bolna.synthesizer.speechify_synthesizer.aiohttp.ClientSession", return_value=session_ctx):
        audio = await synth._generate_http("hello there")

    assert audio == b"\x01\x02\x03"
    _, kwargs = session.post.call_args
    assert kwargs["headers"]["Authorization"] == "Bearer test-key"
    assert kwargs["headers"][CALLER_HEADER_NAME] == CALLER_HEADER_VALUE
    assert kwargs["json"]["voice_id"] == "voice-1"
    assert kwargs["json"]["model"] == "simba-3.2"
    assert kwargs["json"]["output_format"] == "ulaw_8000"


@pytest.mark.asyncio
async def test_generate_http_returns_none_on_non_200():
    synth = _make_synth()
    session_ctx, _ = _mock_post_response(status=500)

    with patch("bolna.synthesizer.speechify_synthesizer.aiohttp.ClientSession", return_value=session_ctx):
        audio = await synth._generate_http("hello")

    assert audio is None


@pytest.mark.asyncio
async def test_synthesize_telephony_clip_only_for_mulaw():
    mulaw_synth = _make_synth(audio_format="mulaw")
    session_ctx, _ = _mock_post_response()
    with patch("bolna.synthesizer.speechify_synthesizer.aiohttp.ClientSession", return_value=session_ctx):
        assert await mulaw_synth.synthesize_telephony_clip("hi") == b"\x01\x02\x03"

    pcm_synth = _make_synth(audio_format="pcm")
    assert await pcm_synth.synthesize_telephony_clip("hi") is None


def test_process_http_audio_passthrough_for_mulaw():
    synth = _make_synth(audio_format="mulaw")
    raw = b"\x01\x02\x03\x04"
    assert synth._process_http_audio(raw) is raw


def test_process_http_audio_resamples_pcm():
    synth = _make_synth(audio_format="pcm", sampling_rate="16000")
    synth.pcm_wire_rate = 24000
    with patch("bolna.synthesizer.speechify_synthesizer.resample", return_value=b"resampled") as mock_resample:
        result = synth._process_http_audio(b"raw-pcm")
    mock_resample.assert_called_once_with(b"raw-pcm", 16000, format="pcm", original_sample_rate=24000)
    assert result == b"resampled"
