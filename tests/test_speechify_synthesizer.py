"""Speechify synthesizer: attribution (Speechify-Caller slug + version on the SDK
client), output_format selection, streaming synthesis, and the mulaw/pcm telephony-clip
split. Synthesis goes through the Speechify SDK's streaming endpoint
(client.audio.stream), so this synthesizer is HTTP-only unlike ElevenlabsSynthesizer."""

from unittest.mock import MagicMock

import pytest

from bolna.enums import SynthesizerProvider
from bolna.providers import SUPPORTED_SYNTHESIZER_MODELS
from bolna.synthesizer.speechify_synthesizer import CALLER, CALLER_VERSION, SpeechifySynthesizer


@pytest.fixture(autouse=True)
def sdk(monkeypatch):
    """Stub AsyncSpeechify so no test touches the network. Returns the constructor
    mock (to assert client config) and the client mock (to stub audio.stream)."""
    client = MagicMock()
    ctor = MagicMock(return_value=client)
    monkeypatch.setattr("bolna.synthesizer.speechify_synthesizer.AsyncSpeechify", ctor)
    return {"ctor": ctor, "client": client}


def _make_synth(**overrides):
    kwargs = {"voice_id": "voice-1", "synthesizer_key": "test-key", "caching": False}
    kwargs.update(overrides)
    return SpeechifySynthesizer(**kwargs)


async def _agen(chunks):
    for chunk in chunks:
        yield chunk


def _set_stream(sdk, chunks=(b"\x01\x02\x03",), raise_exc=None):
    def _stream(**kwargs):
        if raise_exc is not None:
            raise raise_exc
        return _agen(chunks)

    sdk["client"].audio.stream = MagicMock(side_effect=_stream)


# --- registration / config ---

def test_registered_under_speechify_provider():
    assert SynthesizerProvider.SPEECHIFY.value == "speechify"
    assert SUPPORTED_SYNTHESIZER_MODELS[SynthesizerProvider.SPEECHIFY.value] is SpeechifySynthesizer


def test_default_model_is_simba_3_2():
    assert _make_synth().model == "simba-3.2"


def test_api_key_prefers_explicit_synthesizer_key():
    assert _make_synth(synthesizer_key="explicit-key").api_key == "explicit-key"


def test_api_key_falls_back_to_env_var(monkeypatch):
    monkeypatch.setenv("SPEECHIFY_API_KEY", "env-key")
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
    assert _make_synth(audio_format="pcm", sampling_rate="16000").wire_output_format == "pcm_16000"
    assert _make_synth(audio_format="pcm", sampling_rate="12345").wire_output_format == "pcm_24000"


# --- attribution: caller slug + version pinned on the SDK client, not the SDK's own ---

def test_client_configured_with_caller_slug_and_version(sdk):
    _make_synth()
    args, kwargs = sdk["ctor"].call_args
    assert args[0] == "2026-09-13"  # Speechify-Version API pin
    assert kwargs["token"] == "test-key"
    assert kwargs["headers"]["Speechify-Caller"] == CALLER == "bolna"
    assert kwargs["headers"]["Speechify-Caller-Version"] == CALLER_VERSION


# --- streaming synthesis ---

@pytest.mark.asyncio
async def test_generate_http_streams_and_concatenates(sdk):
    synth = _make_synth(audio_format="mulaw")
    _set_stream(sdk, chunks=(b"\x01", b"\x02\x03"))
    audio = await synth._generate_http("hello there")
    assert audio == b"\x01\x02\x03"
    _, kwargs = sdk["client"].audio.stream.call_args
    assert kwargs["voice_id"] == "voice-1"
    assert kwargs["model"] == "simba-3.2"
    assert kwargs["output_format"] == "ulaw_8000"


@pytest.mark.asyncio
async def test_generate_http_returns_none_on_error(sdk):
    synth = _make_synth()
    _set_stream(sdk, raise_exc=RuntimeError("boom"))
    assert await synth._generate_http("hello") is None


@pytest.mark.asyncio
async def test_synthesize_telephony_clip_only_for_mulaw(sdk):
    _set_stream(sdk)
    assert await _make_synth(audio_format="mulaw").synthesize_telephony_clip("hi") == b"\x01\x02\x03"
    assert await _make_synth(audio_format="pcm").synthesize_telephony_clip("hi") is None


# --- audio post-processing ---

def test_process_http_audio_passthrough_for_mulaw():
    synth = _make_synth(audio_format="mulaw")
    raw = b"\x01\x02\x03\x04"
    assert synth._process_http_audio(raw) is raw


def test_process_http_audio_resamples_pcm(monkeypatch):
    synth = _make_synth(audio_format="pcm", sampling_rate="16000")
    synth.pcm_wire_rate = 24000
    mock_resample = MagicMock(return_value=b"resampled")
    monkeypatch.setattr("bolna.synthesizer.speechify_synthesizer.resample", mock_resample)
    result = synth._process_http_audio(b"raw-pcm")
    mock_resample.assert_called_once_with(b"raw-pcm", 16000, format="pcm", original_sample_rate=24000)
    assert result == b"resampled"
