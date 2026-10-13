"""Sarvam TTS: the streaming models open with a bare WAV header, and every chunk after it is
raw PCM that only resamples if the model's source rate is known."""

import struct
from unittest.mock import MagicMock

import asyncio

import numpy as np
import pytest

from bolna.synthesizer.sarvam_synthesizer import SarvamSynthesizer

# Named explicitly so retiring a generation from the constant cannot silently skip these.
STREAMING_MODELS = ["bulbul:v3", "bulbul:v4-flash"]

KEY = "sarvam_sk_test_dummy"
NATIVE_RATE = 22050
TELEPHONY_RATE = 8000


def _synth(model, **kwargs):
    kwargs.setdefault("voice_id", "shubh")
    kwargs.setdefault("language", "hi-IN")
    kwargs.setdefault("sampling_rate", str(TELEPHONY_RATE))
    kwargs.setdefault("stream", True)
    return SarvamSynthesizer(model=model, synthesizer_key=KEY, task_manager_instance=MagicMock(), **kwargs)


def _wav_header(rate=NATIVE_RATE, data_bytes=0):
    """The 44-byte RIFF header Sarvam sends as its first streaming chunk."""
    return (
        b"RIFF"
        + struct.pack("<I", 36 + data_bytes)
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
        + b"data"
        + struct.pack("<I", data_bytes)
    )


def _pcm(samples):
    return b"\x10\x00" * samples


@pytest.mark.parametrize("model", STREAMING_MODELS)
def test_the_opening_header_is_swallowed_and_sets_the_source_rate(model):
    synth = _synth(model)
    assert synth._process_audio_data(_wav_header()) is None
    assert synth.original_sampling_rate == NATIVE_RATE


@pytest.mark.parametrize("model", STREAMING_MODELS)
def test_audio_after_the_header_reaches_the_caller_downsampled(model):
    synth = _synth(model)
    synth._process_audio_data(_wav_header())

    out = synth._process_audio_data(_pcm(NATIVE_RATE))

    assert out, f"{model} produced no audio, the agent would be silent"
    assert len(out) == pytest.approx(TELEPHONY_RATE * 2, rel=0.01)


@pytest.mark.parametrize("model", STREAMING_MODELS)
def test_a_header_rate_that_contradicts_the_model_wins(model):
    synth = _synth(model)
    synth._process_audio_data(_wav_header(rate=24000))
    assert synth.original_sampling_rate == 24000


def _speech_like(samples, rate=NATIVE_RATE):
    t = np.arange(samples) / rate
    return (6000 * np.sin(2 * np.pi * 220 * t) + 2500 * np.sin(2 * np.pi * 1500 * t)).astype(np.int16)


@pytest.mark.parametrize("model", STREAMING_MODELS)
def test_ws_chunks_join_without_a_seam(model):
    """Each WS chunk used to resample alone, which clicked at every chunk edge. Chunk by
    chunk must now equal converting the whole utterance in one pass."""
    synth = _synth(model)
    synth._process_audio_chunk(_wav_header())
    pcm = _speech_like(NATIVE_RATE)
    chunks = [pcm[i : i + 1881] for i in range(0, len(pcm), 1881)]
    streamed = np.frombuffer(b"".join(synth._process_audio_chunk(c.tobytes()) or b"" for c in chunks), np.int16)
    whole = np.frombuffer(synth._process_audio_data(pcm.tobytes()), np.int16)
    assert len(whole) - len(streamed) <= 12  # only the held-back tail
    assert np.abs(streamed.astype(int) - whole[: len(streamed)].astype(int)).max() <= 1


@pytest.mark.parametrize("model", STREAMING_MODELS)
def test_a_new_header_rate_rebuilds_the_stream(model):
    synth = _synth(model)
    synth._process_audio_chunk(_wav_header(rate=NATIVE_RATE))
    synth._process_audio_chunk(_pcm(2000))
    synth._process_audio_chunk(_wav_header(rate=24000))
    synth._process_audio_chunk(_pcm(2400))
    assert synth.stream_resampler_rate == 24000 and synth.stream_resampler.down == 3


@pytest.mark.parametrize("model", STREAMING_MODELS)
def test_interruption_drops_the_held_back_tail(model):
    synth = _synth(model)
    synth._process_audio_chunk(_wav_header())
    synth._process_audio_chunk(_pcm(4000))
    assert len(synth.stream_resampler.history) > 0
    asyncio.run(synth.handle_interruption())
    assert len(synth.stream_resampler.history) == 0 and synth.stream_resampler.next_output == 0


def test_http_audio_converts_in_one_pass_outside_the_ws_stream():
    synth = _synth("bulbul:v3")
    synth._process_audio_data(_wav_header())
    out = synth._process_http_audio(_pcm(NATIVE_RATE))
    assert len(out) == pytest.approx(TELEPHONY_RATE * 2, rel=0.01)
    assert synth.stream_resampler is None
