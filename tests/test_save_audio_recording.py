"""Mixing a recorded call into a stereo WAV must not depend on torch/torchaudio.

save_audio_file_to_s3 used to reference torch/torchaudio, which are not declared
dependencies and are never imported, so every call with recording enabled raised
NameError. These tests exercise the mix end to end with pydub only.
"""

import io
import wave

import pytest

import bolna.helpers.utils as utils
from bolna.helpers.utils import save_audio_file_to_s3


def _wav_bytes(duration_ms, sample_rate=24000, freq_byte=b"\x10\x00"):
    """A tiny mono 16-bit WAV of the requested duration."""
    n_frames = int(sample_rate * duration_ms / 1000)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(freq_byte * n_frames)
    return buf.getvalue()


@pytest.fixture
def captured_upload(monkeypatch):
    """Capture what save_audio_file_to_s3 hands to store_file instead of hitting S3."""
    captured = {}

    async def fake_store_file(bucket_name=None, file_key=None, file_data=None, content_type=None, **kwargs):
        captured["file_key"] = file_key
        captured["file_data"] = file_data
        captured["content_type"] = content_type

    monkeypatch.setattr(utils, "store_file", fake_store_file)
    monkeypatch.setattr(utils, "RECORDING_BUCKET_URL", "https://recordings.example/")
    return captured


async def test_recording_is_mixed_into_a_stereo_wav(captured_upload):
    recording = {
        "metadata": {"started": 0},
        "input": {"data": _wav_bytes(500)},
        "output": [
            {"data": _wav_bytes(200), "start_time": 0.0, "duration": 0.2},
            {"data": _wav_bytes(200), "start_time": 0.5, "duration": 0.2},
        ],
    }

    url = await save_audio_file_to_s3(recording, sampling_rate=24000, assistant_id="agent", run_id="run")

    assert url == "https://recordings.example/agentrun.wav"
    assert captured_upload["content_type"] == "wav"

    with wave.open(io.BytesIO(captured_upload["file_data"]), "rb") as w:
        # Caller on one channel, agent on the other.
        assert w.getnchannels() == 2
        assert w.getframerate() == 24000
        assert w.getnframes() > 0


async def test_recording_survives_when_no_agent_audio_was_captured(captured_upload):
    """An empty output list must not IndexError; the caller channel is still saved."""
    recording = {
        "metadata": {"started": 0},
        "input": {"data": _wav_bytes(300)},
        "output": [],
    }

    url = await save_audio_file_to_s3(recording, sampling_rate=24000, assistant_id="a", run_id="r")

    assert url.endswith("ar.wav")
    with wave.open(io.BytesIO(captured_upload["file_data"]), "rb") as w:
        assert w.getnchannels() == 2
