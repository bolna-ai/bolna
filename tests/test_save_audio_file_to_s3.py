"""Stereo call recording built by save_audio_file_to_s3.

The recording is uploaded at the end of every recorded call, so it has to be built from the
dependencies the package actually ships (pydub and numpy).
"""

import asyncio
import io
import wave

from pydub import AudioSegment

from bolna.helpers import utils


def _wav_bytes(duration_ms, frame_rate, amplitude):
    samples = int(frame_rate * duration_ms / 1000)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(frame_rate)
        wav_file.writeframes(amplitude.to_bytes(2, "little", signed=True) * samples)
    return buffer.getvalue()


def test_recording_is_stereo_with_caller_left_and_assistant_right(monkeypatch):
    uploads = {}

    async def fake_store_file(bucket_name, file_key, file_data, content_type):
        uploads[file_key] = file_data.read()

    monkeypatch.setattr(utils, "store_file", fake_store_file)

    recording = {
        "metadata": {"started": 100.0},
        "input": {"data": _wav_bytes(1500, 16000, 1000)},
        "output": [{"start_time": 100.0, "duration": 0.5, "data": _wav_bytes(500, 24000, -2000)}],
    }

    url = asyncio.run(utils.save_audio_file_to_s3(recording, 24000, "assistant-", "run"))

    assert url.endswith("assistant-run.wav")
    stereo = AudioSegment.from_file(io.BytesIO(uploads["assistant-run.wav"]), format="wav")
    assert stereo.channels == 2
    assert stereo.frame_rate == 24000
    # The shorter assistant channel is padded with silence up to the length of the caller channel
    assert 1490 <= len(stereo) <= 1510
    caller, assistant = stereo.split_to_mono()
    assert caller.get_array_of_samples()[100] > 0
    assert assistant.get_array_of_samples()[100] < 0
    assert assistant.get_array_of_samples()[-1] == 0
