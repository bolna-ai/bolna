from __future__ import annotations

import wave
from pathlib import Path

import pytest

from bolna.vad import SileroVAD, SpeechEventType

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "benchmarks" / "fixtures"
LEADING_SPEECH = FIXTURE_DIR / "leading_speech.wav"
PURE_SILENCE = FIXTURE_DIR / "pure_silence.wav"


def _read_pcm_16k_int16(wav_path: Path) -> bytes:
    with wave.open(str(wav_path), "rb") as w:
        assert w.getframerate() == 16000, "fixture must be 16kHz"
        assert w.getsampwidth() == 2, "fixture must be 16-bit"
        assert w.getnchannels() == 1, "fixture must be mono"
        return w.readframes(w.getnframes())


def test_rejects_unsupported_sample_rate() -> None:
    with pytest.raises(ValueError):
        SileroVAD(sample_rate=22050)


def test_chunk_size_bytes_matches_sample_rate() -> None:
    assert SileroVAD(sample_rate=8000).chunk_bytes == 256 * 2
    assert SileroVAD(sample_rate=16000).chunk_bytes == 512 * 2


def test_speech_clip_yields_speech_started() -> None:
    if not LEADING_SPEECH.exists():
        pytest.skip(f"missing fixture: {LEADING_SPEECH}")
    pcm = _read_pcm_16k_int16(LEADING_SPEECH)

    vad = SileroVAD(sample_rate=16000, threshold=0.5, min_silence_duration_ms=100)
    events = list(vad.feed(pcm))

    start_events = [e for e in events if e.type is SpeechEventType.SPEECH_STARTED]
    assert len(start_events) >= 1, "expected at least one speech_started event"

    # Fixture puts speech onset at 1500ms. With speech_pad_ms=30 default
    # silero may report a few tens of ms early; allow a generous window.
    first_start_ms = start_events[0].sample_offset * 1000 / 16000
    assert 1200 <= first_start_ms <= 1800, (
        f"first speech_started reported at {first_start_ms}ms, expected ~1500ms"
    )


def test_digital_silence_yields_no_events() -> None:
    if not PURE_SILENCE.exists():
        pytest.skip(f"missing fixture: {PURE_SILENCE}")
    pcm = _read_pcm_16k_int16(PURE_SILENCE)

    vad = SileroVAD(sample_rate=16000, threshold=0.5, min_silence_duration_ms=100)
    events = list(vad.feed(pcm))

    assert events == [], f"expected no events on digital silence, got {events}"


def test_buffers_partial_chunks_until_full() -> None:
    """Feeding fewer than chunk_bytes should simply buffer, not fire any VAD calls."""
    vad = SileroVAD(sample_rate=16000)

    # Feed a single byte; the model should not run since chunk is incomplete.
    events = list(vad.feed(b"\x00"))
    assert events == []
    assert vad.samples_consumed == 0


def test_reset_clears_internal_state() -> None:
    vad = SileroVAD(sample_rate=16000)
    # feed() is a generator; exhaust it so the VAD actually processes the chunk.
    list(vad.feed(b"\x00" * vad.chunk_bytes))
    assert vad.samples_consumed == vad.chunk_samples
    vad.reset()
    assert vad.samples_consumed == 0
