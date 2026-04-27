from __future__ import annotations

import pytest

from bolna.vad import PreSpeechRingBuffer


def test_rejects_non_positive_capacity() -> None:
    with pytest.raises(ValueError):
        PreSpeechRingBuffer(capacity_bytes=0)
    with pytest.raises(ValueError):
        PreSpeechRingBuffer(capacity_bytes=-5)


def test_from_duration_computes_capacity() -> None:
    # 500ms at 8kHz mulaw (1 byte/sample) -> 4000 bytes
    rb = PreSpeechRingBuffer.from_duration(
        duration_ms=500, sample_rate_hz=8000, bytes_per_sample=1
    )
    assert rb.capacity_bytes == 4000

    # 500ms at 16kHz int16 (2 bytes/sample) -> 16000 bytes
    rb = PreSpeechRingBuffer.from_duration(
        duration_ms=500, sample_rate_hz=16000, bytes_per_sample=2
    )
    assert rb.capacity_bytes == 16000


def test_append_below_capacity_keeps_everything() -> None:
    rb = PreSpeechRingBuffer(capacity_bytes=100)
    rb.append(b"a" * 30)
    rb.append(b"b" * 40)
    assert len(rb) == 70
    assert rb.flush() == b"a" * 30 + b"b" * 40


def test_append_trims_oldest_bytes_on_overflow() -> None:
    rb = PreSpeechRingBuffer(capacity_bytes=10)
    rb.append(b"aaaa")      # size=4
    rb.append(b"bbbb")      # size=8
    rb.append(b"cccccc")    # size=14 -> trim 4 -> keeps last 10: "aaabbbbcccccc"[-10:]
    assert len(rb) == 10
    assert rb.flush() == b"aabbbbcccccc"[-10:]


def test_append_can_trim_mid_chunk() -> None:
    rb = PreSpeechRingBuffer(capacity_bytes=5)
    rb.append(b"abcdef")    # overflows immediately; should keep last 5
    assert rb.flush() == b"bcdef"


def test_flush_empties_buffer() -> None:
    rb = PreSpeechRingBuffer(capacity_bytes=50)
    rb.append(b"hello")
    assert rb.flush() == b"hello"
    assert len(rb) == 0
    assert rb.flush() == b""


def test_append_empty_bytes_is_noop() -> None:
    rb = PreSpeechRingBuffer(capacity_bytes=10)
    rb.append(b"")
    assert len(rb) == 0
    assert not rb.is_full


def test_is_full_flag() -> None:
    rb = PreSpeechRingBuffer(capacity_bytes=8)
    rb.append(b"1234")
    assert not rb.is_full
    rb.append(b"5678")
    assert rb.is_full
    rb.append(b"9")  # one byte spills; still full
    assert rb.is_full
    assert rb.flush() == b"23456789"


def test_chronological_order_preserved_across_overflow() -> None:
    """After many overflows the buffer still yields bytes in arrival order."""
    rb = PreSpeechRingBuffer(capacity_bytes=6)
    for chunk in [b"12", b"34", b"56", b"78", b"90"]:
        rb.append(chunk)
    # Overall stream was "1234567890"; capacity 6 -> last 6 = "567890"
    assert rb.flush() == b"567890"
