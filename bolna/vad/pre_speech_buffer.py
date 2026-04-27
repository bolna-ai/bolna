from __future__ import annotations

from collections import deque


class PreSpeechRingBuffer:
    def __init__(self, capacity_bytes: int) -> None:
        if capacity_bytes <= 0:
            raise ValueError("capacity_bytes must be positive")
        self.capacity_bytes = capacity_bytes
        self._chunks: deque[bytes] = deque()
        self._size = 0

    @classmethod
    def from_duration(
        cls, duration_ms: int, sample_rate_hz: int, bytes_per_sample: int
    ) -> "PreSpeechRingBuffer":
        capacity = duration_ms * sample_rate_hz * bytes_per_sample // 1000
        return cls(capacity_bytes=capacity)

    def append(self, audio_bytes: bytes) -> None:
        if not audio_bytes:
            return
        self._chunks.append(audio_bytes)
        self._size += len(audio_bytes)

        while self._size > self.capacity_bytes and self._chunks:
            head = self._chunks[0]
            overflow = self._size - self.capacity_bytes
            if len(head) <= overflow:
                self._chunks.popleft()
                self._size -= len(head)
            else:
                self._chunks[0] = head[overflow:]
                self._size -= overflow

    def flush(self) -> bytes:
        if not self._chunks:
            return b""
        out = b"".join(self._chunks)
        self._chunks.clear()
        self._size = 0
        return out

    def __len__(self) -> int:
        return self._size

    @property
    def is_full(self) -> bool:
        return self._size >= self.capacity_bytes
