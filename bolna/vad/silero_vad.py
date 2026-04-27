from __future__ import annotations

from typing import Iterator

import numpy as np
import torch
from silero_vad import VADIterator, load_silero_vad

from .turn_detector import SpeechEvent, SpeechEventType


# Silero requires fixed chunk sizes per sample rate.
SUPPORTED_RATES: dict[int, int] = {8000: 256, 16000: 512}


class SileroVAD:
    def __init__(
        self,
        sample_rate: int = 8000,
        threshold: float = 0.5,
        min_silence_duration_ms: int = 100,
        speech_pad_ms: int = 30,
    ) -> None:
        if sample_rate not in SUPPORTED_RATES:
            raise ValueError(
                f"sample_rate must be one of {sorted(SUPPORTED_RATES)}; got {sample_rate}"
            )
        self.sample_rate = sample_rate
        self.chunk_samples = SUPPORTED_RATES[sample_rate]
        self.chunk_bytes = self.chunk_samples * 2

        self._model = load_silero_vad(onnx=True)
        self._iter = VADIterator(
            self._model,
            threshold=threshold,
            sampling_rate=sample_rate,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
        )
        self._byte_buffer = bytearray()
        self._samples_consumed = 0

    def reset(self) -> None:
        self._iter.reset_states()
        self._byte_buffer.clear()
        self._samples_consumed = 0

    def feed(self, pcm_int16_bytes: bytes) -> Iterator[SpeechEvent]:
        if not pcm_int16_bytes:
            return
        self._byte_buffer.extend(pcm_int16_bytes)

        while len(self._byte_buffer) >= self.chunk_bytes:
            chunk_bytes = bytes(self._byte_buffer[: self.chunk_bytes])
            del self._byte_buffer[: self.chunk_bytes]

            samples_int16 = np.frombuffer(chunk_bytes, dtype=np.int16)
            samples_f32 = samples_int16.astype(np.float32) / 32768.0
            tensor = torch.from_numpy(samples_f32)

            event = self._iter(tensor)
            self._samples_consumed += self.chunk_samples

            if event is None:
                continue

            if "start" in event:
                yield SpeechEvent(
                    type=SpeechEventType.SPEECH_STARTED,
                    sample_offset=int(event["start"]),
                )
            elif "end" in event:
                yield SpeechEvent(
                    type=SpeechEventType.SPEECH_ENDED,
                    sample_offset=int(event["end"]),
                )

    @property
    def samples_consumed(self) -> int:
        return self._samples_consumed
