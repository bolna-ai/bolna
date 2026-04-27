from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator

from .pre_speech_buffer import PreSpeechRingBuffer


class SpeechEventType(str, Enum):
    SPEECH_STARTED = "speech_started"
    SPEECH_ENDED = "speech_ended"


@dataclass
class SpeechEvent:
    type: SpeechEventType
    sample_offset: int
    pre_speech_audio: bytes = b""


@dataclass
class TurnDetector:
    vad: "object"
    pre_speech_buffer: PreSpeechRingBuffer
    _is_in_speech: bool = field(default=False, init=False)

    def feed(
        self,
        pcm_int16_bytes: bytes,
        raw_audio: bytes | None = None,
    ) -> Iterator[SpeechEvent]:
        # raw_audio lets the caller stash the original on-wire encoding
        # (mulaw etc) in the buffer while the VAD sees the decoded PCM.
        self.pre_speech_buffer.append(raw_audio if raw_audio is not None else pcm_int16_bytes)

        for event in self.vad.feed(pcm_int16_bytes):  # type: ignore[attr-defined]
            if event.type is SpeechEventType.SPEECH_STARTED and not self._is_in_speech:
                self._is_in_speech = True
                yield SpeechEvent(
                    type=SpeechEventType.SPEECH_STARTED,
                    sample_offset=event.sample_offset,
                    pre_speech_audio=self.pre_speech_buffer.flush(),
                )
            elif event.type is SpeechEventType.SPEECH_ENDED and self._is_in_speech:
                self._is_in_speech = False
                yield SpeechEvent(
                    type=SpeechEventType.SPEECH_ENDED,
                    sample_offset=event.sample_offset,
                )

    @property
    def is_in_speech(self) -> bool:
        return self._is_in_speech
