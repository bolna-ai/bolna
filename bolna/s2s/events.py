from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SessionReady:
    """Provider accepted the session config and is ready for audio."""

    connection_time_ms: float


@dataclass
class AudioDelta:
    """A chunk of output audio, PCM-16 at the provider's output_sample_rate."""

    data: bytes


@dataclass
class TranscriptDelta:
    """Assistant speech transcript."""

    content: str
    is_final: bool


@dataclass
class InputTranscript:
    """User speech transcript from the provider's built-in transcription."""

    content: str
    is_final: bool


@dataclass
class FunctionCall:
    """The provider wants to invoke a tool."""

    name: str
    call_id: str
    arguments: str  # JSON string


@dataclass
class FunctionCallCancelled:
    """Provider withdrew tool calls it had previously requested."""

    call_ids: list


@dataclass
class ResponseDone:
    """A full model response (turn) has completed."""

    transcript: str
    usage: Optional[dict] = field(default=None)


@dataclass
class Interrupted:
    """User barged in, provider stopped generating."""


@dataclass
class SessionExpiring:
    """The provider will close this session soon and it must be resumed."""

    time_left_ms: Optional[int] = None


@dataclass
class SessionResumed:
    """A dropped session was transparently re-established mid-call."""

    reconnect_ms: float


@dataclass
class S2SError:
    """An error from the S2S provider."""

    message: str
    code: str = ""
    fatal: bool = True
