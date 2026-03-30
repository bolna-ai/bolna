from dataclasses import dataclass


@dataclass
class AudioDelta:
    """A chunk of audio from the S2S provider."""
    data: bytes
    is_preamble: bool = False


@dataclass
class TranscriptDelta:
    """Assistant speech transcript (partial or final)."""
    content: str
    is_final: bool


@dataclass
class InputTranscript:
    """User speech transcript from the provider's built-in transcription."""
    content: str
    is_final: bool


@dataclass
class FunctionCall:
    """The provider wants to invoke a tool/function."""
    name: str
    call_id: str
    arguments: str  # JSON string


@dataclass
class FunctionCallOutputReady:
    """Provider acknowledged the function result and will continue."""
    pass


@dataclass
class CommentaryText:
    """Text-only commentary/preamble from a reasoning model (e.g. 'One moment while I check...')."""
    content: str
    is_final: bool


@dataclass
class ResponseDone:
    """A full model response (turn) has completed."""
    transcript: str


@dataclass
class Interrupted:
    """User barged in — provider detected speech while responding."""
    pass


@dataclass
class S2SError:
    """An error from the S2S provider."""
    message: str
    code: str = ""
