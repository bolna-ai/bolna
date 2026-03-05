from enum import Enum


class ChatRole(str, Enum):
    """Chat message roles."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class TelephonyProvider(str, Enum):
    """Enum for telephony/IO providers (input/output handlers)."""
    TWILIO = "twilio"
    EXOTEL = "exotel"
    PLIVO = "plivo"
    VOBIZ = "vobiz"
    SIP_TRUNK = "sip-trunk"
    DEFAULT = "default"
    DATABASE = "database"

    @classmethod
    def telephony_providers(cls):
        """Return only telephony providers (excluding default and database)."""
        return [cls.TWILIO, cls.EXOTEL, cls.PLIVO, cls.VOBIZ, cls.SIP_TRUNK]

    @classmethod
    def all_values(cls):
        """Return all provider values as a list of strings."""
        return [provider.value for provider in cls]

    @classmethod
    def telephony_values(cls):
        """Return telephony provider values as a list of strings."""
        return [provider.value for provider in cls.telephony_providers()]


class SynthesizerProvider(str, Enum):
    """Enum for synthesizer (TTS) providers."""
    POLLY = "polly"
    ELEVENLABS = "elevenlabs"
    OPENAI = "openai"
    DEEPGRAM = "deepgram"
    AZURETTS = "azuretts"
    CARTESIA = "cartesia"
    SMALLEST = "smallest"
    SARVAM = "sarvam"
    RIME = "rime"
    PIXA = "pixa"

    @classmethod
    def all_values(cls):
        """Return all provider values as a list of strings."""
        return [provider.value for provider in cls]


class TranscriberProvider(str, Enum):
    """Enum for transcriber (STT) providers."""
    DEEPGRAM = "deepgram"
    AZURE = "azure"
    SARVAM = "sarvam"
    ASSEMBLY = "assembly"
    GOOGLE = "google"
    PIXA = "pixa"
    GLADIA = "gladia"
    ELEVENLABS = "elevenlabs"
    SMALLEST = "smallest"

    @classmethod
    def all_values(cls):
        """Return all provider values as a list of strings."""
        return [provider.value for provider in cls]


class LLMProvider(str, Enum):
    """Enum for LLM providers."""
    OPENAI = "openai"
    COHERE = "cohere"
    OLLAMA = "ollama"
    DEEPINFRA = "deepinfra"
    TOGETHER = "together"
    FIREWORKS = "fireworks"
    AZURE_OPENAI = "azure-openai"
    PERPLEXITY = "perplexity"
    VLLM = "vllm"
    ANYSCALE = "anyscale"
    CUSTOM = "custom"
    OLA = "ola"
    GROQ = "groq"
    ANTHROPIC = "anthropic"
    DEEPSEEK = "deepseek"
    OPENROUTER = "openrouter"
    AZURE = "azure"

    @classmethod
    def all_values(cls):
        """Return all provider values as a list of strings."""
        return [provider.value for provider in cls]


class ReasoningEffort(str, Enum):
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    @classmethod
    def all_values(cls):
        """Return all reasoning effort values as a list of strings."""
        return [effort.value for effort in cls]


class Verbosity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    @classmethod
    def all_values(cls):
        """Return all verbosity values as a list of strings."""
        return [verbosity.value for verbosity in cls]


class ResponseStreamEvent(str, Enum):
    """Responses API server-sent event types."""
    CREATED = "response.created"
    COMPLETED = "response.completed"
    FAILED = "response.failed"
    INCOMPLETE = "response.incomplete"
    OUTPUT_TEXT_DELTA = "response.output_text.delta"
    OUTPUT_ITEM_ADDED = "response.output_item.added"
    FUNCTION_CALL_ARGS_DELTA = "response.function_call_arguments.delta"

    @classmethod
    def all_values(cls):
        return [e.value for e in cls]


class ResponseItemType(str, Enum):
    """Responses API input item types."""
    MESSAGE = "message"
    FUNCTION_CALL = "function_call"
    FUNCTION_CALL_OUTPUT = "function_call_output"
    FUNCTION = "function"

    @classmethod
    def all_values(cls):
        return [e.value for e in cls]


class AsteriskEvent(str, Enum):
    """Asterisk WebSocket channel control events (chan_websocket)."""
    MEDIA_START = "MEDIA_START"
    MEDIA_XOFF = "MEDIA_XOFF"
    MEDIA_XON = "MEDIA_XON"
    DTMF_END = "DTMF_END"
    QUEUE_DRAINED = "QUEUE_DRAINED"
    STATUS = "STATUS"
    MEDIA_BUFFERING_COMPLETED = "MEDIA_BUFFERING_COMPLETED"


class AsteriskCommand(str, Enum):
    """Asterisk WebSocket channel commands sent by the application."""
    START_MEDIA_BUFFERING = "START_MEDIA_BUFFERING"
    STOP_MEDIA_BUFFERING = "STOP_MEDIA_BUFFERING"
    FLUSH_MEDIA = "FLUSH_MEDIA"
    REPORT_QUEUE_DRAINED = "REPORT_QUEUE_DRAINED"
    HANGUP = "HANGUP"


class AudioFormat(str, Enum):
    """Audio encoding formats."""
    ULAW = "ulaw"
    MULAW = "mulaw"
    PCM = "pcm"
    WAV = "wav"

    @classmethod
    def is_compressed(cls, fmt: str) -> bool:
        return fmt in (cls.ULAW.value, cls.MULAW.value)
