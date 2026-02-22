from enum import Enum


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


class LogComponent(str, Enum):
    """Enum for CSV trace log component types."""
    ERROR = "error"
    FUNCTION_CALL = "function_call"
    GRAPH_ROUTING = "graph_routing"
    LLM = "llm"
    LLM_HANGUP = "llm_hangup"
    LLM_LANGUAGE_DETECTION = "llm_language_detection"
    LLM_VOICEMAIL = "llm_voicemail"
    SYNTHESIZER = "synthesizer"
    TRANSCRIBER = "transcriber"
    WARNING = "warning"

    @classmethod
    def all_values(cls):
        return [c.value for c in cls]


class LogDirection(str, Enum):
    """Enum for CSV trace log direction types."""
    ERROR = "error"
    REQUEST = "request"
    RESPONSE = "response"
    WARNING = "warning"

    @classmethod
    def all_values(cls):
        return [d.value for d in cls]
