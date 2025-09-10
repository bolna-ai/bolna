"""
Provider enums for Bolna - Centralized provider constants.

This module defines all supported providers for synthesizers, transcribers, LLMs, and telephony.
Using these enums ensures type safety and provides IDE autocompletion.
"""

from enum import Enum


class SynthesizerProvider(str, Enum):
    """Supported Text-to-Speech (TTS) providers."""
    POLLY = "polly"
    ELEVENLABS = "elevenlabs"
    OPENAI = "openai"
    DEEPGRAM = "deepgram"
    AZURE_TTS = "azuretts"
    CARTESIA = "cartesia"
    SMALLEST = "smallest"
    SARVAM = "sarvam"
    RIME = "rime"


class TranscriberProvider(str, Enum):
    """Supported Speech-to-Text (STT) providers."""
    DEEPGRAM = "deepgram"
    WHISPER = "whisper"
    AZURE = "azure"
    ASSEMBLY_AI = "assemblyai"


class LLMProvider(str, Enum):
    """Supported Large Language Model providers."""
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
    AZURE = "azure"  # Backwards compatibility


class TelephonyProvider(str, Enum):
    """Supported telephony/communication providers."""
    DEFAULT = "default"
    TWILIO = "twilio"
    EXOTEL = "exotel"
    PLIVO = "plivo"
    DAILY = "daily"


# Utility functions for backward compatibility and validation
def get_all_synthesizer_providers():
    """Get all synthesizer provider values as a list."""
    return [provider.value for provider in SynthesizerProvider]


def get_all_transcriber_providers():
    """Get all transcriber provider values as a list."""
    return [provider.value for provider in TranscriberProvider]


def get_all_llm_providers():
    """Get all LLM provider values as a list."""
    return [provider.value for provider in LLMProvider]


def get_all_telephony_providers():
    """Get all telephony provider values as a list."""
    return [provider.value for provider in TelephonyProvider]


def is_valid_synthesizer_provider(provider: str) -> bool:
    """Check if a provider string is a valid synthesizer provider."""
    try:
        SynthesizerProvider(provider)
        return True
    except ValueError:
        return False


def is_valid_transcriber_provider(provider: str) -> bool:
    """Check if a provider string is a valid transcriber provider."""
    try:
        TranscriberProvider(provider)
        return True
    except ValueError:
        return False


def is_valid_llm_provider(provider: str) -> bool:
    """Check if a provider string is a valid LLM provider."""
    try:
        LLMProvider(provider)
        return True
    except ValueError:
        return False


def is_valid_telephony_provider(provider: str) -> bool:
    """Check if a provider string is a valid telephony provider."""
    try:
        TelephonyProvider(provider)
        return True
    except ValueError:
        return False
