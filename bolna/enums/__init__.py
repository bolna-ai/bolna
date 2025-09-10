"""
Enums package for Bolna - Centralized constants and enums.

This package provides type-safe enums for all hardcoded values in the Bolna codebase,
improving maintainability, providing IDE autocompletion, and preventing runtime errors.
"""

from .providers import (
    SynthesizerProvider,
    TranscriberProvider, 
    LLMProvider,
    TelephonyProvider
)

from .models import (
    OpenAIModel,
    ModelCapability,
    JSON_CAPABLE_MODELS,
    supports_json_mode
)

from .tasks import (
    TaskType,
    AudioFormat,
    AudioEncoding,
    ResponseFormat,
    PipelineComponent,
    ExecutionType
)

from .constants import (
    AgentStatus,
    AgentType,
    MessageType,
    LanguageCode
)

__all__ = [
    # Providers
    "SynthesizerProvider",
    "TranscriberProvider", 
    "LLMProvider",
    "TelephonyProvider",
    
    # Models
    "OpenAIModel",
    "ModelCapability", 
    "JSON_CAPABLE_MODELS",
    "supports_json_mode",
    
    # Tasks & Formats
    "TaskType",
    "AudioFormat", 
    "AudioEncoding",
    "ResponseFormat",
    "PipelineComponent",
    "ExecutionType",
    
    # Constants
    "AgentStatus",
    "AgentType",
    "MessageType", 
    "LanguageCode",
]
