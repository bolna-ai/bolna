"""
Model enums for Bolna - OpenAI model capabilities and compatibility.

This module fixes the hardcoded OpenAI model compatibility issue and provides
a scalable way to add new models with their capabilities.
"""

from enum import Enum
from typing import Set


class OpenAIModel(str, Enum):
    """OpenAI model identifiers."""
    # GPT-4 models
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4_TURBO = "gpt-4-turbo"
    GPT_41_NANO = "gpt-4.1-nano"
    GPT_4_1106_PREVIEW = "gpt-4-1106-preview"
    
    # GPT-3.5 models
    GPT_35_TURBO = "gpt-3.5-turbo"
    GPT_35_TURBO_16K = "gpt-3.5-turbo-16k"
    GPT_35_TURBO_1106 = "gpt-3.5-turbo-1106"


class ModelCapability(str, Enum):
    """Model capabilities for feature checking."""
    JSON_MODE = "json_mode"
    FUNCTION_CALLING = "function_calling"
    STREAMING = "streaming"
    VISION = "vision"


# Models that support JSON mode - fixes the hardcoded list issue
JSON_CAPABLE_MODELS: Set[OpenAIModel] = {
    OpenAIModel.GPT_4O,
    OpenAIModel.GPT_4O_MINI,
    OpenAIModel.GPT_4_TURBO,
    OpenAIModel.GPT_41_NANO,  # NEW: Now supports JSON mode!
    OpenAIModel.GPT_4_1106_PREVIEW,
    OpenAIModel.GPT_35_TURBO_1106,
}

# Models that support function calling
FUNCTION_CALLING_CAPABLE_MODELS: Set[OpenAIModel] = {
    OpenAIModel.GPT_4O,
    OpenAIModel.GPT_4O_MINI,
    OpenAIModel.GPT_4_TURBO,
    OpenAIModel.GPT_41_NANO,
    OpenAIModel.GPT_4_1106_PREVIEW,
    OpenAIModel.GPT_35_TURBO,
    OpenAIModel.GPT_35_TURBO_1106,
}

# Models that support streaming
STREAMING_CAPABLE_MODELS: Set[OpenAIModel] = {
    # All OpenAI models support streaming
    model for model in OpenAIModel
}

# Models that support vision
VISION_CAPABLE_MODELS: Set[OpenAIModel] = {
    OpenAIModel.GPT_4O,
    OpenAIModel.GPT_4O_MINI,
    OpenAIModel.GPT_4_TURBO,
    OpenAIModel.GPT_4_1106_PREVIEW,
}


def supports_json_mode(model: str) -> bool:
    """
    Check if a model supports JSON mode.
    
    This replaces the hardcoded list in openai_llm.py:199 and now supports
    newer models like gpt-4.1-nano, gpt-4o, gpt-4-turbo.
    
    Args:
        model: Model identifier string
        
    Returns:
        True if model supports JSON mode, False otherwise
    """
    try:
        model_enum = OpenAIModel(model)
        return model_enum in JSON_CAPABLE_MODELS
    except ValueError:
        # Unknown model, assume it doesn't support JSON mode
        return False


def supports_function_calling(model: str) -> bool:
    """Check if a model supports function calling."""
    try:
        model_enum = OpenAIModel(model)
        return model_enum in FUNCTION_CALLING_CAPABLE_MODELS
    except ValueError:
        return False


def supports_streaming(model: str) -> bool:
    """Check if a model supports streaming."""
    try:
        model_enum = OpenAIModel(model)
        return model_enum in STREAMING_CAPABLE_MODELS
    except ValueError:
        return False


def supports_vision(model: str) -> bool:
    """Check if a model supports vision capabilities."""
    try:
        model_enum = OpenAIModel(model)
        return model_enum in VISION_CAPABLE_MODELS
    except ValueError:
        return False


def get_model_capabilities(model: str) -> Set[ModelCapability]:
    """Get all capabilities for a given model."""
    capabilities = set()
    
    if supports_json_mode(model):
        capabilities.add(ModelCapability.JSON_MODE)
    if supports_function_calling(model):
        capabilities.add(ModelCapability.FUNCTION_CALLING)
    if supports_streaming(model):
        capabilities.add(ModelCapability.STREAMING)
    if supports_vision(model):
        capabilities.add(ModelCapability.VISION)
        
    return capabilities


def is_valid_openai_model(model: str) -> bool:
    """Check if a model string is a valid OpenAI model."""
    try:
        OpenAIModel(model)
        return True
    except ValueError:
        return False
