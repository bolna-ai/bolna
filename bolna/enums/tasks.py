"""
Task and format enums for Bolna - Task types, audio formats, and pipeline components.

This module defines task types, audio formats, and pipeline components used throughout Bolna.
"""

from enum import Enum


class TaskType(str, Enum):
    """Types of tasks that agents can perform."""
    CONVERSATION = "conversation"
    EXTRACTION = "extraction"
    SUMMARIZATION = "summarization"
    NOTIFICATION = "notification"
    WEBHOOK = "webhook"


class AudioFormat(str, Enum):
    """Supported audio formats."""
    WAV = "wav"
    PCM = "pcm"
    MP3 = "mp3"
    FLAC = "flac"


class AudioEncoding(str, Enum):
    """Supported audio encodings."""
    LINEAR16 = "linear16"
    MULAW = "mulaw"
    ALAW = "alaw"


class ResponseFormat(str, Enum):
    """LLM response format types."""
    TEXT = "text"
    JSON_OBJECT = "json_object"


class PipelineComponent(str, Enum):
    """Components that can be part of a processing pipeline."""
    TRANSCRIBER = "transcriber"
    LLM = "llm"
    SYNTHESIZER = "synthesizer"
    INPUT = "input"
    OUTPUT = "output"


class ExecutionType(str, Enum):
    """Execution strategies for pipelines."""
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"


# Utility functions for backward compatibility
def get_all_task_types():
    """Get all task type values as a list."""
    return [task_type.value for task_type in TaskType]


def get_all_audio_formats():
    """Get all audio format values as a list."""
    return [format.value for format in AudioFormat]


def get_all_audio_encodings():
    """Get all audio encoding values as a list."""
    return [encoding.value for encoding in AudioEncoding]


def get_all_pipeline_components():
    """Get all pipeline component values as a list."""
    return [component.value for component in PipelineComponent]


def is_valid_task_type(task_type: str) -> bool:
    """Check if a task type string is valid."""
    try:
        TaskType(task_type)
        return True
    except ValueError:
        return False


def is_valid_audio_format(format: str) -> bool:
    """Check if an audio format string is valid."""
    try:
        AudioFormat(format)
        return True
    except ValueError:
        return False


def is_valid_audio_encoding(encoding: str) -> bool:
    """Check if an audio encoding string is valid."""
    try:
        AudioEncoding(encoding)
        return True
    except ValueError:
        return False


def is_valid_pipeline_component(component: str) -> bool:
    """Check if a pipeline component string is valid."""
    try:
        PipelineComponent(component)
        return True
    except ValueError:
        return False
