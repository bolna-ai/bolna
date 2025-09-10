"""
Constants enums for Bolna - Status values, agent types, and other constants.

This module defines status values, agent types, message types, and other constants used in Bolna.
"""

from enum import Enum


class AgentStatus(str, Enum):
    """Agent status values."""
    CREATED = "created"
    SEEDING = "seeding"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    DELETED = "deleted"


class AgentType(str, Enum):
    """Agent type identifiers."""
    SIMPLE_LLM_AGENT = "simple_llm_agent"
    CONVERSATIONAL_AGENT = "conversational_agent"
    EXTRACTION_AGENT = "extraction_agent"
    GRAPH_BASED_AGENT = "graph_based_agent"
    KNOWLEDGEBASE_AGENT = "knowledgebase_agent"
    MULTI_AGENT = "multiagent"
    LLM_AGENT_GRAPH = "llm_agent_graph"
    GRAPH_AGENT = "graph_agent"
    OTHER = "other"


class MessageType(str, Enum):
    """Message/data type identifiers."""
    AUDIO = "audio"
    TEXT = "text"
    MARK = "mark"
    PRE_MARK_MESSAGE = "pre_mark_message"


class LanguageCode(str, Enum):
    """Supported language codes."""
    ENGLISH = "en"
    GERMAN = "ge"
    FRENCH = "fr"
    SPANISH = "es"
    PORTUGUESE = "pt"
    ITALIAN = "it"
    DUTCH = "nl"
    POLISH = "pl"
    RUSSIAN = "ru"
    JAPANESE = "ja"
    CHINESE = "zh"
    KOREAN = "ko"
    ARABIC = "ar"
    HINDI = "hi"


class AgentFlowType(str, Enum):
    """Agent flow type identifiers."""
    STREAMING = "streaming"
    NON_STREAMING = "non_streaming"


# Utility functions for backward compatibility
def get_all_agent_statuses():
    """Get all agent status values as a list."""
    return [status.value for status in AgentStatus]


def get_all_agent_types():
    """Get all agent type values as a list."""
    return [agent_type.value for agent_type in AgentType]


def get_all_message_types():
    """Get all message type values as a list."""
    return [msg_type.value for msg_type in MessageType]


def get_all_language_codes():
    """Get all language code values as a list."""
    return [lang.value for lang in LanguageCode]


def is_valid_agent_status(status: str) -> bool:
    """Check if an agent status string is valid."""
    try:
        AgentStatus(status)
        return True
    except ValueError:
        return False


def is_valid_agent_type(agent_type: str) -> bool:
    """Check if an agent type string is valid."""
    try:
        AgentType(agent_type)
        return True
    except ValueError:
        return False


def is_valid_message_type(msg_type: str) -> bool:
    """Check if a message type string is valid."""
    try:
        MessageType(msg_type)
        return True
    except ValueError:
        return False


def is_valid_language_code(lang_code: str) -> bool:
    """Check if a language code string is valid."""
    try:
        LanguageCode(lang_code)
        return True
    except ValueError:
        return False
