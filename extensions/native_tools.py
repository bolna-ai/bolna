"""
extensions/native_tools.py -- built-in tool schemas for native agent actions.

Native tools are actions that the LLM can invoke directly without an external
API call.  They are handled by the agent runtime itself (e.g. hanging up the
call, transferring to a human, toggling recording).

Each native tool is defined in the :data:`NATIVE_TOOLS` dict using the
standard OpenAI function-calling schema format.  The agent manager injects
these schemas alongside any user-configured API tools when building the
``tools`` list for the LLM request.

Usage::

    from extensions.native_tools import NATIVE_TOOLS, get_native_tool_schema

    # Get a single tool schema
    schema = get_native_tool_schema("end_call")

    # Get all native tools to inject into the LLM
    all_native = list(NATIVE_TOOLS.values())
"""

from __future__ import annotations

import copy
import logging
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Built-in tool schemas (OpenAI function-calling format)
# ---------------------------------------------------------------------------

NATIVE_TOOLS: dict[str, dict[str, Any]] = {

    "end_call": {
        "type": "function",
        "function": {
            "name": "end_call",
            "description": (
                "End the current phone call. Use this when the conversation "
                "has reached a natural conclusion, the caller explicitly asks "
                "to hang up, or there is nothing more to discuss."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": (
                            "Brief reason for ending the call "
                            "(e.g. 'conversation_complete', 'caller_requested', "
                            "'no_response')."
                        ),
                    },
                },
                "required": [],
            },
        },
    },

    "transfer_call": {
        "type": "function",
        "function": {
            "name": "transfer_call",
            "description": (
                "Transfer the current call to another phone number or "
                "department. Use this when the caller needs to speak with "
                "a human agent, a specific department, or the issue cannot "
                "be resolved by the AI assistant."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "call_transfer_number": {
                        "type": "string",
                        "description": (
                            "The phone number to transfer the call to, "
                            "in E.164 format (e.g. '+14155551234')."
                        ),
                    },
                    "department": {
                        "type": "string",
                        "description": (
                            "The department to transfer to "
                            "(e.g. 'sales', 'support', 'billing')."
                        ),
                    },
                    "transfer_type": {
                        "type": "string",
                        "enum": ["cold", "warm"],
                        "description": (
                            "Type of transfer. 'cold' = immediate redirect. "
                            "'warm' = play a summary to the target before connecting."
                        ),
                    },
                    "summary": {
                        "type": "string",
                        "description": (
                            "A brief summary of the conversation so far, "
                            "read to the transfer target in a warm transfer."
                        ),
                    },
                },
                "required": ["call_transfer_number"],
            },
        },
    },

    "hold_call": {
        "type": "function",
        "function": {
            "name": "hold_call",
            "description": (
                "Place the caller on hold while you look up information "
                "or perform a background action.  The caller will hear "
                "hold music or a comfort message."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": (
                            "Reason for placing on hold, communicated to the "
                            "caller (e.g. 'Looking up your account')."
                        ),
                    },
                    "duration_seconds": {
                        "type": "integer",
                        "description": (
                            "Maximum hold duration in seconds before automatically "
                            "returning to the caller. Default is 30."
                        ),
                    },
                },
                "required": [],
            },
        },
    },

    "send_dtmf": {
        "type": "function",
        "function": {
            "name": "send_dtmf",
            "description": (
                "Send DTMF (touch-tone) digits on the call. Used for "
                "navigating IVR menus or entering numeric codes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "digits": {
                        "type": "string",
                        "description": (
                            "The DTMF digits to send (0-9, *, #, w for wait). "
                            "Example: '1234#'"
                        ),
                    },
                },
                "required": ["digits"],
            },
        },
    },

    "toggle_recording": {
        "type": "function",
        "function": {
            "name": "toggle_recording",
            "description": (
                "Start or stop recording the current call. Useful for "
                "compliance when the caller requests recording to be paused."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["start", "stop"],
                        "description": "Whether to start or stop recording.",
                    },
                },
                "required": ["action"],
            },
        },
    },

    "send_sms": {
        "type": "function",
        "function": {
            "name": "send_sms",
            "description": (
                "Send an SMS text message to the caller or another number. "
                "Use this to send confirmation details, links, or follow-up "
                "information during or after the call."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "to_number": {
                        "type": "string",
                        "description": (
                            "Recipient phone number in E.164 format. "
                            "Defaults to the current caller's number if omitted."
                        ),
                    },
                    "message": {
                        "type": "string",
                        "description": "The SMS body text to send.",
                    },
                },
                "required": ["message"],
            },
        },
    },

}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_native_tool_schema(tool_name: str) -> dict[str, Any] | None:
    """Return a deep copy of the schema for the named native tool.

    Parameters
    ----------
    tool_name : str
        One of the keys in :data:`NATIVE_TOOLS`
        (e.g. ``"end_call"``, ``"transfer_call"``).

    Returns
    -------
    dict | None
        The OpenAI function-calling schema dict, or ``None`` if the
        tool name is not recognised.
    """
    schema = NATIVE_TOOLS.get(tool_name)
    if schema is None:
        logger.warning("Unknown native tool requested: %s", tool_name)
        return None
    return copy.deepcopy(schema)


def get_all_native_tool_schemas() -> list[dict[str, Any]]:
    """Return deep copies of all registered native tool schemas.

    This list can be appended directly to the ``tools`` parameter of an
    OpenAI chat completion request.
    """
    return [copy.deepcopy(schema) for schema in NATIVE_TOOLS.values()]


def get_native_tool_names() -> list[str]:
    """Return the names of all registered native tools."""
    return list(NATIVE_TOOLS.keys())


def is_native_tool(tool_name: str) -> bool:
    """Check whether a tool name corresponds to a native (built-in) tool.

    This is used by the agent runtime to decide whether to handle a tool
    call internally or dispatch it to an external API.
    """
    return tool_name in NATIVE_TOOLS
