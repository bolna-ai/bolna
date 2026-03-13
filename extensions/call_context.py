"""
extensions/call_context.py — builds the call-context dict injected into the
agent's system prompt at call start.

main_server.py reference: search "call_context", "build_initial_prompt",
"direction", "timezone" — various inline code sections.
"""

from __future__ import annotations

from datetime import datetime, timezone


def build_call_context(
    direction: str,
    from_number: str,
    to_number: str,
    agent_phone_number: str,
    user_phone_number: str,
    call_sid: str,
    campaign_id: str | None = None,
    contact_variables: dict | None = None,
) -> dict:
    """
    Return a dict of context values available to the system prompt template.

    These are resolved *once* at call start and embedded in the system prompt,
    so neither the ASR loop nor the LLM needs to query the DB at speech time.
    """
    return {
        # Call metadata
        "call_sid": call_sid,
        "direction": direction,          # "inbound" | "outbound"
        "from_number": from_number,
        "to_number": to_number,
        "agent_phone_number": agent_phone_number,
        "user_phone_number": user_phone_number,
        "campaign_id": campaign_id or "",
        # Timestamps
        "call_start_utc": datetime.now(tz=timezone.utc).isoformat(),
        # Per-contact variables (campaigns only)
        **(contact_variables or {}),
    }


def resolve_system_prompt(template: str, context: dict) -> str:
    """
    Apply context variables to the system-prompt template.

    Uses Python's built-in str.format_map() — no Jinja2 dependency needed.
    Unknown keys are left as-is (format_map with a default dict).

    Example:
        template = "You are calling {user_name} at {user_phone_number}."
        context = {"user_name": "Alice", "user_phone_number": "+1234567890"}
        → "You are calling Alice at +1234567890."
    """

    class SafeMap(dict):
        def __missing__(self, key: str) -> str:
            return "{" + key + "}"

    return template.format_map(SafeMap(context))
