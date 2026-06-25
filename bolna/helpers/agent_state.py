"""Typed agent state (working memory).

State lives as a typed namespace inside context_data (writes go under "state."),
declared via the same variable_types map that drives expression routing. This module
is the state layer: the read path (the pinned state block) and the two write paths
(tool-result assignments and the built-in update_state tool). It reuses the dot-notation
and coercion primitives from expression_evaluator so they stay defined once.
"""

from typing import Any, Optional

from bolna.helpers.expression_evaluator import MISSING, coerce_to_type, enum_values, resolve_variable, set_variable
from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)

STATE_NAMESPACE = "state"


def _render_state_value(value: Any) -> str:
    """Render a value for the pinned state block (booleans as true/false)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def format_state_block(context_data: Optional[dict], variable_types: Optional[dict]) -> str:
    """Render declared variables and their current values as a compact, pinned state
    block for the LLM. Returns '' when there is nothing to show. The model reads these
    authoritative values instead of re-deriving them from the transcript."""
    if not context_data or not variable_types:
        return ""
    lines = []
    for path in variable_types:
        value = resolve_variable(context_data, path)
        if value is MISSING:
            continue
        lines.append(f"- {path}: {_render_state_value(value)}")
    if not lines:
        return ""
    return (
        "## Current state\n"
        "Authoritative values for this call. Use these directly; do not re-derive them "
        "from the conversation.\n" + "\n".join(lines)
    )


def apply_state_assignments(
    context_data: dict, assignments: dict, source: dict, variable_types: Optional[dict]
) -> list:
    """Apply a tool's response_assignments map. For each {state_path: source_path},
    read source_path from the tool response (dot-notation), coerce to the declared
    type, and write it into context_data at state_path. Missing source paths are
    skipped. Returns the list of state paths written (for logging)."""
    written = []
    if not assignments or not isinstance(source, dict) or context_data is None:
        return written
    for state_path, source_path in assignments.items():
        value = resolve_variable(source, source_path)
        if value is MISSING:
            continue
        set_variable(context_data, state_path, _coerce_for_state(value, state_path, variable_types))
        written.append(state_path)
    return written


def apply_state_updates(
    context_data: dict, updates: dict, variable_types: Optional[dict], namespace: str = STATE_NAMESPACE
) -> list:
    """Write update_state arguments into the state namespace, coercing each to its
    declared type. updates keys are bare variable names (no namespace prefix).
    Returns the list of state paths written (for logging)."""
    written = []
    if not updates or context_data is None:
        return written
    for name, value in updates.items():
        state_path = f"{namespace}.{name}"
        set_variable(context_data, state_path, _coerce_for_state(value, state_path, variable_types))
        written.append(state_path)
    return written


def _coerce_for_state(value: Any, path: str, variable_types: Optional[dict]) -> Any:
    """Coerce value to the type declared for path; store raw (and warn) on failure.
    For enum-typed variables, warn if the value is outside the allowed set."""
    try:
        coerced = coerce_to_type(value, path, variable_types)
    except (TypeError, ValueError):
        logger.warning(f"State coercion failed: {path}={value!r}; storing raw")
        return value
    allowed = enum_values(path, variable_types)
    if allowed is not None and coerced not in allowed:
        logger.warning(f"State {path}={coerced!r} not in allowed values {allowed}; storing anyway")
    return coerced
