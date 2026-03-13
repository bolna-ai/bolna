"""
services/template_service.py — Template variable resolution.

Replaces the old Jinja2-based template system with simple Python string
substitution.  Template variables in agent prompts look like ``{{variable}}``.

Usage in routes:
    from services.template_service import render_prompt, resolve_variables

    variables = await resolve_variables(db, redis, agent_id, overrides)
    final_prompt = render_prompt(raw_prompt, variables)
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import asyncpg
from redis.asyncio import Redis

logger = logging.getLogger(__name__)

_VAR_RE = re.compile(r"\{\{(\w+)\}\}")


async def resolve_variables(
    db: asyncpg.Connection,
    redis: Redis,
    agent_id: str,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Load template variables for an agent.

    Priority:
      1. ``overrides`` (per-call / per-contact values passed by the API caller)
      2. Redis cache  ``tpl_vars:{agent_id}``
      3. DB fallback  (agents.template_variables column)

    Returns a flat dict of ``{variable_name: value}``.
    """
    variables: dict[str, Any] = {}

    # ── DB / Redis layer ────────────────────────────────────────────────────
    cache_key = f"tpl_vars:{agent_id}"
    cached = await redis.get(cache_key)

    if cached:
        try:
            variables = json.loads(cached)
        except (json.JSONDecodeError, TypeError):
            pass
    else:
        row = await db.fetchrow(
            "SELECT template_variables FROM agents WHERE agent_id = $1::uuid",
            agent_id if not isinstance(agent_id, str) else __import__("uuid").UUID(agent_id),
        )
        if row and row["template_variables"]:
            raw = row["template_variables"]
            if isinstance(raw, str):
                try:
                    variables = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    pass
            elif isinstance(raw, dict):
                variables = raw

            # Cache for subsequent calls using the same agent
            if variables:
                await redis.setex(cache_key, 3600, json.dumps(variables))

    # ── Per-call overrides win ──────────────────────────────────────────────
    if overrides:
        variables.update(overrides)

    return variables


def render_prompt(template: str, variables: dict[str, Any]) -> str:
    """
    Replace ``{{variable_name}}`` placeholders in *template* with values
    from *variables*.

    Special case — **whole-prompt variable**: if the entire template is a
    single ``{{variable}}`` and that variable exists in *variables*, the
    returned value is the variable value verbatim (the caller supplies the
    entire prompt).

    Missing variables are left as-is (``{{unknown}}`` stays in the output)
    so the LLM sees them literally rather than receiving a half-rendered
    prompt.
    """
    if not template:
        return template

    # Fast path: the entire prompt is one template variable
    stripped = template.strip()
    m = re.fullmatch(r"\{\{(\w+)\}\}", stripped)
    if m:
        key = m.group(1)
        if key in variables:
            return str(variables[key])

    # General path: replace each {{var}} occurrence
    def _replace(match: re.Match) -> str:
        key = match.group(1)
        if key in variables:
            return str(variables[key])
        return match.group(0)  # leave unresolved

    return _VAR_RE.sub(_replace, template)


async def invalidate_template_cache(redis: Redis, agent_id: str) -> None:
    """Call after agent template_variables are updated."""
    await redis.delete(f"tpl_vars:{agent_id}")
