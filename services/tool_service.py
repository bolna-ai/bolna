"""
services/tool_service.py — API tool loading and schema caching.

main_server.py reference: search "function_calling", "get_tools_for_agent"
— various places.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import asyncpg
from redis.asyncio import Redis

from db.queries.tools import list_tools

logger = logging.getLogger(__name__)

_CACHE_TTL = 3600  # 1 hour


async def get_tools_for_agent(
    db: asyncpg.Connection,
    redis: Redis,
    org_id: str,
    agent_tool_ids: list[str],
) -> list[dict[str, Any]]:
    """
    Return OpenAI-formatted tool schemas for the given tool IDs.

    Flow:
      1. Check Redis cache key ``tools_schema:{org_id}``
      2. If miss, load from DB and build OpenAI schema
      3. Populate cache
      4. Return only the tool_ids requested by the agent config

    TODO: implement full schema builder from tool rows.
    """
    cache_key = f"tools_schema:{org_id}"
    cached = await redis.get(cache_key)
    if cached:
        all_tools: list[dict] = json.loads(cached)
    else:
        rows = await list_tools(db, org_id)
        all_tools = [_row_to_openai_schema(dict(r)) for r in rows]
        await redis.setex(cache_key, _CACHE_TTL, json.dumps(all_tools))

    # Filter to only the tools assigned to this agent
    return [t for t in all_tools if t.get("id") in agent_tool_ids]


def _row_to_openai_schema(row: dict) -> dict:
    """Convert a tool DB row into an OpenAI function-calling schema.

    TODO: flesh out from old main_server.py logic.
    """
    return {
        "id": str(row["id"]),
        "type": "function",
        "function": {
            "name": row["name"],
            "description": row["description"],
            "parameters": row.get("request_body") or {},
        },
    }


async def invalidate_tool_cache(redis: Redis, org_id: str) -> None:
    """Call this after a tool is created, updated, or deleted."""
    await redis.delete(f"tools_schema:{org_id}")
