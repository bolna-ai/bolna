"""
Tool database queries.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime

import asyncpg


async def create_tool(
    db: asyncpg.Connection,
    org_id: str,
    name: str,
    description: str,
    tool_schema: dict,
    category: str | None = None,
    tags: list[str] | None = None,
) -> asyncpg.Record:
    """Insert a new API tool row and return the created record."""
    tool_id = uuid.uuid4()
    return await db.fetchrow(
        """INSERT INTO api_tools
               (id, account_id, name, description, tool_schema,
                category, tags, is_active, created_at, updated_at)
           VALUES ($1, $2::uuid, $3, $4, $5::jsonb, $6, $7, true, NOW(), NOW())
           RETURNING *""",
        tool_id,
        uuid.UUID(org_id) if isinstance(org_id, str) else org_id,
        name,
        description,
        json.dumps(tool_schema),
        category,
        tags or [],
    )


async def get_tool(
    db: asyncpg.Connection,
    tool_id: str,
    org_id: str,
) -> asyncpg.Record | None:
    """Fetch a single tool by ID, scoped to the organization."""
    return await db.fetchrow(
        "SELECT * FROM api_tools WHERE id = $1::uuid AND account_id = $2::uuid",
        uuid.UUID(tool_id) if isinstance(tool_id, str) else tool_id,
        uuid.UUID(org_id) if isinstance(org_id, str) else org_id,
    )


async def list_tools(
    db: asyncpg.Connection,
    org_id: str,
    category: str | None = None,
    is_active: bool | None = None,
    search: str | None = None,
) -> list[asyncpg.Record]:
    """List tools with optional filters, newest first."""
    oid = uuid.UUID(org_id) if isinstance(org_id, str) else org_id
    query = "SELECT * FROM api_tools WHERE account_id = $1"
    params: list = [oid]
    idx = 2

    if category is not None:
        query += f" AND category = ${idx}"
        params.append(category)
        idx += 1
    if is_active is not None:
        query += f" AND is_active = ${idx}"
        params.append(is_active)
        idx += 1
    if search:
        query += f" AND (name ILIKE ${idx} OR description ILIKE ${idx})"
        params.append(f"%{search}%")
        idx += 1

    query += " ORDER BY created_at DESC"
    return await db.fetch(query, *params)


async def update_tool(
    db: asyncpg.Connection,
    tool_id: str,
    org_id: str,
    name: str | None = None,
    description: str | None = None,
    tool_schema: dict | None = None,
    category: str | None = None,
    tags: list[str] | None = None,
    is_active: bool | None = None,
) -> asyncpg.Record | None:
    """Partial update -- only provided fields are changed. Returns updated row."""
    tid = uuid.UUID(tool_id) if isinstance(tool_id, str) else tool_id
    oid = uuid.UUID(org_id) if isinstance(org_id, str) else org_id

    sets: list[str] = []
    params: list = []
    idx = 1

    if name is not None:
        sets.append(f"name = ${idx}")
        params.append(name)
        idx += 1
    if description is not None:
        sets.append(f"description = ${idx}")
        params.append(description)
        idx += 1
    if tool_schema is not None:
        sets.append(f"tool_schema = ${idx}::jsonb")
        params.append(json.dumps(tool_schema))
        idx += 1
    if category is not None:
        sets.append(f"category = ${idx}")
        params.append(category)
        idx += 1
    if tags is not None:
        sets.append(f"tags = ${idx}")
        params.append(tags)
        idx += 1
    if is_active is not None:
        sets.append(f"is_active = ${idx}")
        params.append(is_active)
        idx += 1

    if not sets:
        return await get_tool(db, tool_id, org_id)

    sets.append("updated_at = NOW()")
    query = (
        f"UPDATE api_tools SET {', '.join(sets)} "
        f"WHERE id = ${idx} AND account_id = ${idx + 1} RETURNING *"
    )
    params.extend([tid, oid])
    return await db.fetchrow(query, *params)


async def update_test_status(
    db: asyncpg.Connection,
    tool_id: str,
    tested_at: datetime,
    test_status: str,
) -> None:
    """Record the result of a tool test execution."""
    await db.execute(
        "UPDATE api_tools SET last_tested = $1, test_status = $2 WHERE id = $3::uuid",
        tested_at,
        test_status,
        uuid.UUID(tool_id) if isinstance(tool_id, str) else tool_id,
    )


async def delete_tool(
    db: asyncpg.Connection,
    tool_id: str,
    org_id: str,
) -> bool:
    """Delete a tool. Returns True if a row was actually removed."""
    result = await db.execute(
        "DELETE FROM api_tools WHERE id = $1::uuid AND account_id = $2::uuid",
        uuid.UUID(tool_id) if isinstance(tool_id, str) else tool_id,
        uuid.UUID(org_id) if isinstance(org_id, str) else org_id,
    )
    return result == "DELETE 1"


async def get_tools_by_ids(
    db: asyncpg.Connection,
    tool_ids: list[str],
    org_id: str,
) -> list[asyncpg.Record]:
    """Fetch multiple tools by their IDs (for agent tool loading)."""
    if not tool_ids:
        return []
    uuids = [uuid.UUID(tid) if isinstance(tid, str) else tid for tid in tool_ids]
    return await db.fetch(
        "SELECT * FROM api_tools WHERE id = ANY($1::uuid[]) AND account_id = $2::uuid",
        uuids,
        uuid.UUID(org_id) if isinstance(org_id, str) else org_id,
    )
