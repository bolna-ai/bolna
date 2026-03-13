"""
Knowledgebase database queries.
"""

from __future__ import annotations

import uuid

import asyncpg


async def get_knowledgebase_by_name(
    db: asyncpg.Connection,
    org_id: str,
    friendly_name: str,
) -> asyncpg.Record | None:
    """Return the KB row matching *org_id* + *friendly_name*, or ``None``."""
    return await db.fetchrow(
        "SELECT id FROM knowledgebases WHERE account_id = $1::uuid AND friendly_name = $2",
        uuid.UUID(org_id) if isinstance(org_id, str) else org_id,
        friendly_name,
    )


async def create_knowledgebase(
    db: asyncpg.Connection,
    kb_id: uuid.UUID,
    org_id: str,
    name: str,
    vector_store_id: str,
    file_name: str,
    file_size: int,
    content_type: str,
) -> asyncpg.Record:
    """Insert a new knowledgebase row.

    *kb_id* is accepted as a parameter so callers that need the id before the
    insert (e.g. for file naming) can generate it themselves.
    """
    return await db.fetchrow(
        """INSERT INTO knowledgebases
               (id, account_id, friendly_name, vector_store_id,
                file_name, file_size, content_type, status, created_at)
           VALUES ($1, $2::uuid, $3, $4, $5, $6, $7, 'processing', NOW())
           RETURNING *""",
        kb_id,
        uuid.UUID(org_id) if isinstance(org_id, str) else org_id,
        name,
        vector_store_id,
        file_name,
        file_size,
        content_type,
    )


async def get_knowledgebase(
    db: asyncpg.Connection,
    kb_id: str,
    org_id: str,
) -> asyncpg.Record | None:
    return await db.fetchrow(
        "SELECT * FROM knowledgebases WHERE id = $1::uuid AND account_id = $2::uuid",
        uuid.UUID(kb_id) if isinstance(kb_id, str) else kb_id,
        uuid.UUID(org_id) if isinstance(org_id, str) else org_id,
    )


async def list_knowledgebases(
    db: asyncpg.Connection,
    org_id: str,
) -> list[asyncpg.Record]:
    return await db.fetch(
        """SELECT * FROM knowledgebases
           WHERE account_id = $1::uuid
           ORDER BY created_at DESC""",
        uuid.UUID(org_id) if isinstance(org_id, str) else org_id,
    )


async def delete_knowledgebase(
    db: asyncpg.Connection,
    kb_id: str,
    org_id: str,
) -> bool:
    result = await db.execute(
        "DELETE FROM knowledgebases WHERE id = $1::uuid AND account_id = $2::uuid",
        uuid.UUID(kb_id) if isinstance(kb_id, str) else kb_id,
        uuid.UUID(org_id) if isinstance(org_id, str) else org_id,
    )
    return result == "DELETE 1"


async def update_knowledgebase_status(
    db: asyncpg.Connection,
    kb_id: str,
    status: str,
) -> None:
    """Update KB status (processing -> ready / failed)."""
    await db.execute(
        """UPDATE knowledgebases
           SET status = $2, updated_at = NOW()
           WHERE id = $1::uuid""",
        uuid.UUID(kb_id) if isinstance(kb_id, str) else kb_id,
        status,
    )
