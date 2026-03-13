"""
Call log database queries.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

import asyncpg

# ── Existing CRUD ─────────────────────────────────────────────────────────────


async def create_call_record(
    db: asyncpg.Connection,
    org_id: str,
    agent_id: str,
    call_sid: str,
    direction: str,
    from_number: str,
    to_number: str,
    campaign_id: str | None = None,
) -> asyncpg.Record:
    """Insert a pending call row and return it."""
    call_id = uuid.uuid4()
    return await db.fetchrow(
        """INSERT INTO calls
               (id, account_id, agent_id, call_sid, direction,
                from_number, to_number, status, campaign_id, created_at)
           VALUES ($1, $2::uuid, $3::uuid, $4, $5, $6, $7, 'initiated', $8, NOW())
           RETURNING *""",
        call_id,
        uuid.UUID(org_id) if isinstance(org_id, str) else org_id,
        uuid.UUID(agent_id) if isinstance(agent_id, str) else agent_id,
        call_sid,
        direction,
        from_number,
        to_number,
        uuid.UUID(campaign_id) if campaign_id else None,
    )


async def get_call(
    db: asyncpg.Connection,
    call_id: str,
    org_id: str,
) -> asyncpg.Record | None:
    return await db.fetchrow(
        "SELECT * FROM calls WHERE id = $1::uuid AND account_id = $2::uuid",
        uuid.UUID(call_id) if isinstance(call_id, str) else call_id,
        uuid.UUID(org_id) if isinstance(org_id, str) else org_id,
    )


async def get_call_by_sid(
    db: asyncpg.Connection,
    call_sid: str,
) -> asyncpg.Record | None:
    """Used inside Twilio webhook callbacks where we only have call_sid."""
    return await db.fetchrow(
        "SELECT * FROM calls WHERE call_sid = $1",
        call_sid,
    )


async def list_calls(
    db: asyncpg.Connection,
    org_id: str,
    agent_id: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[asyncpg.Record]:
    """Return call log rows for an org, optionally filtered by agent."""
    oid = uuid.UUID(org_id) if isinstance(org_id, str) else org_id
    if agent_id:
        aid = uuid.UUID(agent_id) if isinstance(agent_id, str) else agent_id
        return await db.fetch(
            """SELECT * FROM calls
               WHERE account_id = $1 AND agent_id = $2
               ORDER BY created_at DESC LIMIT $3 OFFSET $4""",
            oid, aid, limit, offset,
        )
    return await db.fetch(
        """SELECT * FROM calls
           WHERE account_id = $1
           ORDER BY created_at DESC LIMIT $2 OFFSET $3""",
        oid, limit, offset,
    )


async def update_call_status(
    db: asyncpg.Connection,
    call_sid: str,
    status: str,
    duration_seconds: int | None = None,
) -> None:
    """Update call status (and optionally duration) after Twilio callback."""
    if duration_seconds is not None:
        await db.execute(
            """UPDATE calls
               SET status = $2, duration = $3, updated_at = NOW()
               WHERE call_sid = $1""",
            call_sid, status, duration_seconds,
        )
    else:
        await db.execute(
            """UPDATE calls
               SET status = $2, updated_at = NOW()
               WHERE call_sid = $1""",
            call_sid, status,
        )


async def update_call_transcript(
    db: asyncpg.Connection,
    call_sid: str,
    transcript: list[dict],
) -> None:
    """Persist the conversation transcript JSON."""
    await db.execute(
        """UPDATE calls
           SET transcription = $2::jsonb, updated_at = NOW()
           WHERE call_sid = $1""",
        call_sid, json.dumps(transcript),
    )


async def update_call_recording(
    db: asyncpg.Connection,
    call_sid: str,
    recording_url: str,
) -> None:
    """Persist the recording URL after Twilio recording callback."""
    await db.execute(
        """UPDATE calls
           SET recording_url = $2, updated_at = NOW()
           WHERE call_sid = $1""",
        call_sid, recording_url,
    )


async def update_call_cost(
    db: asyncpg.Connection,
    call_sid: str,
    cost: float,
) -> None:
    """Update the cost of a call."""
    await db.execute(
        """UPDATE calls
           SET cost = $2, updated_at = NOW()
           WHERE call_sid = $1""",
        call_sid, cost,
    )


# ── Insert helpers for routers ────────────────────────────────────────────────


async def insert_call(
    db: asyncpg.Connection,
    call_sid: str,
    account_id: uuid.UUID,
    from_number: str,
    to_number: str,
    status: str,
    agent_id: uuid.UUID | None = None,
    call_type: str | None = None,
) -> None:
    """Insert a call record with flexible fields."""
    await db.execute(
        """INSERT INTO calls
               (call_sid, agent_id, account_id, from_number, to_number,
                status, call_type)
           VALUES ($1, $2, $3, $4, $5, $6, $7)""",
        call_sid,
        agent_id,
        account_id,
        from_number,
        to_number,
        status,
        call_type,
    )


async def insert_web_call(
    db: asyncpg.Connection,
    call_sid: str,
    agent_id: uuid.UUID | None,
    account_id: uuid.UUID,
    status: str = "initiated",
) -> None:
    """Insert a call record for a web call."""
    await db.execute(
        """INSERT INTO calls (
               call_sid, agent_id, account_id, call_type, status,
               from_number, to_number, created_at
           ) VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())""",
        call_sid,
        agent_id,
        account_id,
        "web_call",
        status,
        "web-browser",
        "web-browser",
    )


async def insert_skeleton_call(
    db: asyncpg.Connection,
    call_sid: str,
    status: str,
) -> None:
    """Insert a minimal call record for temp agent early callbacks."""
    await db.execute(
        "INSERT INTO calls (call_sid, status) VALUES ($1, $2)",
        call_sid,
        status,
    )


async def insert_campaign_call(
    db: asyncpg.Connection,
    call_sid: str,
    agent_id: uuid.UUID,
    account_id: uuid.UUID,
    from_number: str,
    to_number: str,
) -> None:
    """Insert a call record for a campaign call, ignoring conflicts."""
    await db.execute(
        """INSERT INTO calls
               (call_sid, agent_id, account_id, from_number, to_number,
                status, created_at)
           VALUES ($1,$2,$3,$4,$5,'ringing',CURRENT_TIMESTAMP)
           ON CONFLICT (call_sid) DO NOTHING""",
        call_sid,
        agent_id,
        account_id,
        from_number,
        to_number,
    )


# ── Read queries for routers ─────────────────────────────────────────────────


async def get_calls_with_agent_name(
    db: asyncpg.Connection,
    account_id: uuid.UUID,
) -> list[asyncpg.Record]:
    """Get all calls for an account with agent name joined."""
    return await db.fetch(
        """SELECT
               c.call_sid,
               c.agent_id,
               c.account_id,
               c.from_number,
               c.to_number,
               c.duration,
               c.recording_url,
               c.transcription,
               c.status,
               c.call_type,
               c.cost,
               c.created_at,
               a.name
           FROM calls c
           LEFT JOIN agents a ON c.agent_id = a.agent_id
           WHERE c.account_id = $1
           ORDER BY c.created_at DESC""",
        account_id,
    )


async def get_call_detail_by_sid(
    db: asyncpg.Connection,
    call_sid: str,
    account_id: uuid.UUID,
) -> asyncpg.Record | None:
    """Get detailed call record by call SID and account."""
    return await db.fetchrow(
        """SELECT
               call_sid,
               agent_id,
               account_id,
               from_number,
               to_number,
               duration,
               recording_url,
               transcription,
               status,
               call_type,
               cost,
               created_at,
               transferred,
               transfer_department,
               transfer_number,
               transfer_time
           FROM calls
           WHERE call_sid = $1 AND account_id = $2""",
        call_sid,
        account_id,
    )


async def get_call_with_agent_info(
    db: asyncpg.Connection,
    call_sid: str,
) -> asyncpg.Record | None:
    """Get call with agent name and webhook URL for callback processing."""
    return await db.fetchrow(
        """SELECT c.agent_id, c.account_id, c.call_type,
                  c.from_number, c.to_number, c.recording_url,
                  c.transferred, c.transfer_department,
                  c.transfer_number, c.transfer_time,
                  c.created_at, c.recording_sid, c.transcription,
                  a.name AS agent_name, a.webhook_url
           FROM calls c
           LEFT JOIN agents a ON c.agent_id = a.agent_id
           WHERE c.call_sid = $1""",
        call_sid,
    )


async def get_call_with_all_fields(
    db: asyncpg.Connection,
    call_sid: str,
) -> asyncpg.Record | None:
    """Get full call row with agent info for trigger_webhook."""
    return await db.fetchrow(
        """SELECT c.*, a.name AS agent_name, a.webhook_url
           FROM calls c
           LEFT JOIN agents a ON c.agent_id = a.agent_id
           WHERE c.call_sid = $1""",
        call_sid,
    )


# ── Update helpers for routers ────────────────────────────────────────────────


async def update_call_fields(
    db: asyncpg.Connection,
    call_sid: str,
    status: str,
    duration: float | None = None,
    cost: float | None = None,
    transcription_json: str | None = None,
) -> None:
    """Dynamic update of call status and optional fields."""
    update_fields = ["status = $1"]
    params: list[Any] = [status, call_sid]
    param_idx = 3

    if duration is not None:
        update_fields.append(f"duration = ${param_idx}")
        params.insert(param_idx - 1, duration)
        param_idx += 1

    if cost is not None:
        update_fields.append(f"cost = ${param_idx}")
        params.insert(param_idx - 1, cost)
        param_idx += 1

    if transcription_json is not None:
        update_fields.append(f"transcription = ${param_idx}::jsonb")
        params.insert(param_idx - 1, transcription_json)
        param_idx += 1

    query = f"UPDATE calls SET {', '.join(update_fields)} WHERE call_sid = $2"
    await db.execute(query, *params)


async def update_call_recording_with_sid(
    db: asyncpg.Connection,
    call_sid: str,
    recording_url: str,
    recording_sid: str,
) -> None:
    """Update recording URL and recording SID together."""
    await db.execute(
        """UPDATE calls
           SET recording_url = $1, recording_sid = $2
           WHERE call_sid = $3""",
        recording_url,
        recording_sid,
        call_sid,
    )


async def set_call_recording_sid(
    db: asyncpg.Connection,
    call_sid: str,
    recording_sid: str,
) -> None:
    """Set the recording SID on a call record."""
    await db.execute(
        "UPDATE calls SET recording_sid = $1 WHERE call_sid = $2",
        recording_sid,
        call_sid,
    )


# ── Existence / lookup helpers ────────────────────────────────────────────────


async def call_exists(
    db: asyncpg.Connection,
    call_sid: str,
) -> bool:
    """Check if a call record exists."""
    result = await db.fetchval(
        "SELECT 1 FROM calls WHERE call_sid = $1",
        call_sid,
    )
    return result is not None


async def get_call_account_id(
    db: asyncpg.Connection,
    call_sid: str,
) -> asyncpg.Record | None:
    """Get just the account_id for a call (recording callback)."""
    return await db.fetchrow(
        "SELECT account_id FROM calls WHERE call_sid = $1",
        call_sid,
    )
