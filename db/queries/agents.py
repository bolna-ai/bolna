"""
Agent database queries.

DB table: ``agents``
PK:       ``agent_id``  (UUID)
FK:       ``account_id`` → accounts(account_id)

main_server.py reference:
  - create_agent   (~line 1100)
  - list_agents    (~line 1320)
  - get_agent      (~line 1370)
  - update_agent   (~line 1419)
  - delete_agent   (~line 1290)
"""

from __future__ import annotations

import uuid

import asyncpg

# ── List / Read ───────────────────────────────────────────────────────────────

async def list_agents(
    db: asyncpg.Connection,
    account_id: uuid.UUID,
) -> list[asyncpg.Record]:
    """Return all non-deleted agents for an account, newest first."""
    return await db.fetch(
        """
        SELECT agent_id, name, agent_type, call_direction, inbound_phone_number,
               timezone, country, template_variables, agent_config, agent_prompts,
               total_calls, total_duration, total_cost, agent_emoji,
               COALESCE(agent_image, '') AS agent_image,
               webhook_url, is_compliant, created_at
        FROM   agents
        WHERE  account_id = $1
          AND  deleted_at IS NULL
        ORDER BY created_at DESC
        """,
        account_id,
    )


async def get_agent(
    db: asyncpg.Connection,
    agent_id: uuid.UUID,
    account_id: uuid.UUID,
) -> asyncpg.Record | None:
    """Return a single agent with optional knowledgebase join (matched by vector_id).

    The knowledgebase join mirrors main_server.py ~line 1370.
    """
    return await db.fetchrow(
        """
        SELECT a.*,
               k.file_name AS kb_file_name,
               k.kb_id
        FROM   agents a
        LEFT JOIN knowledgebases k
               ON k.vector_store_id = (
                   a.agent_config -> 'tools_config' -> 'api_tools'
                                     -> 'provider_config' ->> 'vector_id'
               )
        WHERE  a.agent_id   = $1
          AND  a.account_id = $2
          AND  a.deleted_at IS NULL
        LIMIT 1
        """,
        agent_id,
        account_id,
    )


# ── Create ────────────────────────────────────────────────────────────────────

async def create_agent(
    db: asyncpg.Connection,
    account_id: uuid.UUID,
    name: str,
    agent_type: str,
    call_direction: str,
    inbound_phone_number: str | None,
    timezone: str,
    country: str,
    agent_image: str | None,
    webhook_url: str | None,
    agent_config: str,   # JSON-serialised string
    agent_prompts: str,  # JSON-serialised string
    is_compliant: bool,
) -> uuid.UUID:
    """Insert a new agent row and bump the account's number_of_agents counter.

    Returns the new ``agent_id``.  Both DB operations run in the same implicit
    transaction; callers should wrap in ``async with db.transaction()``.
    """
    agent_id = uuid.uuid4()
    await db.execute(
        """
        INSERT INTO agents (
            agent_id, account_id, name, agent_type, call_direction,
            inbound_phone_number, timezone, country, agent_image,
            webhook_url, agent_config, agent_prompts, is_compliant
        ) VALUES (
            $1, $2, $3, $4, $5,
            $6, $7, $8, $9,
            $10, $11::jsonb, $12::jsonb, $13
        )
        """,
        agent_id, account_id, name, agent_type, call_direction,
        inbound_phone_number, timezone, country, agent_image,
        webhook_url, agent_config, agent_prompts, is_compliant,
    )
    await db.execute(
        "UPDATE accounts SET number_of_agents = number_of_agents + 1 WHERE account_id = $1",
        account_id,
    )
    return agent_id


# ── Update ────────────────────────────────────────────────────────────────────

async def update_agent(
    db: asyncpg.Connection,
    agent_id: uuid.UUID,
    account_id: uuid.UUID,
    agent_config: str,
    agent_prompts: str,
    name: str,
    agent_type: str,
    call_direction: str,
    inbound_phone_number: str | None,
    timezone: str,
    country: str,
    agent_image: str | None,
    webhook_url: str | None,
    is_compliant: bool,
) -> asyncpg.Record | None:
    """Update an agent's mutable fields.

    Returns the updated row (RETURNING *) so callers can inspect
    inbound_phone_number etc., or None if the row was not found.
    """
    return await db.fetchrow(
        """
        UPDATE agents
        SET    agent_config          = $1::jsonb,
               agent_prompts        = $2::jsonb,
               name                 = $3,
               agent_type           = $4,
               call_direction       = $5,
               inbound_phone_number = $6,
               timezone             = $7,
               country              = $8,
               agent_image          = $9,
               webhook_url          = $10,
               is_compliant         = $11
        WHERE  agent_id   = $12
          AND  account_id = $13
          AND  deleted_at IS NULL
        RETURNING *
        """,
        agent_config, agent_prompts, name, agent_type, call_direction,
        inbound_phone_number, timezone, country, agent_image, webhook_url,
        is_compliant, agent_id, account_id,
    )


# ── Delete (soft) ─────────────────────────────────────────────────────────────

async def soft_delete_agent(
    db: asyncpg.Connection,
    agent_id: uuid.UUID,
    account_id: uuid.UUID,
) -> asyncpg.Record | None:
    """Mark an agent as deleted (sets deleted_at) and return its row.

    Returns the record so the caller can clean up Redis / Twilio,
    or None if no matching active agent was found.

    Also decrements accounts.number_of_agents (floored at 0).
    """
    row = await db.fetchrow(
        """
        UPDATE agents
        SET    deleted_at = CURRENT_TIMESTAMP
        WHERE  agent_id   = $1
          AND  account_id = $2
          AND  deleted_at IS NULL
        RETURNING agent_id, inbound_phone_number
        """,
        agent_id,
        account_id,
    )
    if row:
        await db.execute(
            """
            UPDATE accounts
            SET    number_of_agents = GREATEST(number_of_agents - 1, 0)
            WHERE  account_id = $1
            """,
            account_id,
        )
    return row


# ── Conflict check ────────────────────────────────────────────────────────────

async def get_agents_by_inbound_phone(
    db: asyncpg.Connection,
    phone_number: str,
    exclude_agent_id: uuid.UUID | None = None,
) -> list[asyncpg.Record]:
    """Return any active agents already using this inbound phone number.

    Used to block duplicate inbound number assignments.
    """
    if exclude_agent_id:
        return await db.fetch(
            """
            SELECT agent_id FROM agents
            WHERE  inbound_phone_number = $1
              AND  deleted_at IS NULL
              AND  agent_id <> $2
            """,
            phone_number,
            exclude_agent_id,
        )
    return await db.fetch(
        """
        SELECT agent_id FROM agents
        WHERE  inbound_phone_number = $1
          AND  deleted_at IS NULL
        """,
        phone_number,
    )


# ── Targeted lookups for routers ──────────────────────────────────────────────


async def get_agent_call_info(
    db: asyncpg.Connection,
    agent_id: uuid.UUID,
) -> asyncpg.Record | None:
    """Return call_direction, is_compliant, and webhook_url for an agent."""
    return await db.fetchrow(
        "SELECT call_direction, is_compliant, webhook_url "
        "FROM agents WHERE agent_id = $1",
        agent_id,
    )


async def get_agent_webhook_info(
    db: asyncpg.Connection,
    agent_id: uuid.UUID,
) -> asyncpg.Record | None:
    """Return webhook_url and name for webhook notifications."""
    return await db.fetchrow(
        "SELECT webhook_url, name FROM agents WHERE agent_id = $1",
        agent_id,
    )


async def get_agent_compliance(
    db: asyncpg.Connection,
    agent_id: uuid.UUID,
) -> asyncpg.Record | None:
    """Return the is_compliant flag for an agent."""
    return await db.fetchrow(
        "SELECT is_compliant FROM agents WHERE agent_id = $1",
        agent_id,
    )


async def get_agent_basic_info(
    db: asyncpg.Connection,
    agent_id: uuid.UUID,
    account_id: uuid.UUID,
) -> asyncpg.Record | None:
    """Return agent_id and name for validation (web calls, chat sessions)."""
    return await db.fetchrow(
        "SELECT agent_id, name FROM agents "
        "WHERE agent_id = $1 AND account_id = $2",
        agent_id,
        account_id,
    )


async def get_agent_config_info(
    db: asyncpg.Connection,
    agent_id: uuid.UUID,
    account_id: uuid.UUID,
) -> asyncpg.Record | None:
    """Return agent_config, agent_prompts, name, account_id for web call config."""
    return await db.fetchrow(
        """SELECT agent_config, agent_prompts, name, account_id
           FROM agents
           WHERE agent_id = $1 AND account_id = $2""",
        agent_id,
        account_id,
    )


async def get_agent_direction(
    db: asyncpg.Connection,
    agent_id: uuid.UUID,
    account_id: uuid.UUID,
) -> asyncpg.Record | None:
    """Return call_direction for campaign validation."""
    return await db.fetchrow(
        "SELECT call_direction FROM agents WHERE agent_id=$1 AND account_id=$2",
        agent_id,
        account_id,
    )


# ── Stat counters ─────────────────────────────────────────────────────────────


async def increment_agent_total_calls(
    db: asyncpg.Connection,
    agent_id: uuid.UUID,
) -> None:
    """Atomically increment total_calls on an agent."""
    await db.execute(
        "UPDATE agents SET total_calls = total_calls + 1 WHERE agent_id = $1",
        agent_id,
    )


async def update_agent_call_stats(
    db: asyncpg.Connection,
    agent_id: uuid.UUID,
    duration: float,
    cost: float,
) -> None:
    """Update total_duration and total_cost on an agent."""
    await db.execute(
        """UPDATE agents
           SET total_duration = total_duration + $1,
               total_cost = total_cost + $2
           WHERE agent_id = $3""",
        duration,
        cost,
        agent_id,
    )
