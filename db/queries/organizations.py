"""
Account / API-key database queries.

DB table: ``accounts``
PK:       ``account_id`` (UUID)

main_server.py reference:
  - get_account_from_api_key  (~line 540)
  - create_account             (~line 762)
  - renew_api_key              (~line 810)
  - delete_account             (~line 830)
  - get_account                (~line 905)
"""

from __future__ import annotations

import secrets
import uuid

import asyncpg

# ── API key helper ────────────────────────────────────────────────────────────

def generate_api_key() -> str:
    """Return a new ``sk_`` prefixed API key (matches production format)."""
    return "sk_" + secrets.token_urlsafe(32)


# ── Read queries ──────────────────────────────────────────────────────────────

async def get_org_by_api_key(
    db: asyncpg.Connection,
    api_key: str,
) -> asyncpg.Record | None:
    """Return the accounts row matching the given API key, or None.

    Only ``account_id`` is required by the auth layer but we return the full
    row so callers can inspect other fields if needed.
    """
    return await db.fetchrow(
        "SELECT account_id FROM accounts WHERE api_key = $1 LIMIT 1",
        api_key,
    )


async def get_org_by_id(
    db: asyncpg.Connection,
    org_id: str,
) -> asyncpg.Record | None:
    """Return the full accounts row for a given account_id UUID string."""
    return await db.fetchrow(
        "SELECT * FROM accounts WHERE account_id = $1 LIMIT 1",
        uuid.UUID(org_id),
    )


async def get_account_detail(
    db: asyncpg.Connection,
    account_id: uuid.UUID,
) -> asyncpg.Record | None:
    """Return summary fields for the accounts dashboard endpoint."""
    return await db.fetchrow(
        """
        SELECT account_id, number_of_agents, total_calls, total_duration,
               total_cost, created_at, api_key_created_at
        FROM   accounts
        WHERE  account_id = $1
        """,
        account_id,
    )


async def get_account_agents_list(
    db: asyncpg.Connection,
    account_id: uuid.UUID,
) -> list[asyncpg.Record]:
    """Return (agent_id, name) for every agent under this account."""
    return await db.fetch(
        "SELECT agent_id, name FROM agents WHERE account_id = $1",
        account_id,
    )


async def get_account_twilio_sid(
    db: asyncpg.Connection,
    account_id: uuid.UUID,
) -> asyncpg.Record | None:
    """Return twilio_subaccount_sid for the given account (used for deletion)."""
    return await db.fetchrow(
        "SELECT twilio_subaccount_sid FROM accounts WHERE account_id = $1",
        account_id,
    )


async def get_account_agent_ids(
    db: asyncpg.Connection,
    account_id: uuid.UUID,
) -> list[asyncpg.Record]:
    """Return all agent_id UUIDs for the account (used for Redis cleanup on delete)."""
    return await db.fetch(
        "SELECT agent_id FROM agents WHERE account_id = $1",
        account_id,
    )


# ── Write queries ─────────────────────────────────────────────────────────────

async def create_org(
    db: asyncpg.Connection,
    twilio_subaccount_sid: str,
    twilio_subaccount_auth_token: str,
) -> dict:
    """Insert a new account row and return id / api_key for the response."""
    account_id = uuid.uuid4()
    api_key = generate_api_key()
    await db.execute(
        """
        INSERT INTO accounts
            (account_id, twilio_subaccount_sid, twilio_subaccount_auth_token,
             api_key, api_key_created_at)
        VALUES ($1, $2, $3, $4, CURRENT_TIMESTAMP)
        """,
        account_id,
        twilio_subaccount_sid,
        twilio_subaccount_auth_token,
        api_key,
    )
    return {
        "account_id": str(account_id),
        "api_key": api_key,
        "twilio_subaccount_sid": twilio_subaccount_sid,
    }


async def update_org_api_key(
    db: asyncpg.Connection,
    account_id: uuid.UUID,
) -> dict:
    """Regenerate the API key for the given account and return new values."""
    new_api_key = generate_api_key()
    await db.execute(
        """
        UPDATE accounts
        SET    api_key = $1, api_key_created_at = CURRENT_TIMESTAMP
        WHERE  account_id = $2
        """,
        new_api_key,
        account_id,
    )
    return {"account_id": str(account_id), "api_key": new_api_key}


async def delete_account_cascade(
    db: asyncpg.Connection,
    account_id: uuid.UUID,
) -> None:
    """Delete an account and all related rows in dependency order.

    Order mirrored from main_server.py ~line 870:
    contacts → contact_lists → campaigns → calls → agents →
    "Organization" (Prisma-managed, camelCase) → accounts
    """
    await db.execute("DELETE FROM contacts       WHERE account_id = $1", account_id)
    await db.execute("DELETE FROM contact_lists  WHERE account_id = $1", account_id)
    await db.execute("DELETE FROM campaigns      WHERE account_id = $1", account_id)
    await db.execute("DELETE FROM calls          WHERE account_id = $1", account_id)
    await db.execute("DELETE FROM agents         WHERE account_id = $1", account_id)
    # "Organization" is the Prisma-managed table (camelCase, double-quoted).
    # accountId column is a UUID stored as text in Prisma; cast explicitly.
    await db.execute(
        'DELETE FROM "Organization" WHERE "accountId"::uuid = $1',
        account_id,
    )
    await db.execute("DELETE FROM accounts WHERE account_id = $1", account_id)


# ── Twilio credential queries ────────────────────────────────────────────────


async def get_twilio_credentials(
    db: asyncpg.Connection,
    account_id: uuid.UUID,
) -> asyncpg.Record | None:
    """Return Twilio sub-account SID and auth token for an account."""
    return await db.fetchrow(
        "SELECT twilio_subaccount_sid, twilio_subaccount_auth_token "
        "FROM accounts WHERE account_id = $1",
        account_id,
    )


# ── Organization plan / minutes queries ───────────────────────────────────────


async def get_org_plan_and_minutes(
    db: asyncpg.Connection,
    account_id: uuid.UUID,
) -> asyncpg.Record | None:
    """Return the plan type and remaining minutes for an org."""
    return await db.fetchrow(
        'SELECT "planType", minutes FROM "Organization" WHERE "accountId" = $1',
        str(account_id),
    )


async def get_org_pricing(
    db: asyncpg.Connection,
    account_id: uuid.UUID,
) -> asyncpg.Record | None:
    """Return pricing info (custom and default price per minute)."""
    return await db.fetchrow(
        """SELECT custom_price_per_minute, default_price_per_minute, "planType"
           FROM "Organization"
           WHERE "accountId" = $1""",
        str(account_id),
    )


async def get_org_auto_refill_settings(
    db: asyncpg.Connection,
    account_id: uuid.UUID,
) -> asyncpg.Record | None:
    """Return auto-refill settings for an org."""
    return await db.fetchrow(
        """SELECT "autoRefillMinutesEnabled",
                  "autoRefillMinutesThreshold",
                  "autoRefillMinutesAmount",
                  "stripeCustomerId"
           FROM "Organization"
           WHERE "accountId" = $1""",
        str(account_id),
    )


async def get_org_minutes(
    db: asyncpg.Connection,
    account_id: uuid.UUID,
) -> asyncpg.Record | None:
    """Get current minutes balance for an org."""
    return await db.fetchrow(
        'SELECT minutes FROM "Organization" WHERE "accountId" = $1',
        str(account_id),
    )


async def add_org_minutes(
    db: asyncpg.Connection,
    account_id: uuid.UUID,
    minutes: float,
) -> None:
    """Add minutes to an org's balance."""
    await db.execute(
        """UPDATE "Organization"
           SET minutes = minutes + $1
           WHERE "accountId" = $2""",
        minutes,
        str(account_id),
    )


async def deduct_org_minutes(
    db: asyncpg.Connection,
    account_id: uuid.UUID,
    minutes: float,
) -> None:
    """Deduct minutes from an org's balance."""
    await db.execute(
        """UPDATE "Organization"
           SET minutes = minutes - $1
           WHERE "accountId" = $2""",
        minutes,
        str(account_id),
    )


# ── Account stat counters ────────────────────────────────────────────────────


async def increment_account_total_calls(
    db: asyncpg.Connection,
    account_id: uuid.UUID,
) -> None:
    """Increment the total_calls counter on the account."""
    await db.execute(
        "UPDATE accounts SET total_calls = total_calls + 1 WHERE account_id = $1",
        account_id,
    )


async def update_account_call_stats(
    db: asyncpg.Connection,
    account_id: uuid.UUID,
    duration: float,
    cost: float,
) -> None:
    """Update total_duration and total_cost on the account."""
    await db.execute(
        """UPDATE accounts
           SET total_duration = total_duration + $1,
               total_cost = total_cost + $2
           WHERE account_id = $3""",
        duration,
        cost,
        account_id,
    )


async def get_account_full(
    db: asyncpg.Connection,
    account_id: uuid.UUID,
) -> asyncpg.Record | None:
    """Return full account row for webhook payloads."""
    return await db.fetchrow(
        "SELECT * FROM accounts WHERE account_id = $1",
        account_id,
    )


# ── Phone-number eligibility queries ──────────────────────────────────────────


async def get_free_number_eligibility(
    db: asyncpg.Connection,
    account_id: uuid.UUID,
) -> asyncpg.Record | None:
    """Get plan type and free phone number used status for eligibility check."""
    return await db.fetchrow(
        '''SELECT
               o."planType",
               o."freePhoneNumberUsed"
           FROM accounts a
           JOIN "Organization" o ON o."accountId"::uuid = a.account_id
           WHERE a.account_id = $1''',
        account_id,
    )


async def get_org_and_twilio_for_purchase(
    db: asyncpg.Connection,
    account_id: uuid.UUID,
) -> asyncpg.Record | None:
    """Get org info + twilio credentials for phone number purchase."""
    return await db.fetchrow(
        '''SELECT
               o.id              AS organization_id,
               o."planType",
               o."freePhoneNumberUsed",
               a.twilio_subaccount_sid,
               a.twilio_subaccount_auth_token,
               o."stripeCustomerId" AS stripe_customer_id
           FROM accounts a
           JOIN "Organization" o ON o."accountId"::uuid = a.account_id
           WHERE a.account_id = $1''',
        account_id,
    )


async def mark_free_phone_number_used(
    db: asyncpg.Connection,
    organization_id: str,
) -> None:
    """Mark that the org has used their free phone number."""
    await db.execute(
        '''UPDATE "Organization"
           SET "freePhoneNumberUsed" = true, "updatedAt" = NOW()
           WHERE id = $1''',
        organization_id,
    )
