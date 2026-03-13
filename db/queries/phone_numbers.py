"""
Phone number database queries.
"""

from __future__ import annotations

import uuid

import asyncpg

# ── New table (phone_numbers) ─────────────────────────────────────────────────


async def create_phone_number(
    db: asyncpg.Connection,
    org_id: str,
    phone_number: str,
    friendly_name: str,
    country_code: str,
    twilio_sid: str,
) -> asyncpg.Record:
    """Insert a newly-purchased phone number row."""
    number_id = uuid.uuid4()
    return await db.fetchrow(
        """INSERT INTO phone_numbers
               (id, account_id, phone_number, friendly_name,
                country_code, twilio_sid, created_at)
           VALUES ($1, $2::uuid, $3, $4, $5, $6, NOW())
           RETURNING *""",
        number_id,
        uuid.UUID(org_id) if isinstance(org_id, str) else org_id,
        phone_number,
        friendly_name,
        country_code,
        twilio_sid,
    )


async def get_phone_number(
    db: asyncpg.Connection,
    number_id: str,
    org_id: str,
) -> asyncpg.Record | None:
    return await db.fetchrow(
        "SELECT * FROM phone_numbers WHERE id = $1::uuid AND account_id = $2::uuid",
        uuid.UUID(number_id) if isinstance(number_id, str) else number_id,
        uuid.UUID(org_id) if isinstance(org_id, str) else org_id,
    )


async def get_phone_number_by_number(
    db: asyncpg.Connection,
    phone_number: str,
    org_id: str,
) -> asyncpg.Record | None:
    """Look up by the actual phone number string."""
    return await db.fetchrow(
        "SELECT * FROM phone_numbers WHERE phone_number = $1 AND account_id = $2::uuid",
        phone_number,
        uuid.UUID(org_id) if isinstance(org_id, str) else org_id,
    )


async def list_phone_numbers(
    db: asyncpg.Connection,
    org_id: str,
) -> list[asyncpg.Record]:
    return await db.fetch(
        "SELECT * FROM phone_numbers WHERE account_id = $1::uuid ORDER BY created_at DESC",
        uuid.UUID(org_id) if isinstance(org_id, str) else org_id,
    )


async def delete_phone_number(
    db: asyncpg.Connection,
    number_id: str,
    org_id: str,
) -> bool:
    result = await db.execute(
        "DELETE FROM phone_numbers WHERE id = $1::uuid AND account_id = $2::uuid",
        uuid.UUID(number_id) if isinstance(number_id, str) else number_id,
        uuid.UUID(org_id) if isinstance(org_id, str) else org_id,
    )
    return result == "DELETE 1"


async def delete_phone_number_by_sid(
    db: asyncpg.Connection,
    twilio_sid: str,
    org_id: str,
) -> bool:
    """Delete by Twilio SID (used when releasing a number)."""
    result = await db.execute(
        "DELETE FROM phone_numbers WHERE twilio_sid = $1 AND account_id = $2::uuid",
        twilio_sid,
        uuid.UUID(org_id) if isinstance(org_id, str) else org_id,
    )
    return result == "DELETE 1"


async def update_regulatory_bundle(
    db: asyncpg.Connection,
    number_id: str,
    regulatory_bundle_sid: str,
) -> None:
    """Link a regulatory compliance bundle to a phone number."""
    await db.execute(
        """UPDATE phone_numbers
           SET regulatory_bundle_sid = $2, updated_at = NOW()
           WHERE id = $1::uuid""",
        uuid.UUID(number_id) if isinstance(number_id, str) else number_id,
        regulatory_bundle_sid,
    )


async def get_phone_number_by_twilio_sid(
    db: asyncpg.Connection,
    twilio_sid: str,
    account_id: uuid.UUID,
) -> asyncpg.Record | None:
    """Get phone number from new table by Twilio SID."""
    return await db.fetchrow(
        "SELECT * FROM phone_numbers "
        "WHERE twilio_sid = $1 AND account_id = $2",
        twilio_sid,
        account_id,
    )


async def check_ownership_new(
    db: asyncpg.Connection,
    twilio_sid: str,
    account_id: uuid.UUID,
) -> asyncpg.Record | None:
    """Check ownership in new phone_numbers table."""
    return await db.fetchrow(
        "SELECT 1 FROM phone_numbers "
        "WHERE twilio_sid = $1 AND account_id = $2",
        twilio_sid,
        account_id,
    )


async def update_phone_number_friendly_name(
    db: asyncpg.Connection,
    friendly_name: str,
    twilio_sid: str,
    account_id: uuid.UUID,
) -> None:
    """Update friendly name in new phone_numbers table."""
    await db.execute(
        "UPDATE phone_numbers SET friendly_name = $1 "
        "WHERE twilio_sid = $2 AND account_id = $3",
        friendly_name,
        twilio_sid,
        account_id,
    )


# ── Legacy table ("PhoneNumber") ──────────────────────────────────────────────


async def get_legacy_phone_numbers(
    db: asyncpg.Connection,
    account_id: uuid.UUID,
) -> list[asyncpg.Record]:
    """Get phone numbers from the legacy PhoneNumber table."""
    return await db.fetch(
        '''SELECT p.*
           FROM "PhoneNumber" p
           JOIN "Organization" o ON o.id = p."organizationId"
           WHERE o."accountId"::uuid = $1''',
        account_id,
    )


async def get_legacy_phone_number_by_sid(
    db: asyncpg.Connection,
    sid: str,
    account_id: uuid.UUID,
) -> asyncpg.Record | None:
    """Get legacy phone number by SID for ownership verification."""
    return await db.fetchrow(
        '''SELECT p.*
           FROM "PhoneNumber" p
           JOIN "Organization" o ON o.id = p."organizationId"
           WHERE p.sid = $1 AND o."accountId"::uuid = $2''',
        sid,
        account_id,
    )


async def insert_legacy_phone_number(
    db: asyncpg.Connection,
    phone_number: str,
    friendly_name: str | None,
    sid: str,
    organization_id: str,
    stripe_subscription_id: str | None,
) -> None:
    """Insert into legacy PhoneNumber table."""
    await db.execute(
        '''INSERT INTO "PhoneNumber"
               ("id", "phoneNumber", "friendlyName", "sid",
                "organizationId", "stripeSubscriptionId",
                "createdAt", "updatedAt")
           VALUES ($1, $2, $3, $4, $5, $6, NOW(), NOW())''',
        str(uuid.uuid4()),
        phone_number,
        friendly_name,
        sid,
        organization_id,
        stripe_subscription_id,
    )


async def delete_legacy_phone_number(
    db: asyncpg.Connection,
    sid: str,
) -> None:
    """Delete from legacy PhoneNumber table."""
    await db.execute(
        'DELETE FROM "PhoneNumber" WHERE sid = $1',
        sid,
    )


async def check_ownership_legacy(
    db: asyncpg.Connection,
    sid: str,
    account_id: uuid.UUID,
) -> asyncpg.Record | None:
    """Check ownership in legacy PhoneNumber table."""
    return await db.fetchrow(
        '''SELECT 1
           FROM "PhoneNumber" p
           JOIN "Organization" o ON o.id = p."organizationId"
           WHERE p.sid = $1 AND o."accountId"::uuid = $2''',
        sid,
        account_id,
    )


async def update_legacy_friendly_name(
    db: asyncpg.Connection,
    friendly_name: str,
    sid: str,
    account_id: uuid.UUID,
) -> None:
    """Update friendly name in legacy PhoneNumber table."""
    await db.execute(
        '''UPDATE "PhoneNumber"
           SET "friendlyName" = $1
           WHERE sid = $2
             AND "organizationId" = (
                 SELECT id FROM "Organization"
                 WHERE "accountId"::uuid = $3
             )''',
        friendly_name,
        sid,
        account_id,
    )
