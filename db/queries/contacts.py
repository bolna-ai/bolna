"""
Contact list / contact member database queries.

Every function accepts an ``asyncpg.Connection`` as its first argument so that
callers can compose multiple calls inside a single transaction when needed.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

import asyncpg

# -- Contact lists -------------------------------------------------------------


async def create_contact_list(
    db: asyncpg.Connection,
    account_id: uuid.UUID,
    name: str,
    description: str = "",
) -> uuid.UUID:
    """Insert a new contact list and return the generated *list_id*."""
    list_id = uuid.uuid4()
    await db.execute(
        """INSERT INTO contact_lists (list_id, account_id, name, description)
           VALUES ($1, $2, $3, $4)""",
        list_id,
        account_id,
        name,
        description,
    )
    return list_id


async def get_contact_list(
    db: asyncpg.Connection,
    list_id: uuid.UUID,
    account_id: uuid.UUID,
) -> asyncpg.Record | None:
    """Return the contact-list row if it belongs to *account_id*, else ``None``."""
    return await db.fetchrow(
        "SELECT list_id FROM contact_lists WHERE list_id = $1 AND account_id = $2",
        list_id,
        account_id,
    )


async def list_contact_lists(
    db: asyncpg.Connection,
    account_id: uuid.UUID,
) -> list[asyncpg.Record]:
    """Return every contact list owned by *account_id*, each with its member count."""
    return await db.fetch(
        """SELECT
               cl.list_id,
               cl.name,
               cl.description,
               cl.created_at,
               COUNT(c.contact_id) AS contact_count
           FROM contact_lists cl
           LEFT JOIN contacts c ON cl.list_id = c.list_id
           WHERE cl.account_id = $1
           GROUP BY cl.list_id, cl.name, cl.description, cl.created_at
           ORDER BY cl.created_at DESC""",
        account_id,
    )


async def check_campaign_dependency(
    db: asyncpg.Connection,
    list_id: uuid.UUID,
) -> asyncpg.Record | None:
    """Return a campaign record if *any* campaign references *list_id*, else ``None``.

    Used as a guard before deleting a contact list.
    """
    return await db.fetchrow(
        "SELECT campaign_id FROM campaigns WHERE list_id = $1",
        list_id,
    )


async def delete_contact_list(
    db: asyncpg.Connection,
    list_id: uuid.UUID,
    account_id: uuid.UUID,
) -> None:
    """Delete the contact-list row itself (contacts should be removed first)."""
    await db.execute(
        "DELETE FROM contact_lists WHERE list_id = $1 AND account_id = $2",
        list_id,
        account_id,
    )


# -- Contacts (members) -------------------------------------------------------


async def insert_contact(
    db: asyncpg.Connection,
    list_id: uuid.UUID,
    account_id: uuid.UUID,
    phone_number: str,
    template_variables: dict[str, Any] | None = None,
) -> uuid.UUID:
    """Insert a single contact and return the generated *contact_id*."""
    contact_id = uuid.uuid4()
    await db.execute(
        """INSERT INTO contacts
               (contact_id, list_id, account_id, phone_number, template_variables)
           VALUES ($1, $2, $3, $4, $5)""",
        contact_id,
        list_id,
        account_id,
        phone_number,
        json.dumps(template_variables or {}),
    )
    return contact_id


async def list_contacts(
    db: asyncpg.Connection,
    list_id: uuid.UUID,
    account_id: uuid.UUID,
) -> list[asyncpg.Record]:
    """Return all contacts in a specific list, newest first."""
    return await db.fetch(
        """SELECT contact_id, phone_number, template_variables, created_at
           FROM contacts
           WHERE list_id = $1 AND account_id = $2
           ORDER BY created_at DESC""",
        list_id,
        account_id,
    )


async def find_contact_by_phone(
    db: asyncpg.Connection,
    list_id: uuid.UUID,
    phone_number: str,
) -> asyncpg.Record | None:
    """Return a contact row if *phone_number* already exists in the list."""
    return await db.fetchrow(
        "SELECT contact_id FROM contacts WHERE list_id = $1 AND phone_number = $2",
        list_id,
        phone_number,
    )


async def delete_contacts_by_list(
    db: asyncpg.Connection,
    list_id: uuid.UUID,
    account_id: uuid.UUID,
) -> None:
    """Delete every contact belonging to a list, scoped by account."""
    await db.execute(
        "DELETE FROM contacts WHERE list_id = $1 AND account_id = $2",
        list_id,
        account_id,
    )


async def get_campaign_contacts(
    db: asyncpg.Connection,
    list_id: str,
) -> list[asyncpg.Record]:
    """Return all contacts for campaign processing (no pagination).

    Accepts *list_id* as either ``str`` or ``uuid.UUID`` for caller convenience.
    """
    return await db.fetch(
        "SELECT * FROM contacts WHERE list_id = $1::uuid ORDER BY created_at",
        uuid.UUID(list_id) if isinstance(list_id, str) else list_id,
    )


# -- Webhook enrichment lookups ------------------------------------------------


async def get_contact_by_call_sid(
    db: asyncpg.Connection,
    call_sid: str,
) -> asyncpg.Record | None:
    """Find a contact by its associated call_sid."""
    return await db.fetchrow(
        "SELECT * FROM contacts WHERE call_sid = $1",
        call_sid,
    )


async def get_contact_by_phone_latest(
    db: asyncpg.Connection,
    phone_number: str,
) -> asyncpg.Record | None:
    """Find the most recent contact by phone number."""
    return await db.fetchrow(
        "SELECT * FROM contacts WHERE phone_number = $1 "
        "ORDER BY created_at DESC LIMIT 1",
        phone_number,
    )
