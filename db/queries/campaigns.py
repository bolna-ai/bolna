"""
Campaign database queries.
"""

from __future__ import annotations

import json
import uuid

import asyncpg

# ── Original CRUD (kept for backward compat) ──────────────────────────────────


async def create_campaign(
    db: asyncpg.Connection,
    org_id: str,
    agent_id: str,
    name: str,
    contact_list_id: str,
    scheduled_at: str | None = None,
    phone_numbers: list[str] | None = None,
    concurrent_calls: int = 1,
) -> asyncpg.Record:
    """Insert a new campaign in 'created' status."""
    campaign_id = uuid.uuid4()
    return await db.fetchrow(
        """INSERT INTO campaigns
               (id, account_id, agent_id, name, contact_list_id,
                status, scheduled_at, phone_numbers, concurrent_calls, created_at)
           VALUES ($1, $2::uuid, $3::uuid, $4, $5::uuid,
                   'created', $6, $7, $8, NOW())
           RETURNING *""",
        campaign_id,
        uuid.UUID(org_id) if isinstance(org_id, str) else org_id,
        uuid.UUID(agent_id) if isinstance(agent_id, str) else agent_id,
        name,
        uuid.UUID(contact_list_id) if isinstance(contact_list_id, str) else contact_list_id,
        scheduled_at,
        json.dumps(phone_numbers) if phone_numbers else None,
        concurrent_calls,
    )


async def get_campaign(
    db: asyncpg.Connection,
    campaign_id: str,
    org_id: str,
) -> asyncpg.Record | None:
    return await db.fetchrow(
        "SELECT * FROM campaigns WHERE id = $1::uuid AND account_id = $2::uuid",
        uuid.UUID(campaign_id) if isinstance(campaign_id, str) else campaign_id,
        uuid.UUID(org_id) if isinstance(org_id, str) else org_id,
    )


async def list_campaigns(
    db: asyncpg.Connection,
    org_id: str,
    limit: int = 100,
    offset: int = 0,
) -> list[asyncpg.Record]:
    return await db.fetch(
        """SELECT * FROM campaigns WHERE account_id = $1::uuid
           ORDER BY created_at DESC LIMIT $2 OFFSET $3""",
        uuid.UUID(org_id) if isinstance(org_id, str) else org_id,
        limit, offset,
    )


async def update_campaign_status(
    db: asyncpg.Connection,
    campaign_id: str,
    org_id: str,
    status: str,
) -> asyncpg.Record | None:
    """Transition a campaign to a new status."""
    return await db.fetchrow(
        """UPDATE campaigns
           SET status = $3, updated_at = NOW()
           WHERE id = $1::uuid AND account_id = $2::uuid
           RETURNING *""",
        uuid.UUID(campaign_id) if isinstance(campaign_id, str) else campaign_id,
        uuid.UUID(org_id) if isinstance(org_id, str) else org_id,
        status,
    )


async def delete_campaign(
    db: asyncpg.Connection,
    campaign_id: str,
    org_id: str,
) -> bool:
    result = await db.execute(
        "DELETE FROM campaigns WHERE id = $1::uuid AND account_id = $2::uuid",
        uuid.UUID(campaign_id) if isinstance(campaign_id, str) else campaign_id,
        uuid.UUID(org_id) if isinstance(org_id, str) else org_id,
    )
    return result == "DELETE 1"


async def increment_campaign_dialed(
    db: asyncpg.Connection,
    campaign_id: str,
) -> None:
    """Atomically increment the dialed counter after each call attempt."""
    await db.execute(
        """UPDATE campaigns
           SET dialed = COALESCE(dialed, 0) + 1, updated_at = NOW()
           WHERE id = $1::uuid""",
        uuid.UUID(campaign_id) if isinstance(campaign_id, str) else campaign_id,
    )


async def increment_campaign_answered(
    db: asyncpg.Connection,
    campaign_id: str,
) -> None:
    """Atomically increment the answered counter when a call is answered."""
    await db.execute(
        """UPDATE campaigns
           SET answered = COALESCE(answered, 0) + 1, updated_at = NOW()
           WHERE id = $1::uuid""",
        uuid.UUID(campaign_id) if isinstance(campaign_id, str) else campaign_id,
    )


async def update_campaign_total_contacts(
    db: asyncpg.Connection,
    campaign_id: str,
    total: int,
) -> None:
    """Set the total_contacts count once the contact list is loaded."""
    await db.execute(
        """UPDATE campaigns
           SET total_contacts = $2, updated_at = NOW()
           WHERE id = $1::uuid""",
        uuid.UUID(campaign_id) if isinstance(campaign_id, str) else campaign_id,
        total,
    )


# ── Router-level queries ─────────────────────────────────────────────────────


async def get_campaign_for_account(
    db: asyncpg.Connection,
    campaign_id: uuid.UUID,
    account_id: uuid.UUID,
) -> asyncpg.Record | None:
    """Get campaign_id for ownership verification."""
    return await db.fetchrow(
        "SELECT campaign_id FROM campaigns WHERE campaign_id=$1 AND account_id=$2",
        campaign_id,
        account_id,
    )


async def get_campaign_with_status(
    db: asyncpg.Connection,
    campaign_id: uuid.UUID,
    account_id: uuid.UUID,
) -> asyncpg.Record | None:
    """Get campaign with status and scheduled_time for resume."""
    return await db.fetchrow(
        "SELECT campaign_id, status, scheduled_time FROM campaigns "
        "WHERE campaign_id=$1 AND account_id=$2",
        campaign_id,
        account_id,
    )


async def get_campaign_display_info(
    db: asyncpg.Connection,
    campaign_id: uuid.UUID,
    account_id: uuid.UUID,
) -> asyncpg.Record | None:
    """Get campaign fields for the detail endpoint."""
    return await db.fetchrow(
        """SELECT campaign_id, name, status,
                  total_contacts, completed_calls, failed_calls, pending_calls,
                  created_at, scheduled_time, started_at, completed_at, timezone
           FROM campaigns
           WHERE campaign_id=$1 AND account_id=$2""",
        campaign_id,
        account_id,
    )


async def get_campaign_for_delete(
    db: asyncpg.Connection,
    campaign_id: uuid.UUID,
    account_id: uuid.UUID,
) -> asyncpg.Record | None:
    """Get list_id and status needed for deletion."""
    return await db.fetchrow(
        "SELECT list_id, status FROM campaigns WHERE campaign_id=$1 AND account_id=$2",
        campaign_id,
        account_id,
    )


async def get_campaign_for_edit(
    db: asyncpg.Connection,
    campaign_id: uuid.UUID,
    account_id: uuid.UUID,
) -> asyncpg.Record | None:
    """Get campaign fields needed for edit validation."""
    return await db.fetchrow(
        "SELECT campaign_id, list_id, status FROM campaigns "
        "WHERE campaign_id=$1 AND account_id=$2",
        campaign_id,
        account_id,
    )


async def create_campaign_full(
    db: asyncpg.Connection,
    account_id: uuid.UUID,
    agent_id: uuid.UUID,
    list_id: uuid.UUID,
    name: str,
    description: str,
    status: str,
    sender_phone_numbers: str,
    total_contacts: int,
    delay_between_calls: int,
    scheduled_time,
    timezone: str,
    rotate_numbers_after: int,
) -> uuid.UUID:
    """Insert a campaign with full parameters. Returns the campaign_id."""
    return await db.fetchval(
        """INSERT INTO campaigns (
               account_id, agent_id, list_id, name, description, status,
               sender_phone_numbers, total_contacts, delay_between_calls,
               scheduled_time, timezone, rotate_numbers_after,
               completed_calls, failed_calls, pending_calls
           ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12, 0, 0, $8)
           RETURNING campaign_id""",
        account_id,
        agent_id,
        list_id,
        name,
        description,
        status,
        sender_phone_numbers,
        total_contacts,
        delay_between_calls,
        scheduled_time,
        timezone,
        rotate_numbers_after,
    )


async def update_campaign_paused(
    db: asyncpg.Connection,
    campaign_id: uuid.UUID,
) -> None:
    """Mark campaign as paused."""
    await db.execute(
        "UPDATE campaigns SET status='paused', status_changed_at=CURRENT_TIMESTAMP "
        "WHERE campaign_id=$1",
        campaign_id,
    )


async def update_campaign_status_with_timestamp(
    db: asyncpg.Connection,
    campaign_id: uuid.UUID,
    status: str,
) -> None:
    """Update campaign status with status_changed_at timestamp."""
    await db.execute(
        "UPDATE campaigns SET status=$2, status_changed_at=CURRENT_TIMESTAMP "
        "WHERE campaign_id=$1",
        campaign_id,
        status,
    )


async def delete_campaign_by_id(
    db: asyncpg.Connection,
    campaign_id: uuid.UUID,
) -> None:
    """Delete a campaign by its primary key."""
    await db.execute(
        "DELETE FROM campaigns WHERE campaign_id=$1",
        campaign_id,
    )


async def delete_contacts_by_list_id(
    db: asyncpg.Connection,
    list_id: uuid.UUID,
) -> None:
    """Delete all contacts belonging to a contact list."""
    await db.execute(
        "DELETE FROM contacts WHERE list_id=$1",
        list_id,
    )


async def delete_contact_list_by_id(
    db: asyncpg.Connection,
    list_id: uuid.UUID,
) -> None:
    """Delete a contact list by its primary key."""
    await db.execute(
        "DELETE FROM contact_lists WHERE list_id=$1",
        list_id,
    )


async def count_contacts_for_list(
    db: asyncpg.Connection,
    list_id: uuid.UUID,
    account_id: uuid.UUID,
) -> int:
    """Count contacts in a list for an account."""
    return await db.fetchval(
        "SELECT COUNT(*) FROM contacts WHERE list_id=$1 AND account_id=$2",
        list_id,
        account_id,
    ) or 0


async def update_campaign_dynamic(
    db: asyncpg.Connection,
    campaign_id: uuid.UUID,
    set_clauses: list[str],
    values: list,
) -> None:
    """Execute a dynamic UPDATE on a campaign."""
    values.append(campaign_id)
    await db.execute(
        f"UPDATE campaigns SET {', '.join(set_clauses)} "
        f"WHERE campaign_id = ${len(values)}",
        *values,
    )


async def list_campaigns_with_details(
    db: asyncpg.Connection,
    account_id: uuid.UUID,
) -> list[asyncpg.Record]:
    """List all campaigns with agent and contact list names joined."""
    return await db.fetch(
        """SELECT
               c.campaign_id,
               c.name,
               c.status,
               c.total_contacts,
               c.completed_calls,
               c.failed_calls,
               c.pending_calls,
               c.created_at,
               c.scheduled_time,
               c.started_at,
               c.completed_at,
               c.timezone,
               c.delay_between_calls,
               c.sender_phone_numbers,
               a.name  AS agent_name,
               cl.name AS contact_list_name,
               CASE c.status
                   WHEN 'running'   THEN 'Active'
                   WHEN 'paused'    THEN 'Paused'
                   WHEN 'completed' THEN 'Completed'
                   WHEN 'failed'    THEN 'Failed'
                   WHEN 'pending'   THEN 'Scheduled'
                   WHEN 'overdue'   THEN 'Overdue'
                   ELSE c.status
               END AS campaign_status
           FROM campaigns c
           LEFT JOIN agents       a  ON c.agent_id = a.agent_id
           LEFT JOIN contact_lists cl ON c.list_id  = cl.list_id
           WHERE c.account_id = $1
           ORDER BY c.created_at DESC""",
        account_id,
    )


# ── Background task helpers ──────────────────────────────────────────────────


async def get_campaign_processing_info(
    db: asyncpg.Connection,
    campaign_id: uuid.UUID,
) -> asyncpg.Record | None:
    """Get campaign info needed for background processing."""
    return await db.fetchrow(
        "SELECT campaign_id, account_id, agent_id, list_id, "
        "delay_between_calls, rotate_numbers_after, sender_phone_numbers "
        "FROM campaigns WHERE campaign_id=$1",
        campaign_id,
    )


async def mark_campaign_running(
    db: asyncpg.Connection,
    campaign_id: uuid.UUID,
) -> None:
    """Mark a campaign as running with started_at timestamp."""
    await db.execute(
        "UPDATE campaigns SET status='running', started_at=CURRENT_TIMESTAMP, "
        "status_changed_at=CURRENT_TIMESTAMP WHERE campaign_id=$1",
        campaign_id,
    )


async def mark_campaign_completed(
    db: asyncpg.Connection,
    campaign_id: uuid.UUID,
) -> None:
    """Mark a campaign as completed."""
    await db.execute(
        """UPDATE campaigns
           SET status='completed',
               completed_at=CURRENT_TIMESTAMP,
               status_changed_at=CURRENT_TIMESTAMP
           WHERE campaign_id=$1""",
        campaign_id,
    )


async def mark_campaign_failed(
    db: asyncpg.Connection,
    campaign_id: uuid.UUID,
) -> None:
    """Mark a campaign as failed."""
    await db.execute(
        "UPDATE campaigns SET status='failed', status_changed_at=CURRENT_TIMESTAMP "
        "WHERE campaign_id=$1",
        campaign_id,
    )


async def get_overdue_campaigns(
    db: asyncpg.Connection,
) -> list[asyncpg.Record]:
    """Get campaigns that are pending and past their scheduled time."""
    return await db.fetch(
        "SELECT campaign_id FROM campaigns "
        "WHERE status='pending' AND scheduled_time <= NOW()"
    )


async def mark_campaign_overdue(
    db: asyncpg.Connection,
    campaign_id: uuid.UUID,
) -> None:
    """Mark a campaign as overdue."""
    await db.execute(
        "UPDATE campaigns SET status='overdue' WHERE campaign_id=$1",
        campaign_id,
    )


async def get_next_pending_contact(
    db: asyncpg.Connection,
    campaign_id: uuid.UUID,
) -> asyncpg.Record | None:
    """SELECT ... FOR UPDATE SKIP LOCKED to avoid concurrent task conflicts."""
    return await db.fetchrow(
        """SELECT c.contact_id, c.phone_number, c.template_variables, c.account_id
           FROM contacts c
           JOIN campaigns camp ON c.list_id = camp.list_id
           WHERE camp.campaign_id = $1
             AND c.status = 'pending'
           ORDER BY c.created_at ASC
           LIMIT 1
           FOR UPDATE SKIP LOCKED""",
        campaign_id,
    )


async def count_remaining_contacts(
    db: asyncpg.Connection,
    campaign_id: uuid.UUID,
) -> int:
    """Count remaining pending/in-progress contacts for a campaign."""
    return await db.fetchval(
        """SELECT COUNT(*) FROM contacts c
           JOIN campaigns camp ON c.list_id = camp.list_id
           WHERE camp.campaign_id = $1
             AND c.status IN ('pending', 'in_progress')""",
        campaign_id,
    ) or 0


async def mark_contact_in_progress(
    db: asyncpg.Connection,
    contact_id: uuid.UUID,
) -> None:
    """Mark a contact as in-progress."""
    await db.execute(
        "UPDATE contacts SET status='in_progress', started_at=CURRENT_TIMESTAMP "
        "WHERE contact_id=$1",
        contact_id,
    )


async def mark_contact_completed(
    db: asyncpg.Connection,
    contact_id: uuid.UUID,
    call_sid: str,
) -> None:
    """Mark a contact as completed with call SID."""
    await db.execute(
        "UPDATE contacts SET status='completed', call_sid=$2, "
        "processed_at=CURRENT_TIMESTAMP WHERE contact_id=$1",
        contact_id,
        call_sid,
    )


async def mark_contact_failed(
    db: asyncpg.Connection,
    contact_id: uuid.UUID,
    error_message: str,
) -> None:
    """Mark a contact as failed."""
    await db.execute(
        "UPDATE contacts SET status='failed', error_message=$2, "
        "processed_at=CURRENT_TIMESTAMP WHERE contact_id=$1",
        contact_id,
        error_message,
    )


async def get_campaign_rotation_info(
    db: asyncpg.Connection,
    campaign_id: uuid.UUID,
) -> asyncpg.Record | None:
    """Get completed_calls and sender_phone_numbers for number rotation."""
    return await db.fetchrow(
        "SELECT completed_calls, sender_phone_numbers FROM campaigns "
        "WHERE campaign_id=$1",
        campaign_id,
    )


async def increment_campaign_completed_calls(
    db: asyncpg.Connection,
    campaign_id: uuid.UUID,
) -> None:
    """Increment the completed_calls counter."""
    await db.execute(
        "UPDATE campaigns SET completed_calls=completed_calls+1 "
        "WHERE campaign_id=$1",
        campaign_id,
    )


async def increment_campaign_failed_calls(
    db: asyncpg.Connection,
    campaign_id: uuid.UUID,
) -> None:
    """Increment the failed_calls counter."""
    await db.execute(
        "UPDATE campaigns SET failed_calls=failed_calls+1 "
        "WHERE campaign_id=$1",
        campaign_id,
    )


# ── Webhook enrichment queries ────────────────────────────────────────────────


async def get_campaign_with_list_by_list_id(
    db: asyncpg.Connection,
    list_id: uuid.UUID,
    account_id: uuid.UUID,
) -> asyncpg.Record | None:
    """Get campaign with contact list name, found by contact's list_id."""
    return await db.fetchrow(
        """SELECT camp.*, cl.name AS list_name, cl.description AS list_description
           FROM campaigns camp
           LEFT JOIN contact_lists cl ON camp.list_id = cl.list_id
           WHERE camp.list_id = $1 AND camp.account_id = $2
           ORDER BY camp.created_at DESC LIMIT 1""",
        list_id,
        account_id,
    )


async def get_campaign_with_list_by_phone(
    db: asyncpg.Connection,
    phone_number: str,
    account_id: uuid.UUID,
) -> asyncpg.Record | None:
    """Get campaign with contact list name, found by sender phone number."""
    return await db.fetchrow(
        """SELECT camp.*, cl.name AS list_name, cl.description AS list_description
           FROM campaigns camp
           LEFT JOIN contact_lists cl ON camp.list_id = cl.list_id
           WHERE camp.sender_phone_number = $1 AND camp.account_id = $2
           ORDER BY camp.created_at DESC LIMIT 1""",
        phone_number,
        account_id,
    )
