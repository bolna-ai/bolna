"""
services/campaign_service.py -- campaign lifecycle and batch calling.

Ported from main_server.py process_campaign() (~lines 5001-5060),
process_single_contact() (~lines 5087-5111), pause_campaign() (~lines 5397-5435),
and resume_campaign() (~lines 5438-5494).

Campaign flow:
  1. Load contacts from the campaign's contact list
  2. Set campaign status to "running"
  3. Iterate contacts with a semaphore-limited concurrency window
  4. For each contact: resolve per-contact template variables, call initiate_outbound_call
  5. Track dialed/answered/failed counters
  6. Mark campaign completed/failed when all contacts are processed
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid

import asyncpg
from redis.asyncio import Redis

from db.queries.campaigns import (
    get_campaign,
    increment_campaign_dialed,
    update_campaign_status,
    update_campaign_total_contacts,
)
from db.queries.contacts import get_campaign_contacts
from services.call_service import initiate_outbound_call

logger = logging.getLogger(__name__)

# Maximum concurrent calls a campaign will make at one time
DEFAULT_CAMPAIGN_CONCURRENCY = 3
# Redis key TTL for pause flag
_PAUSE_FLAG_TTL = 86400  # 24 hours
# Delay between polling for pause flag within the campaign loop
_PAUSE_CHECK_INTERVAL = 1.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def run_campaign(
    db: asyncpg.Connection,
    redis: Redis,
    campaign_id: str,
    org_id: str,
) -> None:
    """Batch-calling loop: iterate over all campaign contacts, calling each
    with the configured concurrency limit.

    This is designed to run as an ``asyncio.Task`` kicked off by the route
    handler.  It manages its own error handling so the task never raises
    into the event loop uncaught.

    Steps:
    1.  Load campaign row and its contacts from the DB.
    2.  Transition status -> ``running``.
    3.  Use an ``asyncio.Semaphore`` to cap concurrency.
    4.  For each contact:
        a.  Check for pause flag in Redis.
        b.  Resolve per-contact phone number (rotation).
        c.  Call ``initiate_outbound_call`` from ``call_service``.
        d.  Increment dialed counter.
        e.  On failure mark the contact as failed.
    5.  When all contacts are processed, transition to ``completed``
        or ``failed`` depending on the result counts.
    """
    cid = uuid.UUID(campaign_id) if isinstance(campaign_id, str) else campaign_id
    cid_str = str(cid)

    try:
        # -- Load campaign --------------------------------------------------
        campaign = await get_campaign(db, campaign_id, org_id)
        if not campaign:
            logger.error("Campaign %s not found for org %s", campaign_id, org_id)
            return

        agent_id = str(campaign["agent_id"])
        contact_list_id = str(campaign["contact_list_id"])
        concurrency = campaign.get("concurrent_calls") or DEFAULT_CAMPAIGN_CONCURRENCY
        phone_numbers = _parse_phone_numbers(campaign)
        delay_between = campaign.get("delay_between_calls", 1)

        # -- Load contacts --------------------------------------------------
        contacts = await get_campaign_contacts(db, contact_list_id)
        total = len(contacts)
        if total == 0:
            logger.warning("Campaign %s has no contacts", campaign_id)
            await update_campaign_status(db, campaign_id, org_id, "completed")
            return

        await update_campaign_total_contacts(db, campaign_id, total)

        # -- Transition to running ------------------------------------------
        await update_campaign_status(db, campaign_id, org_id, "running")
        logger.info(
            "Campaign %s started: %d contacts, concurrency=%d",
            campaign_id, total, concurrency,
        )

        # -- Semaphore-limited calling loop ---------------------------------
        semaphore = asyncio.Semaphore(concurrency)
        completed_count = 0
        failed_count = 0

        async def _call_contact(contact: asyncpg.Record, phone_idx: int) -> None:
            nonlocal completed_count, failed_count
            async with semaphore:
                # Check for pause signal before each call
                paused = await redis.get(f"campaign_pause:{cid_str}")
                if paused:
                    logger.info("Campaign %s paused, skipping contact %s", campaign_id, contact["id"])
                    return

                current_phone = phone_numbers[phone_idx % len(phone_numbers)]
                contact_vars = _extract_template_variables(contact)

                try:
                    result = await initiate_outbound_call(
                        db=db,
                        redis=redis,
                        org_id=org_id,
                        agent_id=agent_id,
                        to_number=contact["phone_number"],
                        from_number=current_phone,
                        record=True,
                        template_variables=contact_vars,
                        campaign_id=campaign_id,
                    )
                    await increment_campaign_dialed(db, campaign_id)
                    completed_count += 1
                    logger.info(
                        "Campaign %s dialed contact %s -> call_sid=%s",
                        campaign_id, contact["id"], result.get("call_sid"),
                    )
                except Exception as exc:
                    failed_count += 1
                    logger.error(
                        "Campaign %s failed to call contact %s: %s",
                        campaign_id, contact["id"], exc,
                    )
                    await _mark_contact_failed(db, contact["id"], str(exc), campaign_id)

                # Inter-call delay for rate limiting
                if delay_between > 0:
                    await asyncio.sleep(delay_between)

        # Determine phone rotation (round-robin based on index)
        rotate_after = campaign.get("rotate_numbers_after", 10) or 10
        tasks: list[asyncio.Task] = []
        for idx, contact in enumerate(contacts):
            # Re-check pause between scheduling batches
            paused = await redis.get(f"campaign_pause:{cid_str}")
            if paused:
                logger.info("Campaign %s paused mid-batch at index %d", campaign_id, idx)
                break

            phone_idx = (idx // rotate_after) % len(phone_numbers)
            task = asyncio.create_task(_call_contact(contact, phone_idx))
            tasks.append(task)

        # Wait for all in-flight calls to complete
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        # -- Finalise campaign status ---------------------------------------
        final_paused = await redis.get(f"campaign_pause:{cid_str}")
        if final_paused:
            await update_campaign_status(db, campaign_id, org_id, "paused")
            logger.info("Campaign %s ended in paused state", campaign_id)
        elif failed_count == total:
            await update_campaign_status(db, campaign_id, org_id, "failed")
            logger.error("Campaign %s failed: all %d contacts failed", campaign_id, total)
        else:
            await update_campaign_status(db, campaign_id, org_id, "completed")
            logger.info(
                "Campaign %s completed: %d ok, %d failed out of %d",
                campaign_id, completed_count, failed_count, total,
            )

    except Exception:
        logger.exception("Unhandled error in campaign %s", campaign_id)
        try:
            await update_campaign_status(db, campaign_id, org_id, "failed")
        except Exception:
            logger.exception("Failed to mark campaign %s as failed", campaign_id)


async def pause_campaign(
    db: asyncpg.Connection,
    redis: Redis,
    campaign_id: str,
    org_id: str,
) -> dict:
    """Set campaign status to paused; signal the running worker to stop.

    The running ``run_campaign`` task checks the Redis pause flag before
    each call.  Active in-flight calls will complete, but no new calls
    are initiated.
    """
    # Validate campaign exists and belongs to account
    campaign = await get_campaign(db, campaign_id, org_id)
    if not campaign:
        raise ValueError(f"Campaign {campaign_id} not found for org {org_id}")

    # Set pause flag in Redis (checked by campaign loop)
    await redis.set(f"campaign_pause:{campaign_id}", "1", ex=_PAUSE_FLAG_TTL)

    # Update DB status
    await update_campaign_status(db, campaign_id, org_id, "paused")

    logger.info("Campaign %s paused by org %s", campaign_id, org_id)
    return {
        "status": "paused",
        "message": "Campaign paused. Active calls will complete but no new calls will be initiated.",
        "campaign_id": campaign_id,
    }


async def resume_campaign(
    db: asyncpg.Connection,
    redis: Redis,
    campaign_id: str,
    org_id: str,
) -> dict:
    """Clear the pause flag and restart the campaign loop.

    If the campaign's scheduled time has already passed (overdue), processing
    starts immediately.  Otherwise the campaign status is set back to
    ``pending`` for the scheduler to pick up.
    """
    # Validate
    campaign = await get_campaign(db, campaign_id, org_id)
    if not campaign:
        raise ValueError(f"Campaign {campaign_id} not found for org {org_id}")
    if campaign["status"] != "paused":
        raise ValueError(f"Campaign {campaign_id} is not paused (current status: {campaign['status']})")

    # Clear pause flag
    await redis.delete(f"campaign_pause:{campaign_id}")

    # Determine new status
    from datetime import datetime
    from datetime import timezone as tz
    now = datetime.now(tz.utc)
    scheduled = campaign.get("scheduled_at")
    if scheduled and scheduled > now:
        new_status = "pending"
    else:
        new_status = "running"

    await update_campaign_status(db, campaign_id, org_id, new_status)

    logger.info("Campaign %s resumed -> %s by org %s", campaign_id, new_status, org_id)

    # If overdue/running, restart the processing loop
    if new_status == "running":
        asyncio.create_task(run_campaign(db, redis, campaign_id, org_id))

    return {
        "status": new_status,
        "message": f"Campaign resumed successfully. New status: {new_status}",
        "campaign_id": campaign_id,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_phone_numbers(campaign: asyncpg.Record | dict) -> list[str]:
    """Extract the list of sender phone numbers from campaign data.

    The ``phone_numbers`` column may be a JSON string, a Python list, or
    a single string stored in ``sender_phone_number``.
    """
    raw = campaign.get("phone_numbers") or campaign.get("sender_phone_numbers")
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list) and parsed:
                return parsed
        except (json.JSONDecodeError, TypeError):
            return [raw]
    if isinstance(raw, list) and raw:
        return raw
    # Fallback to single phone number field
    single = campaign.get("sender_phone_number")
    if single:
        return [single]
    raise ValueError(f"No sender phone numbers configured for campaign {campaign.get('id')}")


def _extract_template_variables(contact: asyncpg.Record | dict) -> dict:
    """Pull per-contact template variables from the contact row.

    The column ``template_variables`` may be stored as a JSON string or
    as a native ``jsonb`` dict.
    """
    raw = contact.get("template_variables")
    if raw is None:
        return {}
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {}
    if isinstance(raw, dict):
        return raw
    return {}


async def _mark_contact_failed(
    db: asyncpg.Connection,
    contact_id: str | uuid.UUID,
    error_message: str,
    campaign_id: str,
) -> None:
    """Update a contact's status to failed and increment the campaign's
    failed counter."""
    cid = uuid.UUID(contact_id) if isinstance(contact_id, str) else contact_id
    camp_id = uuid.UUID(campaign_id) if isinstance(campaign_id, str) else campaign_id
    try:
        async with db.transaction():
            await db.execute(
                """UPDATE contacts
                   SET status = 'failed', error_message = $1, processed_at = NOW()
                   WHERE id = $2""",
                error_message,
                cid,
            )
            await db.execute(
                """UPDATE campaigns
                   SET failed_calls = COALESCE(failed_calls, 0) + 1, updated_at = NOW()
                   WHERE id = $1""",
                camp_id,
            )
    except Exception:
        logger.exception("Failed to mark contact %s as failed", contact_id)
