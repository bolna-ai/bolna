"""
Campaign management and execution routes.

Routes:
  POST   /campaigns         -- create campaign
  POST   /campaigns/pause   -- pause a running campaign
  POST   /campaigns/resume  -- resume a paused / overdue campaign
  GET    /campaigns/{id}    -- get campaign details
  DELETE /campaigns/{id}    -- delete a campaign
  PUT    /campaigns/{id}    -- edit a campaign
  GET    /campaigns         -- list all campaigns
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
from twilio.rest import Client

from app.config import get_settings
from app.dependencies import DBDep, OrgDep
from db import pool as db_pool
from db.queries.agents import get_agent_direction
from db.queries.calls import insert_campaign_call
from db.queries.campaigns import (
    count_contacts_for_list,
    count_remaining_contacts,
    create_campaign_full,
    delete_campaign_by_id,
    delete_contact_list_by_id,
    delete_contacts_by_list_id,
    get_campaign_display_info,
    get_campaign_for_account,
    get_campaign_for_delete,
    get_campaign_for_edit,
    get_campaign_processing_info,
    get_campaign_rotation_info,
    get_campaign_with_status,
    get_next_pending_contact,
    get_overdue_campaigns,
    increment_campaign_completed_calls,
    increment_campaign_failed_calls,
    list_campaigns_with_details,
    mark_campaign_completed,
    mark_campaign_failed,
    mark_campaign_overdue,
    mark_campaign_running,
    mark_contact_completed,
    mark_contact_failed,
    mark_contact_in_progress,
    update_campaign_dynamic,
    update_campaign_paused,
    update_campaign_status_with_timestamp,
)
from db.queries.organizations import get_twilio_credentials

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(tags=["campaigns"])


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class CreateCampaignPayload(BaseModel):
    name: str
    description: str = ""
    agent_id: str
    list_id: str
    sender_phone_numbers: List[str]
    scheduled_time: Optional[str] = None
    timezone: str = "UTC"
    delay_between_calls: int = 1
    rotate_numbers_after: int = 0


class PauseCampaignPayload(BaseModel):
    campaign_id: str


class ResumeCampaignPayload(BaseModel):
    campaign_id: str


class EditCampaignPayload(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    agent_id: Optional[str] = None
    list_id: Optional[str] = None
    sender_phone_numbers: Optional[List[str]] = None
    scheduled_time: Optional[str] = None
    timezone: Optional[str] = None
    delay_between_calls: Optional[int] = None
    rotate_numbers_after: Optional[int] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def get_twilio_client(conn, account_id: uuid.UUID) -> Client:
    """Return a Twilio client using the account's sub-account credentials."""
    account = await get_twilio_credentials(conn, account_id)
    if not account:
        raise HTTPException(status_code=400, detail="Twilio credentials not configured")
    return Client(account["twilio_subaccount_sid"], account["twilio_subaccount_auth_token"])


def _populate_custom_urls() -> tuple[str, str]:
    base = settings.base_url.rstrip("/")
    if "kallabot.com" in base:
        app_callback_url = "https://api.kallabot.com"
        websocket_url = "wss://ws.kallabot.com"
    else:
        app_callback_url = base
        websocket_url = base.replace("https://", "wss://").replace("http://", "ws://")
    return app_callback_url, websocket_url


# ---------------------------------------------------------------------------
# Background task helpers
# ---------------------------------------------------------------------------

async def _make_campaign_call(
    conn,
    client: Client,
    agent_id: uuid.UUID,
    account_id: uuid.UUID,
    contact_id: uuid.UUID,
    call_details: dict,
) -> None:
    """Place one Twilio call for a campaign contact."""
    app_callback_url, websocket_url = _populate_custom_urls()

    call = client.calls.create(
        to=call_details["phone_number"],
        from_=call_details["sender_phone_number"],
        url=(
            f"{app_callback_url}/twilio_callback"
            f"?ws_url={websocket_url}"
            f"&agent_id={agent_id}"
            f"&account_id={account_id}"
        ),
        method="POST",
        record=True,
        recording_status_callback=f"{app_callback_url}/recording_callback",
        recording_status_callback_event=["completed"],
        status_callback=f"{app_callback_url}/call_status_callback",
        status_callback_event=["initiated", "ringing", "answered", "completed"],
        status_callback_method="POST",
        timeout=30,
    )
    logger.info("Campaign call initiated: %s -> %s", call.sid, call_details["phone_number"])

    await insert_campaign_call(
        conn, call.sid, agent_id, account_id,
        call_details["sender_phone_number"], call_details["phone_number"],
    )
    await mark_contact_completed(conn, contact_id, call.sid)


async def _process_single_contact(
    conn,
    client: Client,
    campaign_id: uuid.UUID,
    agent_id: uuid.UUID,
    account_id: uuid.UUID,
    contact,
) -> None:
    """Process one contact within a campaign: mark in-progress, select phone, dial."""
    contact_id = contact["contact_id"]
    try:
        await mark_contact_in_progress(conn, contact_id)

        # Phone number rotation
        rotation = await get_campaign_rotation_info(conn, campaign_id)
        completed_calls = rotation["completed_calls"] if rotation else 0
        phone_numbers = json.loads(rotation["sender_phone_numbers"]) if rotation else []

        if not phone_numbers:
            raise ValueError("No sender phone numbers configured for campaign")

        sender_index = completed_calls % len(phone_numbers) if len(phone_numbers) > 1 else 0
        sender_number = phone_numbers[sender_index]

        call_details = {
            "phone_number": contact["phone_number"],
            "sender_phone_number": sender_number,
            "template_variables": contact.get("template_variables"),
        }

        await _make_campaign_call(
            conn, client, agent_id, account_id, contact_id, call_details,
        )
        await increment_campaign_completed_calls(conn, campaign_id)
        logger.info("Successfully called contact %s", contact_id)

    except Exception as e:
        logger.error("Error processing contact %s: %s", contact_id, e)
        await mark_contact_failed(conn, contact_id, str(e))
        await increment_campaign_failed_calls(conn, campaign_id)


async def _process_campaign(campaign_id: uuid.UUID) -> None:
    """Main loop: iterate over pending contacts and dial each one."""
    async with db_pool.pool.acquire() as conn:
        campaign = await get_campaign_processing_info(conn, campaign_id)
        if not campaign:
            logger.error("Campaign %s not found", campaign_id)
            return

        account_id = campaign["account_id"]
        agent_id = campaign["agent_id"]
        delay = campaign["delay_between_calls"]

        await mark_campaign_running(conn, campaign_id)
        logger.info("Campaign %s started processing", campaign_id)

        client = await get_twilio_client(conn, account_id)

        while True:
            # Check if campaign was paused externally
            campaign_state = await get_campaign_with_status(conn, campaign_id, account_id)
            if campaign_state and campaign_state["status"] == "paused":
                logger.info("Campaign %s was paused, stopping", campaign_id)
                break

            contact = await get_next_pending_contact(conn, campaign_id)
            if not contact:
                # No more pending contacts -- check if campaign is done
                remaining = await count_remaining_contacts(conn, campaign_id)
                if remaining == 0:
                    await mark_campaign_completed(conn, campaign_id)
                    logger.info("Campaign %s completed", campaign_id)
                break

            await _process_single_contact(
                conn, client, campaign_id, agent_id, account_id, contact,
            )

            if delay > 0:
                await asyncio.sleep(delay)


async def _process_campaign_with_cleanup(campaign_id: uuid.UUID) -> None:
    """Wrapper that marks the campaign failed on unexpected errors."""
    try:
        await _process_campaign(campaign_id)
    except Exception as e:
        logger.error("Campaign %s failed: %s", campaign_id, e)
        try:
            async with db_pool.pool.acquire() as conn:
                await mark_campaign_failed(conn, campaign_id)
        except Exception as db_err:
            logger.error(
                "Failed to mark campaign %s as failed: %s", campaign_id, db_err,
            )


# ---------------------------------------------------------------------------
# Overdue campaign checker (called on startup or cron)
# ---------------------------------------------------------------------------

async def _check_overdue_campaigns() -> None:
    """
    Check for campaigns that should have started but haven't.
    Called periodically by a background scheduler.
    """
    try:
        async with db_pool.pool.acquire() as conn:
            overdue = await get_overdue_campaigns(conn)
            for row in overdue:
                cid = row["campaign_id"]
                try:
                    await mark_campaign_overdue(conn, cid)
                    logger.info("Campaign %s marked as overdue", cid)
                except Exception as e:
                    logger.error("Error marking campaign %s overdue: %s", cid, e)
    except Exception as e:
        logger.error("Error checking overdue campaigns: %s", e)


# ---------------------------------------------------------------------------
# POST /campaigns -- create campaign
# ---------------------------------------------------------------------------

@router.post("/campaigns")
async def create_campaign_route(
    payload: CreateCampaignPayload,
    org: OrgDep,
    db: DBDep,
    background_tasks: BackgroundTasks,
) -> dict:
    """Create a new campaign and optionally start it immediately."""
    try:
        agent = await get_agent_direction(db, uuid.UUID(payload.agent_id), org)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        if agent["call_direction"] != "outbound":
            raise HTTPException(
                status_code=400,
                detail="Agent must be configured for outbound calls",
            )

        contact_count = await count_contacts_for_list(
            db, uuid.UUID(payload.list_id), org,
        )
        if contact_count == 0:
            raise HTTPException(
                status_code=400,
                detail="Contact list is empty. Add contacts before creating a campaign.",
            )

        # Determine status and scheduling
        now = datetime.now(timezone.utc)
        scheduled_time: datetime | None = None
        status = "pending"

        if payload.scheduled_time:
            scheduled_time = datetime.fromisoformat(
                payload.scheduled_time.replace("Z", "+00:00")
            )
            if scheduled_time <= now:
                status = "pending"  # will start immediately via background task
        else:
            status = "pending"

        campaign_id = await create_campaign_full(
            db,
            account_id=org,
            agent_id=uuid.UUID(payload.agent_id),
            list_id=uuid.UUID(payload.list_id),
            name=payload.name,
            description=payload.description,
            status=status,
            sender_phone_numbers=json.dumps(payload.sender_phone_numbers),
            total_contacts=contact_count,
            delay_between_calls=payload.delay_between_calls,
            scheduled_time=scheduled_time or now,
            timezone=payload.timezone,
            rotate_numbers_after=payload.rotate_numbers_after,
        )

        # Start immediately if not future-scheduled
        if not payload.scheduled_time or (scheduled_time and scheduled_time <= now):
            background_tasks.add_task(
                _process_campaign_with_cleanup, campaign_id,
            )

        return {
            "status": "success",
            "campaign_id": str(campaign_id),
            "campaign_status": status,
            "total_contacts": contact_count,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error creating campaign: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to create campaign: {e}")


# ---------------------------------------------------------------------------
# POST /campaigns/pause
# ---------------------------------------------------------------------------

@router.post("/campaigns/pause")
async def pause_campaign(
    payload: PauseCampaignPayload,
    org: OrgDep,
    db: DBDep,
) -> dict:
    """Pause a running campaign."""
    try:
        campaign_id = uuid.UUID(payload.campaign_id)
        campaign = await get_campaign_for_account(db, campaign_id, org)
        if not campaign:
            raise HTTPException(status_code=404, detail="Campaign not found")

        await update_campaign_paused(db, campaign_id)

        return {"status": "success", "message": "Campaign paused successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error pausing campaign: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to pause campaign: {e}")


# ---------------------------------------------------------------------------
# POST /campaigns/resume
# ---------------------------------------------------------------------------

@router.post("/campaigns/resume")
async def resume_campaign(
    payload: ResumeCampaignPayload,
    org: OrgDep,
    db: DBDep,
    background_tasks: BackgroundTasks,
) -> dict:
    """Resume a paused or overdue campaign."""
    try:
        campaign_id = uuid.UUID(payload.campaign_id)
        campaign = await get_campaign_with_status(db, campaign_id, org)
        if not campaign:
            raise HTTPException(status_code=404, detail="Campaign not found")

        if campaign["status"] not in ("paused", "overdue"):
            raise HTTPException(
                status_code=400,
                detail=f"Campaign cannot be resumed from status '{campaign['status']}'",
            )

        # Determine target status
        new_status = "pending"
        if (
            campaign["scheduled_time"]
            and campaign["scheduled_time"] > datetime.now(timezone.utc)
        ):
            new_status = "pending"
        else:
            new_status = "pending"

        await update_campaign_status_with_timestamp(db, campaign_id, new_status)

        # Start processing immediately
        background_tasks.add_task(
            _process_campaign_with_cleanup, campaign_id,
        )

        return {"status": "success", "message": "Campaign resumed successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error resuming campaign: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to resume campaign: {e}")


# ---------------------------------------------------------------------------
# GET /campaigns/{campaign_id}
# ---------------------------------------------------------------------------

@router.get("/campaigns/{campaign_id}")
async def get_campaign_route(
    campaign_id: str,
    org: OrgDep,
    db: DBDep,
) -> dict:
    """Return campaign details."""
    try:
        cid = uuid.UUID(campaign_id)
        campaign = await get_campaign_display_info(db, cid, org)
        if not campaign:
            raise HTTPException(status_code=404, detail="Campaign not found")

        return {
            "campaign_id": str(campaign["campaign_id"]),
            "name": campaign["name"],
            "status": campaign["status"],
            "total_contacts": campaign["total_contacts"],
            "completed_calls": campaign["completed_calls"],
            "failed_calls": campaign["failed_calls"],
            "pending_calls": campaign["pending_calls"],
            "created_at": campaign["created_at"].isoformat() if campaign["created_at"] else None,
            "scheduled_time": campaign["scheduled_time"].isoformat() if campaign["scheduled_time"] else None,
            "started_at": campaign["started_at"].isoformat() if campaign["started_at"] else None,
            "completed_at": campaign["completed_at"].isoformat() if campaign["completed_at"] else None,
            "timezone": campaign["timezone"],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting campaign: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to get campaign: {e}")


# ---------------------------------------------------------------------------
# DELETE /campaigns/{campaign_id}
# ---------------------------------------------------------------------------

@router.delete("/campaigns/{campaign_id}")
async def delete_campaign_route(
    campaign_id: str,
    org: OrgDep,
    db: DBDep,
) -> dict:
    """
    Delete a campaign, its contacts, and its contact list.
    Only non-running campaigns can be deleted.
    """
    try:
        cid = uuid.UUID(campaign_id)
        campaign = await get_campaign_for_delete(db, cid, org)
        if not campaign:
            raise HTTPException(status_code=404, detail="Campaign not found")

        if campaign["status"] == "running":
            raise HTTPException(
                status_code=400,
                detail="Cannot delete a running campaign. Pause it first.",
            )

        list_id = campaign["list_id"]

        # Delete in dependency order
        await delete_campaign_by_id(db, cid)

        if list_id:
            try:
                await delete_contacts_by_list_id(db, list_id)
                await delete_contact_list_by_id(db, list_id)
            except Exception as e:
                logger.error("Error cleaning up campaign resources: %s", e)

        return {"status": "success", "message": "Campaign deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error deleting campaign: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to delete campaign: {e}")


# ---------------------------------------------------------------------------
# PUT /campaigns/{campaign_id}
# ---------------------------------------------------------------------------

@router.put("/campaigns/{campaign_id}")
async def edit_campaign(
    campaign_id: str,
    payload: EditCampaignPayload,
    org: OrgDep,
    db: DBDep,
) -> dict:
    """Edit an existing campaign. Only pending/paused campaigns can be edited."""
    try:
        cid = uuid.UUID(campaign_id)
        campaign = await get_campaign_for_edit(db, cid, org)
        if not campaign:
            raise HTTPException(status_code=404, detail="Campaign not found")

        if campaign["status"] not in ("pending", "paused"):
            raise HTTPException(
                status_code=400,
                detail="Only pending or paused campaigns can be edited",
            )

        # If list_id changed, recount contacts
        if payload.list_id:
            count = await count_contacts_for_list(
                db, uuid.UUID(payload.list_id), org,
            )
            if count == 0:
                raise HTTPException(
                    status_code=400,
                    detail="New contact list is empty",
                )

        # Build dynamic update
        field_mappings: dict[str, tuple[str, Any]] = {
            "name": ("name", str),
            "description": ("description", str),
            "agent_id": ("agent_id", lambda v: uuid.UUID(v)),
            "list_id": ("list_id", lambda v: uuid.UUID(v)),
            "sender_phone_numbers": (
                "sender_phone_numbers",
                lambda v: json.dumps(v),
            ),
            "scheduled_time": (
                "scheduled_time",
                lambda v: datetime.fromisoformat(v.replace("Z", "+00:00")),
            ),
            "timezone": ("timezone", str),
            "delay_between_calls": ("delay_between_calls", int),
            "rotate_numbers_after": ("rotate_numbers_after", int),
        }

        set_clauses: list[str] = []
        values: list = []

        for field, (col, transform) in field_mappings.items():
            val = getattr(payload, field, None)
            if val is not None:
                set_clauses.append(f"{col} = ${len(values) + 1}")
                values.append(transform(val))

                # Update total_contacts if list_id changed
                if field == "list_id":
                    count = await count_contacts_for_list(
                        db, uuid.UUID(val), org,
                    )
                    set_clauses.append(f"total_contacts = ${len(values) + 1}")
                    values.append(count)

        if not set_clauses:
            return {
                "status": "success",
                "message": "No changes to apply",
            }

        await update_campaign_dynamic(db, cid, set_clauses, values)

        return {
            "status": "success",
            "message": "Campaign updated successfully",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error editing campaign: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to edit campaign: {e}")


# ---------------------------------------------------------------------------
# GET /campaigns -- list all campaigns
# ---------------------------------------------------------------------------

@router.get("/campaigns")
async def list_campaigns_route(
    org: OrgDep,
    db: DBDep,
) -> dict:
    """Return all campaigns for the authenticated account."""
    try:
        rows = await list_campaigns_with_details(db, org)

        campaigns = []
        for row in rows:
            campaigns.append({
                "campaign_id": str(row["campaign_id"]),
                "name": row["name"],
                "status": row["campaign_status"],
                "total_contacts": row["total_contacts"],
                "completed_calls": row["completed_calls"],
                "failed_calls": row["failed_calls"],
                "pending_calls": row["pending_calls"],
                "agent_name": row["agent_name"],
                "contact_list_name": row["contact_list_name"],
                "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                "scheduled_time": row["scheduled_time"].isoformat() if row["scheduled_time"] else None,
                "started_at": row["started_at"].isoformat() if row["started_at"] else None,
                "completed_at": row["completed_at"].isoformat() if row["completed_at"] else None,
                "timezone": row["timezone"],
                "delay_between_calls": row["delay_between_calls"],
                "sender_phone_numbers": row["sender_phone_numbers"],
            })

        return {"campaigns": campaigns}

    except Exception as e:
        logger.error("Error listing campaigns: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to list campaigns: {e}")
