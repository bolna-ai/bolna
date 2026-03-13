"""
Agent CRUD routes.

Routes (exact paths matching main_server.py):
  POST   /agent                             — create agent (auth)
  GET    /agents                            — list all agents (auth, plural prefix)
  GET    /agent/{agent_id}                  — get agent detail (auth)
  PUT    /agent/{agent_id}                  — update agent (auth)
  DELETE /agent/{agent_id}                  — soft-delete agent (auth)

Two routers are exported:
  ``router``              prefix="/agent"   (CRUD on a single agent)
  ``agents_list_router``  prefix="/agents"  (list endpoint — plural)

Both must be included in app/main.py.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Optional

import asyncpg
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.config import get_settings
from app.dependencies import DBDep, OrgDep, RedisDep
from db.queries.agents import (
    create_agent,
    get_agent,
    get_agents_by_inbound_phone,
    list_agents,
    soft_delete_agent,
    update_agent,
)
from db.queries.organizations import get_org_by_id

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/agent", tags=["agents"])
agents_list_router = APIRouter(prefix="/agents", tags=["agents"])


# ── Request / response models ─────────────────────────────────────────────────

class AgentCreateRequest(BaseModel):
    name: str
    agent_type: str = "contextual_conversation"
    call_direction: str = "outbound"
    inbound_phone_number: Optional[str] = None
    timezone: str = "America/Los_Angeles"
    country: str = "US"
    agent_image: Optional[str] = None
    webhook_url: Optional[str] = None
    agent_config: Optional[dict] = None
    agent_prompts: Optional[dict] = None
    is_compliant: bool = False


class AgentUpdateRequest(BaseModel):
    name: Optional[str] = None
    agent_type: Optional[str] = None
    call_direction: Optional[str] = None
    inbound_phone_number: Optional[str] = None
    timezone: Optional[str] = None
    country: Optional[str] = None
    agent_image: Optional[str] = None
    webhook_url: Optional[str] = None
    agent_config: Optional[dict] = None
    agent_prompts: Optional[dict] = None
    is_compliant: Optional[bool] = None


# ── Private helpers ───────────────────────────────────────────────────────────

async def _check_org_exists(db: asyncpg.Connection, account_id: uuid.UUID) -> None:
    """Raise 404 if no Organization row links to this account_id."""
    row = await db.fetchrow(
        'SELECT id FROM "Organization" WHERE "accountId"::uuid = $1',
        account_id,
    )
    if not row:
        raise HTTPException(
            status_code=404,
            detail="Organization not found for this account",
        )


async def _configure_inbound_twilio(
    db: asyncpg.Connection,
    account_id: uuid.UUID,
    phone_number: str,
) -> bool:
    """
    Point the Twilio inbound phone number's voice URL at this server.

    Uses the sub-account credentials stored in the accounts table.
    Returns True when at least one number was updated, False otherwise.
    """
    try:
        full_row = await get_org_by_id(db, str(account_id))
        if not full_row:
            return False

        twilio_sid = full_row["twilio_subaccount_sid"]
        twilio_auth = full_row["twilio_subaccount_auth_token"]
        if not (twilio_sid and twilio_auth):
            return False

        from twilio.rest import Client  # lazy import

        sub_client = Client(twilio_sid, twilio_auth)
        callback_url = f"{settings.base_url}/call/inbound"
        incoming = sub_client.incoming_phone_numbers.list(phone_number=phone_number)
        for number in incoming:
            number.update(voice_url=callback_url, voice_method="POST")
        return bool(incoming)

    except Exception as exc:
        logger.warning(
            "Twilio inbound config failed for %s: %s", phone_number, exc
        )
        return False


# ── POST /agent ───────────────────────────────────────────────────────────────

@router.post("")
async def create_agent_route(
    body: AgentCreateRequest,
    org: OrgDep,
    db: DBDep,
    redis: RedisDep,
) -> dict:
    """Create a new agent, configure Redis, and optionally set up Twilio inbound."""
    await _check_org_exists(db, org)

    is_inbound = body.call_direction == "inbound"

    # Outbound agents must not carry an inbound phone number.
    if not is_inbound and body.inbound_phone_number:
        raise HTTPException(
            status_code=400,
            detail="inbound_phone_number can only be set for inbound agents",
        )

    # Ensure inbound phone is not already claimed by another agent.
    inbound_configured = False
    callback_configured = False

    if is_inbound and body.inbound_phone_number:
        conflicts = await get_agents_by_inbound_phone(db, body.inbound_phone_number)
        if conflicts:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Phone number {body.inbound_phone_number} is already "
                    "assigned to another agent"
                ),
            )

    agent_config = body.agent_config or {}
    agent_prompts = body.agent_prompts or {}

    agent_id: uuid.UUID = await create_agent(
        db,
        account_id=org,
        name=body.name,
        agent_type=body.agent_type,
        call_direction=body.call_direction,
        inbound_phone_number=body.inbound_phone_number,
        timezone=body.timezone,
        country=body.country,
        agent_image=body.agent_image,
        webhook_url=body.webhook_url,
        agent_config=agent_config,
        agent_prompts=agent_prompts,
        is_compliant=body.is_compliant,
    )

    # Cache agent config in Redis (bare UUID key).
    try:
        await redis.set(str(agent_id), json.dumps(agent_config))
    except Exception as exc:
        logger.warning("Redis SET agent config failed for %s: %s", agent_id, exc)

    # If inbound: cache phone→agent mapping and wire up Twilio voice URL.
    if is_inbound and body.inbound_phone_number:
        try:
            await redis.set(
                f"inbound_mapping:{body.inbound_phone_number}",
                str(agent_id),
            )
            inbound_configured = True
        except Exception as exc:
            logger.warning("Redis inbound_mapping SET failed: %s", exc)

        callback_configured = await _configure_inbound_twilio(
            db, org, body.inbound_phone_number
        )

    return {
        "agent_id": str(agent_id),
        "message": "Agent created successfully",
        "state": "created",
        "inbound_configured": inbound_configured,
        "callback_configured": callback_configured,
        "ready_for_inbound_calls": (
            inbound_configured and callback_configured if is_inbound else None
        ),
        "ready_for_outbound_calls": True,
    }


# ── GET /agents (plural) ──────────────────────────────────────────────────────

@agents_list_router.get("")
async def list_agents_route(org: OrgDep, db: DBDep) -> dict:
    """List all non-deleted agents for the authenticated org."""
    await _check_org_exists(db, org)
    rows = await list_agents(db, org)
    agents: list[dict] = []
    for row in rows or []:
        a = dict(row)
        a["agent_id"] = str(a["agent_id"])
        a["account_id"] = str(a["account_id"])
        agents.append(a)
    return {"agents": agents}


# ── GET /agent/{agent_id} ─────────────────────────────────────────────────────

@router.get("/{agent_id}")
async def get_agent_route(agent_id: str, org: OrgDep, db: DBDep) -> dict:
    """Return full agent detail (left-joined with knowledgebase info)."""
    try:
        agent_uuid = uuid.UUID(agent_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid agent_id format")

    row = await get_agent(db, agent_uuid, org)
    if not row:
        raise HTTPException(status_code=404, detail="Agent not found")

    result = dict(row)
    result["agent_id"] = str(result["agent_id"])
    result["account_id"] = str(result["account_id"])
    return result


# ── PUT /agent/{agent_id} ─────────────────────────────────────────────────────

@router.put("/{agent_id}")
async def update_agent_route(
    agent_id: str,
    body: AgentUpdateRequest,
    org: OrgDep,
    db: DBDep,
    redis: RedisDep,
) -> dict:
    """Update an agent and sync Redis / Twilio."""
    await _check_org_exists(db, org)

    try:
        agent_uuid = uuid.UUID(agent_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid agent_id format")

    row = await update_agent(
        db,
        agent_id=agent_uuid,
        account_id=org,
        agent_config=body.agent_config,
        agent_prompts=body.agent_prompts,
        name=body.name,
        agent_type=body.agent_type,
        call_direction=body.call_direction,
        inbound_phone_number=body.inbound_phone_number,
        timezone=body.timezone,
        country=body.country,
        agent_image=body.agent_image,
        webhook_url=body.webhook_url,
        is_compliant=body.is_compliant,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Agent not found")

    updated = dict(row)
    agent_config = updated.get("agent_config") or {}
    inbound_phone = updated.get("inbound_phone_number")
    is_inbound = updated.get("call_direction", "outbound") == "inbound"

    callback_configured = False
    ready_for_inbound: bool | None = None
    warning: str | None = None

    # Sync Redis bare-key cache.
    try:
        await redis.set(str(agent_uuid), json.dumps(agent_config))
    except Exception as exc:
        warning = f"Redis sync failed: {exc}"
        logger.warning("Redis SET failed for agent %s: %s", agent_id, exc)

    # Optionally persist conversation details to object storage (best-effort).
    try:
        from bolna.helpers.utils import store_file  # type: ignore[import]

        await store_file(
            f"{agent_id}/conversation_details.json",
            json.dumps(updated),
        )
    except Exception:
        pass  # store_file is not critical

    # Sync inbound Redis mapping + Twilio webhooks.
    if is_inbound and inbound_phone:
        try:
            await redis.set(f"inbound_mapping:{inbound_phone}", str(agent_uuid))
        except Exception as exc:
            logger.warning("Redis inbound_mapping SET failed: %s", exc)

        callback_configured = await _configure_inbound_twilio(
            db, org, inbound_phone
        )
        ready_for_inbound = callback_configured

    return {
        "status": "success",
        "message": "Agent updated successfully",
        "agent_id": str(agent_uuid),
        "callback_configured": callback_configured,
        "ready_for_inbound_calls": ready_for_inbound,
        "ready_for_outbound_calls": True,
        "warning": warning,
    }


# ── DELETE /agent/{agent_id} ──────────────────────────────────────────────────

@router.delete("/{agent_id}")
async def delete_agent_route(
    agent_id: str,
    org: OrgDep,
    db: DBDep,
    redis: RedisDep,
) -> dict:
    """Soft-delete an agent and clean up its Redis keys."""
    try:
        agent_uuid = uuid.UUID(agent_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid agent_id format")

    row = await soft_delete_agent(db, agent_uuid, org)
    if not row:
        raise HTTPException(status_code=404, detail="Agent not found")

    inbound_phone = row.get("inbound_phone_number")

    # Clean up Redis — both the agent config blob and any inbound mapping.
    try:
        await redis.delete(str(agent_uuid))
        if inbound_phone:
            await redis.delete(f"inbound_mapping:{inbound_phone}")
    except Exception as exc:
        logger.warning("Redis cleanup failed for agent %s: %s", agent_id, exc)

    return {
        "status": "success",
        "message": "Agent deleted successfully",
        "agent_id": str(agent_uuid),
    }
