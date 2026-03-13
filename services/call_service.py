"""
services/call_service.py -- outbound call orchestration and call lifecycle.

Ported from main_server.py POST /call handler (~lines 300-420)
and POST /call/end handler (~lines 4270-4310).

Keeps the route handler thin; all business logic lives here.
"""

from __future__ import annotations

import json
import logging
import uuid

import asyncpg
from redis.asyncio import Redis
from twilio.base.exceptions import TwilioRestException

from app.config import get_settings
from db.queries.calls import create_call_record, update_call_status
from services.agent_service import get_agent_config
from services.twilio_helpers import get_callback_urls, get_twilio_client

logger = logging.getLogger(__name__)
settings = get_settings()

# Cache TTL for template variables stored in Redis
_TEMPLATE_VAR_TTL = 3600  # 1 hour
# Cache TTL for compliance/temp-agent flags
_FLAG_TTL = 86400  # 24 hours

# Re-export for backward compat (anything importing from call_service)
_get_callback_urls = get_callback_urls


# ---------------------------------------------------------------------------
# Outbound call flow
# ---------------------------------------------------------------------------

async def initiate_outbound_call(
    db: asyncpg.Connection,
    redis: Redis,
    org_id: str,
    agent_id: str,
    to_number: str,
    from_number: str,
    record: bool = True,
    template_variables: dict | None = None,
    campaign_id: str | None = None,
    agent_config_override: dict | None = None,
) -> dict:
    """Full outbound call flow.

    1. Load agent config (Redis cache -> DB fallback)
    2. Determine compliance flag (is_compliant -> never record)
    3. Store template variables in Redis for prompt resolution
    4. Build call context dict
    5. Initiate the Twilio call
    6. Store call record in DB
    7. Return ``{"call_sid": ..., "call_id": ...}``

    Parameters
    ----------
    db : asyncpg.Connection
        Checked-out database connection.
    redis : Redis
        Shared Redis client.
    org_id : str
        Account / organisation UUID.
    agent_id : str
        Agent UUID.
    to_number : str
        E.164 recipient number.
    from_number : str
        E.164 sender number (must belong to the Twilio sub-account).
    record : bool
        Whether Twilio should record the call.  Overridden to ``False``
        for compliant agents.
    template_variables : dict | None
        Per-call template variables injected into the system prompt.
    campaign_id : str | None
        When set the call is part of a campaign batch.
    agent_config_override : dict | None
        Temporary / inline agent config (skips DB lookup).  Used by the
        "quick call" API endpoint.
    """
    account_id = uuid.UUID(org_id) if isinstance(org_id, str) else org_id

    # 1. Load agent config ------------------------------------------------
    is_temp_agent = agent_config_override is not None
    if is_temp_agent:
        agent_cfg = agent_config_override
        is_compliant = agent_cfg.get("is_compliant", False)
    else:
        agent_cfg = await get_agent_config(db, redis, org_id, agent_id)
        if agent_cfg is None:
            raise ValueError(f"Agent {agent_id} not found for org {org_id}")
        is_compliant = agent_cfg.get("is_compliant", False)

    # 2. Compliance override: never record for compliant agents -----------
    effective_record = False if is_compliant else record

    # 3. Template variables -----------------------------------------------
    if template_variables:
        try:
            tv_json = json.dumps(template_variables)
            await redis.set(
                f"template_variables:{agent_id}",
                tv_json,
                ex=_TEMPLATE_VAR_TTL,
            )
        except (TypeError, json.JSONDecodeError) as exc:
            logger.error("Invalid template variables for agent %s: %s", agent_id, exc)
            raise ValueError(f"Invalid template variables format: {exc}") from exc

    # 4. Twilio call setup ------------------------------------------------
    twilio_client = await get_twilio_client(db, account_id)
    app_callback_url, websocket_url = _get_callback_urls()

    recording_status_callback = (
        f"{app_callback_url}/recording_callback" if effective_record else None
    )

    try:
        call = twilio_client.calls.create(
            to=to_number,
            from_=from_number,
            url=(
                f"{app_callback_url}/twilio_callback"
                f"?ws_url={websocket_url}"
                f"&agent_id={agent_id}"
                f"&account_id={org_id}"
            ),
            method="POST",
            record=effective_record,
            recording_status_callback=recording_status_callback,
            recording_status_callback_event=["completed"] if effective_record else None,
            status_callback=f"{app_callback_url}/call_status_callback",
            status_callback_event=[
                "initiated", "ringing", "answered", "completed",
            ],
            status_callback_method="POST",
            timeout=30,
        )
    except TwilioRestException as exc:
        logger.error("Twilio error initiating call to %s: %s", to_number, exc)
        raise

    call_sid = call.sid
    logger.info(
        "Outbound call initiated call_sid=%s agent=%s to=%s record=%s",
        call_sid, agent_id, to_number, effective_record,
    )

    # 5. Store Redis flags ------------------------------------------------
    if is_compliant:
        await redis.set(f"agent_compliant:{call_sid}", "1", ex=_FLAG_TTL)
    if is_temp_agent:
        await redis.set(f"is_temp_agent:{agent_id}", "1", ex=_FLAG_TTL)
        await redis.set(f"temp_agent_mapping:{call_sid}", agent_id, ex=_FLAG_TTL)

    # 6. Database record --------------------------------------------------
    async with db.transaction():
        call_record = await create_call_record(
            db,
            org_id=org_id,
            agent_id=agent_id,
            call_sid=call_sid,
            direction="outbound",
            from_number=from_number,
            to_number=to_number,
            campaign_id=campaign_id,
        )
        if not is_temp_agent:
            await db.execute(
                "UPDATE agents SET total_calls = total_calls + 1 WHERE agent_id = $1",
                uuid.UUID(agent_id),
            )
        await db.execute(
            "UPDATE accounts SET total_calls = total_calls + 1 WHERE account_id = $1",
            account_id,
        )

    return {
        "call_sid": call_sid,
        "call_id": str(call_record["id"]) if call_record else None,
        "status": "initiated",
    }


# ---------------------------------------------------------------------------
# Force-end call
# ---------------------------------------------------------------------------

async def end_call(
    db: asyncpg.Connection,
    call_sid: str,
    org_id: str,
) -> dict:
    """Force-end an active call via the Twilio API and update the DB.

    Parameters
    ----------
    db : asyncpg.Connection
        Checked-out database connection.
    call_sid : str
        Twilio Call SID to terminate.
    org_id : str
        Account / organisation UUID (used to look up Twilio credentials).

    Returns
    -------
    dict
        ``{"status": "success", "call_sid": ..., "message": ...}``

    Raises
    ------
    ValueError
        When the account has no Twilio credentials.
    TwilioRestException
        When Twilio rejects the update (invalid SID, call already ended, etc.)
    """
    account_id = uuid.UUID(org_id) if isinstance(org_id, str) else org_id

    twilio_client = await get_twilio_client(db, account_id)

    try:
        twilio_client.calls(call_sid).update(status="completed")
    except TwilioRestException as exc:
        logger.error("Twilio error ending call %s: %s", call_sid, exc)
        raise

    # Update DB status
    await update_call_status(db, call_sid, "completed")

    logger.info("Call %s force-ended for org %s", call_sid, org_id)
    return {
        "status": "success",
        "message": "Call ended successfully",
        "call_sid": call_sid,
    }
