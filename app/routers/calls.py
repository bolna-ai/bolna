"""
Call initiation, management, and retrieval routes.

main_server.py reference: lines ~1725-2442 and ~3502-4310.

Routes (exact paths from main_server.py):
  POST   /call                              -- initiate outbound call
  POST   /call/end                          -- end an active call via Twilio
  GET    /details                           -- list call logs for an account
  GET    /call-details/{call_sid}           -- get specific call details
  POST   /web-call                          -- initiate browser-based WebRTC call
  POST   /web-call-status-callback          -- web call status updates
  GET    /web-call/config/{agent_id}        -- get web call configuration
  POST   /chat-session                      -- initiate chat session
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

import aiohttp
from fastapi import APIRouter, Form, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel
from twilio.base.exceptions import TwilioRestException
from twilio.rest import Client

from app.config import get_settings
from app.dependencies import DBDep, OrgDep, RedisDep
from db.queries.agents import (
    get_agent_basic_info,
    get_agent_call_info,
    get_agent_config_info,
    increment_agent_total_calls,
    update_agent_call_stats,
)
from db.queries.calls import (
    get_call_detail_by_sid,
    get_call_with_agent_info,
    get_calls_with_agent_name,
    insert_call,
    insert_web_call,
    update_call_fields,
)
from db.queries.organizations import (
    deduct_org_minutes,
    get_org_plan_and_minutes,
    get_org_pricing,
    get_twilio_credentials,
    increment_account_total_calls,
    update_account_call_stats,
)

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(tags=["calls"])

# Plans allowed to make calls
ALLOWED_PLANS = {"free", "basic", "pro", "business", "enterprise"}


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class CallPayload(BaseModel):
    """
    Model for the call payload that can accept either agent_id or direct
    agent configuration. When agent_config is provided, it creates a
    temporary agent that persists during the call.
    """
    agent_id: Optional[str] = None
    agent_config: Optional[Dict[str, Any]] = None
    agent_prompts: Optional[Dict[str, Any]] = None
    sender_phone_number: str
    recipient_phone_number: str
    record: bool = True
    template_variables: Dict[str, Any] = {}


class EndCallPayload(BaseModel):
    call_sid: str


class WebCallPayload(BaseModel):
    agent_id: Optional[str] = None
    agent_config: Optional[Dict[str, Any]] = None
    agent_prompts: Optional[Dict[str, Any]] = None
    template_variables: Dict[str, Any] = {}
    web_call_sid: Optional[str] = None


class ChatSessionPayload(BaseModel):
    agent_id: str
    template_variables: Dict[str, Any] = {}
    chat_session_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

async def _check_org_plan_and_minutes(
    db,
    account_id: uuid.UUID,
) -> None:
    """
    Verify the organisation is on an allowed plan and has remaining minutes.
    Raises HTTPException(403) otherwise.
    """
    org = await get_org_plan_and_minutes(db, account_id)
    if not org:
        raise HTTPException(
            status_code=403,
            detail="Organization not found for this account. Cannot make calls.",
        )
    raw_plan = org["planType"] or ""
    plan = raw_plan.strip().lower()
    if plan not in ALLOWED_PLANS:
        raise HTTPException(
            status_code=403,
            detail=f"Your plan ('{raw_plan}') does not allow making calls.",
        )
    if org["minutes"] is None or org["minutes"] <= 0:
        raise HTTPException(
            status_code=403,
            detail=(
                "You do not have enough minutes to make calls. "
                "Please upgrade or purchase more minutes."
            ),
        )


async def _get_twilio_client(
    db,
    account_id: uuid.UUID,
) -> Client:
    """Build a Twilio REST client from the org's sub-account credentials."""
    account = await get_twilio_credentials(db, account_id)
    if (
        not account
        or not account["twilio_subaccount_sid"]
        or not account["twilio_subaccount_auth_token"]
    ):
        raise HTTPException(
            status_code=400,
            detail="Twilio credentials not configured for this account",
        )
    return Client(
        account["twilio_subaccount_sid"],
        account["twilio_subaccount_auth_token"],
    )


def _populate_custom_urls() -> tuple[str, str]:
    """Return (app_callback_url, websocket_url) derived from settings."""
    base = settings.base_url.rstrip("/")
    # For production the WebSocket URL uses the wss:// scheme on the ws
    # subdomain.  In development it falls back to the base URL with ws://.
    if "kallabot.com" in base:
        app_callback_url = "https://api.kallabot.com"
        websocket_url = "wss://ws.kallabot.com"
    else:
        app_callback_url = base
        websocket_url = base.replace("https://", "wss://").replace("http://", "ws://")
    return app_callback_url, websocket_url


async def _get_org_price_per_minute(db, account_id: uuid.UUID) -> float:
    """Get the organisation's price per minute from the Organization table."""
    org = await get_org_pricing(db, account_id)
    if not org:
        return 0.20  # default fallback
    if org["custom_price_per_minute"] is not None:
        return org["custom_price_per_minute"]
    return org["default_price_per_minute"]


async def _calculate_call_cost(db, duration: int, account_id: uuid.UUID) -> float:
    """Calculate call cost based on exact duration and the org's pricing."""
    price_per_minute = await _get_org_price_per_minute(db, account_id)
    duration_minutes = duration / 60.0
    return duration_minutes * price_per_minute


async def _send_webhook(url: str, payload: dict, agent_name: str) -> None:
    """Fire-and-forget webhook POST (best-effort)."""
    try:
        logger.info("Sending webhook to %s for agent %s", url, agent_name)
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                response_text = await response.text()
                if response.status >= 300:
                    logger.error(
                        "Webhook to %s failed (%s): %s",
                        url, response.status, response_text,
                    )
                else:
                    logger.info("Webhook succeeded with status %s", response.status)
    except Exception as exc:
        logger.error("Error sending webhook to %s: %s", url, exc)


# ---------------------------------------------------------------------------
# POST /call -- initiate outbound call
# ---------------------------------------------------------------------------

@router.post("/call")
async def make_call(
    payload: CallPayload,
    org: OrgDep,
    db: DBDep,
    redis: RedisDep,
) -> JSONResponse:
    """Initiate an outbound Twilio call, optionally with a temporary agent config."""
    account_id = org
    await _check_org_plan_and_minutes(db, account_id)

    try:
        agent_id = payload.agent_id
        agent_config = payload.agent_config
        agent_prompts = payload.agent_prompts

        if not agent_id and not agent_config:
            raise HTTPException(
                status_code=400,
                detail="Either agent_id or agent_config must be provided",
            )

        temp_agent_id = str(uuid.uuid4()) if not agent_id else agent_id

        # Variables to be determined
        is_compliant = False
        webhook_url: str | None = None

        # ----- Temporary agent (agent_config provided) -----
        if agent_config:
            if "agent_config" in agent_config:
                complete_config = agent_config
                if not agent_prompts and "agent_prompts" in agent_config:
                    agent_prompts = agent_config["agent_prompts"]
            else:
                complete_config = {
                    "agent_id": temp_agent_id,
                    "agent_config": agent_config,
                    "agent_prompts": agent_prompts or {},
                }

            # Store complete agent configuration in Redis (24 h TTL)
            await redis.set(
                f"temp_agent_config:{temp_agent_id}",
                json.dumps(complete_config),
                ex=86400,
            )

            if not agent_prompts:
                agent_prompts = {
                    "task_1": {
                        "system_prompt": "You are a helpful AI assistant.",
                    },
                }

            await redis.set(
                f"temp_agent_prompts:{temp_agent_id}",
                json.dumps(agent_prompts),
                ex=86400,
            )

            agent_id = temp_agent_id
            is_compliant = agent_config.get("is_compliant", False)
            webhook_url = agent_config.get("webhook_url", None)

        # ----- Persistent agent (agent_id only) -----
        else:
            agent = await get_agent_call_info(db, uuid.UUID(agent_id))
            if not agent:
                raise HTTPException(status_code=404, detail="Agent not found")

            if agent["call_direction"] == "inbound":
                raise HTTPException(
                    status_code=403,
                    detail="This agent is configured for inbound calls only",
                )

            is_compliant = agent["is_compliant"] if agent["is_compliant"] is not None else False
            webhook_url = agent["webhook_url"]

        # Fetch Twilio sub-account creds
        subaccount_info = await get_twilio_credentials(db, account_id)
        if not subaccount_info:
            raise HTTPException(status_code=404, detail="Account not found")

        sender_phone_number = payload.sender_phone_number
        recipient_phone_number = payload.recipient_phone_number
        requested_record = payload.record
        template_variables = payload.template_variables

        # Compliant agents never record
        record = False if is_compliant else requested_record

        logger.info("Received template variables: %s", template_variables)
        logger.info("Agent is_compliant: %s, recording: %s", is_compliant, record)

        # Store template variables in Redis
        try:
            json_str = json.dumps(template_variables)
            await redis.set(f"template_variables:{agent_id}", json_str, ex=3600)
        except (json.JSONDecodeError, TypeError) as je:
            logger.error("JSON serialization error: %s", je)
            raise HTTPException(
                status_code=400,
                detail=f"Invalid template variables format: {je}",
            )
        except Exception as exc:
            logger.error("Redis storage error: %s", exc)
            raise HTTPException(
                status_code=500,
                detail="Failed to store template variables",
            )

        if not agent_id or not account_id:
            raise HTTPException(
                status_code=404,
                detail="Agent ID or Account ID not provided",
            )
        if not recipient_phone_number:
            raise HTTPException(
                status_code=404,
                detail="Recipient phone number not provided",
            )

        # Build Twilio client for the sub-account
        subaccount_client = await _get_twilio_client(db, account_id)

        app_callback_url, websocket_url = _populate_custom_urls()

        recording_status_callback = None
        if record:
            recording_status_callback = f"{app_callback_url}/recording_callback"

        try:
            call = subaccount_client.calls.create(
                to=recipient_phone_number,
                from_=sender_phone_number,
                url=(
                    f"{app_callback_url}/twilio_callback"
                    f"?ws_url={websocket_url}"
                    f"&agent_id={agent_id}"
                    f"&account_id={account_id}"
                ),
                method="POST",
                record=record,
                recording_status_callback=(
                    recording_status_callback if record else None
                ),
                recording_status_callback_event=(
                    ["completed"] if record else None
                ),
                status_callback=f"{app_callback_url}/call_status_callback",
                status_callback_event=[
                    "initiated", "ringing", "answered", "completed",
                ],
                status_callback_method="POST",
                timeout=30,
            )

            # Store compliance flag in Redis
            if is_compliant:
                await redis.set(
                    f"agent_compliant:{call.sid}", "1", ex=86400,
                )

            is_temp_agent = agent_config is not None
            if is_temp_agent:
                await redis.set(
                    f"is_temp_agent:{agent_id}", "1", ex=86400,
                )

            # Record the call in the database
            async with db.transaction():
                if is_temp_agent:
                    await insert_call(
                        db, call.sid, account_id,
                        sender_phone_number, recipient_phone_number, "ringing",
                    )
                else:
                    await insert_call(
                        db, call.sid, account_id,
                        sender_phone_number, recipient_phone_number, "ringing",
                        agent_id=uuid.UUID(agent_id),
                    )
                    await increment_agent_total_calls(db, uuid.UUID(agent_id))

                await increment_account_total_calls(db, account_id)

            # Store temp agent mapping for call_sid -> agent_id
            if is_temp_agent:
                await redis.set(
                    f"temp_agent_mapping:{call.sid}", agent_id, ex=86400,
                )

            response_data: dict[str, Any] = {
                "status": "success",
                "message": "Call initiated successfully",
                "call_details": {
                    "call_sid": call.sid,
                    "agent_id": agent_id,
                    "is_temporary_agent": is_temp_agent,
                    "from_number": sender_phone_number,
                    "to_number": recipient_phone_number,
                    "created_at": datetime.now().isoformat(),
                },
            }

            if webhook_url:
                response_data["call_details"]["webhook_url"] = webhook_url

            return JSONResponse(response_data, status_code=200)

        except Exception as exc:
            logger.error("Twilio call creation failed: %s", exc)
            return JSONResponse({"error": str(exc)}, status_code=500)

    except HTTPException as he:
        return JSONResponse({"error": he.detail}, status_code=he.status_code)
    except Exception as exc:
        logger.error("Exception in make_call: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")


# ---------------------------------------------------------------------------
# POST /call/end -- end an active call
# ---------------------------------------------------------------------------

@router.post("/call/end")
async def end_call(
    payload: EndCallPayload,
    org: OrgDep,
    db: DBDep,
) -> dict:
    """End an active Twilio call by setting its status to 'completed'."""
    account_id = org
    try:
        subaccount_client = await _get_twilio_client(db, account_id)

        try:
            subaccount_client.calls(payload.call_sid).update(status="completed")
            return {
                "status": "success",
                "message": "Call ended successfully",
                "call_sid": payload.call_sid,
            }
        except TwilioRestException as exc:
            logger.error("Twilio error ending call: %s", exc)
            raise HTTPException(
                status_code=400,
                detail="Failed to end call: Invalid call SID or call already ended",
            )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error ending call: %s", exc)
        raise HTTPException(status_code=500, detail="Internal Server Error")


# ---------------------------------------------------------------------------
# GET /details -- list call logs for an account
# ---------------------------------------------------------------------------

@router.get("/details")
async def get_details(
    org: OrgDep,
    db: DBDep,
) -> dict:
    """Return all call records for the authenticated account."""
    account_id = org
    try:
        calls = await get_calls_with_agent_name(db, account_id)

        if not calls:
            logger.warning("No calls found for account: %s", account_id)
            return {"calls": []}

        call_details = [
            {
                "call_sid": str(call["call_sid"]),
                "agent_id": str(call["agent_id"]) if call["agent_id"] else None,
                "name": call["name"] if call["name"] else "Web Call Agent",
                "account_id": str(call["account_id"]),
                "from_number": call["from_number"],
                "to_number": call["to_number"],
                "duration": (
                    float(f"{call['duration']:.1f}")
                    if call["duration"] is not None
                    else 0
                ),
                "recording_url": call["recording_url"],
                "transcription": call["transcription"],
                "status": call["status"],
                "cost": call["cost"],
                "call_type": call["call_type"],
                "created_at": call["created_at"].isoformat(),
            }
            for call in calls
        ]

        return {"calls": call_details}

    except Exception as exc:
        logger.error(
            "Error retrieving call details for account %s: %s",
            account_id, exc,
        )
        raise HTTPException(status_code=500, detail="Internal server error")


# ---------------------------------------------------------------------------
# GET /call-details/{call_sid} -- get specific call details
# ---------------------------------------------------------------------------

@router.get("/call-details/{call_sid}")
async def get_call_details(
    call_sid: str,
    org: OrgDep,
    db: DBDep,
) -> dict:
    """Return detailed information about a single call."""
    account_id = org
    try:
        call = await get_call_detail_by_sid(db, call_sid, account_id)

        if not call:
            logger.warning("Call not found: %s", call_sid)
            raise HTTPException(status_code=404, detail="Call not found")

        call_details: dict[str, Any] = {
            "call_sid": call["call_sid"],
            "agent_id": str(call["agent_id"]) if call["agent_id"] else None,
            "account_id": str(call["account_id"]),
            "from_number": call["from_number"],
            "to_number": call["to_number"],
            "duration": (
                float(f"{call['duration']:.1f}")
                if call["duration"] is not None
                else 0
            ),
            "recording_url": call["recording_url"],
            "transcription": call["transcription"],
            "status": call["status"],
            "call_type": call["call_type"],
            "cost": call["cost"],
            "created_at": call["created_at"].isoformat(),
            "transferred": call["transferred"],
            "transfer_info": None,
        }

        if call["transferred"]:
            call_details["transfer_info"] = {
                "department": call["transfer_department"],
                "number": call["transfer_number"],
                "time": (
                    call["transfer_time"].isoformat()
                    if call["transfer_time"]
                    else None
                ),
            }

        return call_details

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Error retrieving call details for call SID %s: %s",
            call_sid, exc,
        )
        raise HTTPException(status_code=500, detail="Internal server error")


# ---------------------------------------------------------------------------
# POST /web-call -- initiate browser-based WebRTC call
# ---------------------------------------------------------------------------

@router.post("/web-call")
async def initiate_web_call(
    payload: WebCallPayload,
    org: OrgDep,
    db: DBDep,
    redis: RedisDep,
) -> dict:
    """
    Initiate a web call session.
    Accepts either a persistent agent_id or a temporary agent_config.
    Returns the necessary details for the client to connect to the WebSocket.
    """
    account_id = org
    await _check_org_plan_and_minutes(db, account_id)

    try:
        agent_id = payload.agent_id
        agent_config = payload.agent_config
        web_call_sid = payload.web_call_sid or f"web_call_{uuid.uuid4().hex[:16]}"

        if not agent_id and not agent_config:
            raise HTTPException(
                status_code=400,
                detail="Either agent_id or agent_config must be provided",
            )

        # Generate a unique session token for WebSocket authentication
        session_token = f"session_{uuid.uuid4().hex[:16]}_{int(time.time())}"

        # Store session token with account_id and expiration (1 hour)
        await redis.setex(
            f"web_call_session_token:{session_token}",
            3600,
            str(account_id),
        )

        # ------------------------------------------------------------------
        # Path A: Persistent agent (agent_id only, no agent_config)
        # ------------------------------------------------------------------
        if agent_id and not agent_config:
            agent_result = await get_agent_basic_info(
                db, uuid.UUID(agent_id), account_id,
            )
            if not agent_result:
                raise HTTPException(
                    status_code=404,
                    detail="Agent not found or access denied",
                )

            persistent_agent_id = agent_result["agent_id"]

            # Create call record for persistent agent
            await insert_web_call(db, web_call_sid, persistent_agent_id, account_id)

            # Store session info in Redis
            await redis.set(
                f"web_call_session:{web_call_sid}",
                json.dumps({
                    "account_id": str(account_id),
                    "agent_id": agent_id,
                    "session_token": session_token,
                    "created_at": datetime.now().isoformat(),
                }),
                ex=3600,
            )

            return {
                "status": "success",
                "message": "Web call session ready to connect.",
                "agent_id": agent_id,
                "auth_token": session_token,
                "web_call_sid": web_call_sid,
            }

        # ------------------------------------------------------------------
        # Path B: Temporary agent (agent_config provided)
        # ------------------------------------------------------------------
        agent_prompts = payload.agent_prompts
        template_variables = payload.template_variables
        temp_agent_id = str(uuid.uuid4())

        # Normalise config structure
        if "agent_config" in agent_config:
            complete_config = agent_config
            if not agent_prompts and "agent_prompts" in agent_config:
                agent_prompts = agent_config["agent_prompts"]
        else:
            complete_config = {
                "agent_id": temp_agent_id,
                "agent_config": agent_config,
                "agent_prompts": agent_prompts or {},
            }

        # Merge template variables into agent_config
        if template_variables:
            inner_cfg = complete_config.get("agent_config", complete_config)
            if not isinstance(inner_cfg.get("template_variables"), dict):
                inner_cfg["template_variables"] = {}
            inner_cfg["template_variables"].update(template_variables)

        # Store temporary agent in Redis
        await redis.set(
            f"temp_agent_config:{temp_agent_id}",
            json.dumps(complete_config),
            ex=3600,
        )
        if agent_prompts:
            await redis.set(
                f"temp_agent_prompts:{temp_agent_id}",
                json.dumps(agent_prompts),
                ex=3600,
            )
        await redis.set(f"is_temp_agent:{temp_agent_id}", "1", ex=3600)
        await redis.set(
            f"temp_agent_account:{temp_agent_id}", str(account_id), ex=3600,
        )

        # Create call record (agent_id NULL for temp agents)
        await insert_web_call(db, web_call_sid, None, account_id)

        # Store web call session info
        await redis.set(
            f"web_call_session:{web_call_sid}",
            json.dumps({
                "account_id": str(account_id),
                "agent_id": temp_agent_id,
                "session_token": session_token,
                "created_at": datetime.now().isoformat(),
            }),
            ex=3600,
        )

        return {
            "status": "success",
            "message": "Web call session ready to connect.",
            "agent_id": temp_agent_id,
            "auth_token": session_token,
            "web_call_sid": web_call_sid,
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error initiating web call: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to initiate web call",
        )


# ---------------------------------------------------------------------------
# POST /web-call-status-callback -- web call status updates
# ---------------------------------------------------------------------------

@router.post("/web-call-status-callback")
async def web_call_status_callback(
    web_call_sid: str = Form(...),
    status: str = Form(...),
    duration: Optional[float] = Form(None),
    transcription: Optional[str] = Form(None),
    account_id: str = Form(...),
    db: DBDep = ...,
    redis: RedisDep = ...,
) -> PlainTextResponse:
    """
    Status callback for web calls to track completion, duration, and transcription.
    Similar to call_status_callback but for web-based calls.
    """
    try:
        status_mapping = {
            "initiated": "initiated",
            "connected": "in_progress",
            "completed": "completed",
            "failed": "failed",
            "disconnected": "completed",
        }
        internal_status = status_mapping.get(status, status)

        # Get transcription from Redis if not provided
        transcription_json_str: str | None = None
        if not transcription and internal_status == "completed":
            transcription_json_str = await redis.get(
                f"transcription:{web_call_sid}",
            )
        elif transcription:
            transcription_json_str = transcription

        async with db.transaction():
            # Get existing call details
            call_details = await get_call_with_agent_info(db, web_call_sid)

            if not call_details:
                logger.warning(
                    "Web call record not found for %s", web_call_sid,
                )
                return PlainTextResponse("OK")

            # Calculate cost for completed calls
            call_cost = 0.0
            if duration and internal_status == "completed":
                call_cost = await _calculate_call_cost(
                    db, int(duration), call_details["account_id"],
                )

            # Update call record using query function
            await update_call_fields(
                db, web_call_sid, internal_status,
                duration=float(duration) if duration and internal_status == "completed" else None,
                cost=call_cost if duration and internal_status == "completed" else None,
                transcription_json=transcription_json_str if transcription_json_str and internal_status == "completed" else None,
            )

            # Update agent + account stats for completed calls
            if internal_status == "completed" and duration:
                exact_duration = float(duration)

                if call_details["agent_id"]:
                    await update_agent_call_stats(
                        db, call_details["agent_id"],
                        exact_duration, call_cost,
                    )

                await update_account_call_stats(
                    db, call_details["account_id"],
                    exact_duration, call_cost,
                )

                # Deduct minutes from Organization
                minutes_to_deduct = exact_duration / 60.0
                await deduct_org_minutes(
                    db, call_details["account_id"], minutes_to_deduct,
                )

            # Clean up Redis transcription key
            if transcription_json_str and internal_status == "completed":
                await redis.delete(f"transcription:{web_call_sid}")

            # Send webhook for terminal statuses
            if (
                internal_status in ("completed", "failed")
                and call_details["webhook_url"]
            ):
                webhook_payload = {
                    "event_type": "web_call_completed",
                    "call_sid": web_call_sid,
                    "status": internal_status,
                    "call_type": call_details.get("call_type"),
                    "from_number": call_details.get("from_number"),
                    "to_number": call_details.get("to_number"),
                    "duration": (
                        round(float(duration), 1) if duration else 0
                    ),
                    "cost": call_cost,
                    "created_at": (
                        call_details["created_at"].isoformat()
                        if call_details.get("created_at")
                        else None
                    ),
                    "agent": {
                        "agent_id": (
                            str(call_details["agent_id"])
                            if call_details.get("agent_id")
                            else None
                        ),
                        "name": call_details.get("agent_name"),
                    },
                    "transcription": (
                        json.loads(transcription_json_str)
                        if transcription_json_str
                        else None
                    ),
                }
                await _send_webhook(
                    call_details["webhook_url"],
                    webhook_payload,
                    call_details["agent_name"],
                )

            logger.info(
                "Web call %s status updated to %s (duration=%ss, cost=$%s)",
                web_call_sid, internal_status, duration, call_cost,
            )

    except Exception as exc:
        logger.error(
            "Error in web call status callback: %s", exc, exc_info=True,
        )

    return PlainTextResponse("OK")


# ---------------------------------------------------------------------------
# GET /web-call/config/{agent_id} -- get web call configuration
# ---------------------------------------------------------------------------

@router.get("/web-call/config/{agent_id}")
async def get_web_call_config(
    agent_id: str,
    org: OrgDep,
    db: DBDep,
) -> dict:
    """
    Get web calling configuration for a specific agent.
    Helps with debugging and testing web calling setup.
    """
    account_id = org
    try:
        agent_result = await get_agent_config_info(
            db, uuid.UUID(agent_id), account_id,
        )

        if not agent_result:
            raise HTTPException(
                status_code=404,
                detail="Agent not found or access denied",
            )

        agent_config = (
            json.loads(agent_result["agent_config"])
            if isinstance(agent_result["agent_config"], str)
            else agent_result["agent_config"]
        )

        return {
            "agent_id": agent_id,
            "agent_name": agent_result["name"],
            "account_id": str(account_id),
            "websocket_endpoint": f"/web-call/v1/{agent_id}",
            "supported_features": {
                "real_time_streaming": True,
                "high_quality_audio": True,
                "context_awareness": True,
                "interruption_handling": True,
            },
            "audio_config": {
                "sample_rate": (
                    agent_config
                    .get("synthesizer", {})
                    .get("sampling_rate", 24000)
                ),
                "format": "linear16",
                "channels": 1,
            },
            "agent_capabilities": {
                "transcriber": (
                    agent_config
                    .get("transcriber", {})
                    .get("model", "default")
                ),
                "synthesizer": (
                    agent_config
                    .get("synthesizer", {})
                    .get("provider", "default")
                ),
                "llm": (
                    agent_config
                    .get("llm", {})
                    .get("model", "default")
                ),
            },
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error getting web call config: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Failed to get web call configuration",
        )


# ---------------------------------------------------------------------------
# POST /chat-session -- initiate chat session
# ---------------------------------------------------------------------------

@router.post("/chat-session")
async def initiate_chat_session(
    payload: ChatSessionPayload,
    org: OrgDep,
    db: DBDep,
    redis: RedisDep,
) -> dict:
    """
    Initiate a chat session.
    Validates agent access and returns session token for WebSocket connection.
    """
    account_id = org
    await _check_org_plan_and_minutes(db, account_id)

    try:
        agent_id = payload.agent_id
        template_variables = payload.template_variables
        chat_session_id = (
            payload.chat_session_id or f"chat_{uuid.uuid4().hex[:16]}"
        )

        # Generate a unique session token for WebSocket authentication
        session_token = f"session_{uuid.uuid4().hex[:16]}_{int(time.time())}"

        # Store session token with account_id and expiration (1 hour)
        await redis.setex(
            f"chat_session_token:{session_token}",
            3600,
            str(account_id),
        )

        # Validate agent exists and belongs to account
        agent_result = await get_agent_basic_info(
            db, uuid.UUID(agent_id), account_id,
        )
        if not agent_result:
            raise HTTPException(
                status_code=404,
                detail="Agent not found or access denied",
            )

        # Store chat session info in Redis for websocket access control
        await redis.set(
            f"chat_session:{chat_session_id}",
            json.dumps({
                "account_id": str(account_id),
                "agent_id": agent_id,
                "session_token": session_token,
                "template_variables": template_variables,
                "created_at": datetime.now().isoformat(),
            }),
            ex=3600,
        )

        return {
            "status": "success",
            "message": "Chat session ready to connect.",
            "agent_id": agent_id,
            "auth_token": session_token,
            "chat_session_id": chat_session_id,
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error initiating chat session: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Failed to initiate chat session",
        )
