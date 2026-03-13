"""
Twilio webhook callback routes.

These endpoints are called by Twilio (or by our own backend for trigger_webhook)
and MUST NOT require the x-api-key Bearer token.  Twilio authenticates via
signature validation (TODO: port TwilioRequestValidator middleware).

Routes (exact paths from main_server.py -- must not change!):
  POST  /twilio_callback              -- Twilio audio stream TwiML response
  POST  /call_status_callback         -- Twilio call status updates
  POST  /recording_callback           -- Twilio recording status callback
  POST  /inbound_call                 -- Inbound call TwiML handler
  POST  /trigger_webhook/{call_sid}   -- Manual webhook trigger (Bearer auth)
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime

import aiohttp
import asyncpg
import redis.asyncio as aioredis
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse
from twilio.rest import Client as TwilioClient
from twilio.twiml.voice_response import Connect, VoiceResponse

from app.config import get_settings
from app.dependencies import DBDep, OrgDep, get_db, get_redis
from db.queries.agents import (
    get_agent_compliance,
    get_agent_webhook_info,
    increment_agent_total_calls,
    update_agent_call_stats,
)
from db.queries.calls import (
    call_exists,
    get_call_account_id,
    get_call_by_sid,
    get_call_with_agent_info,
    get_call_with_all_fields,
    insert_call,
    insert_skeleton_call,
    set_call_recording_sid,
    update_call_fields,
    update_call_recording_with_sid,
)
from db.queries.campaigns import (
    get_campaign_with_list_by_list_id,
    get_campaign_with_list_by_phone,
)
from db.queries.contacts import (
    get_contact_by_call_sid,
    get_contact_by_phone_latest,
)
from db.queries.organizations import (
    add_org_minutes,
    deduct_org_minutes,
    get_account_full,
    get_org_auto_refill_settings,
    get_org_minutes,
    get_org_plan_and_minutes,
    get_org_pricing,
    get_twilio_credentials,
    increment_account_total_calls,
    update_account_call_stats,
)

logger = logging.getLogger(__name__)
settings = get_settings()

# No prefix -- these routes are mounted at the root to match old backend paths.
router = APIRouter(tags=["webhooks"])


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _get_callback_url() -> str:
    """HTTP base URL used for Twilio callback endpoints."""
    return settings.base_url.rstrip("/")


def _get_websocket_url() -> str:
    """WebSocket base URL derived from the HTTP base URL.

    ``https://api.example.com`` -> ``wss://api.example.com``
    ``http://localhost:8001``   -> ``ws://localhost:8001``
    """
    base = settings.base_url.rstrip("/")
    return base.replace("https://", "wss://").replace("http://", "ws://")


async def _get_twilio_client(
    db: asyncpg.Connection,
    account_id: uuid.UUID,
) -> TwilioClient:
    """Return a Twilio client initialised with the sub-account credentials."""
    row = await get_twilio_credentials(db, account_id)
    if not row or not row["twilio_subaccount_sid"] or not row["twilio_subaccount_auth_token"]:
        raise HTTPException(
            status_code=400,
            detail="Twilio credentials not configured for this account",
        )
    return TwilioClient(row["twilio_subaccount_sid"], row["twilio_subaccount_auth_token"])


async def _get_org_price_per_minute(
    conn: asyncpg.Connection,
    account_id: uuid.UUID,
) -> float:
    """Look up the per-minute call price for an organisation."""
    org = await get_org_pricing(conn, account_id)
    if not org:
        return 0.20  # default fallback
    if org["custom_price_per_minute"] is not None:
        return org["custom_price_per_minute"]
    return org["default_price_per_minute"]


async def _calculate_call_cost(
    conn: asyncpg.Connection,
    duration_seconds: int,
    account_id: uuid.UUID,
) -> float:
    """Calculate call cost using the organisation's pricing."""
    price_per_minute = await _get_org_price_per_minute(conn, account_id)
    duration_minutes = duration_seconds / 60.0
    return duration_minutes * price_per_minute


async def _check_org_plan_and_minutes(
    db: asyncpg.Connection,
    account_id: uuid.UUID,
) -> None:
    """Raise HTTP 403 if the org's plan or balance prevents calls."""
    ALLOWED_PLANS = {"starter", "pro", "enterprise", "basic", "professional", "scale"}
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
            detail="You do not have enough minutes to make calls. Please upgrade or purchase more minutes.",
        )


async def _trigger_auto_refill(
    conn: asyncpg.Connection,
    account_id: uuid.UUID,
    current_balance: float,
) -> bool:
    """Check if auto-refill should be triggered and execute it if needed.

    Returns True if refill was triggered, False otherwise.
    """
    try:
        org_settings = await get_org_auto_refill_settings(conn, account_id)
        if not org_settings or not org_settings["autoRefillMinutesEnabled"]:
            return False

        threshold = org_settings["autoRefillMinutesThreshold"] or 10
        refill_amount = org_settings["autoRefillMinutesAmount"] or 50

        if refill_amount < 5:
            logger.error(
                "Auto-refill amount %s is below minimum of 5 minutes for account %s",
                refill_amount, account_id,
            )
            return False

        if current_balance > threshold:
            return False

        logger.info(
            "Auto-refill triggered for account %s: balance %s <= threshold %s",
            account_id, current_balance, threshold,
        )

        stripe_customer_id = org_settings["stripeCustomerId"]
        if not stripe_customer_id:
            logger.error("No Stripe customer ID found for account %s", account_id)
            return False

        try:
            import stripe  # lazy import -- only needed when billing fires

            stripe.api_key = settings.stripe_secret_key

            customer = stripe.Customer.retrieve(stripe_customer_id)
            default_pm = customer.invoice_settings.default_payment_method
            if not default_pm:
                logger.error(
                    "No default payment method for customer %s", stripe_customer_id,
                )
                return False

            price_per_minute = await _get_org_price_per_minute(conn, account_id)
            total_cost = refill_amount * price_per_minute

            payment_intent = stripe.PaymentIntent.create(
                amount=int(total_cost * 100),  # cents
                currency="usd",
                customer=stripe_customer_id,
                payment_method=default_pm,
                off_session=True,
                confirm=True,
                metadata={
                    "type": "auto_refill",
                    "account_id": str(account_id),
                    "minutes": str(refill_amount),
                },
            )

            if payment_intent.status == "succeeded":
                await add_org_minutes(conn, account_id, float(refill_amount))
                logger.info(
                    "Auto-refill successful for account %s: added %s minutes for $%.2f",
                    account_id, refill_amount, total_cost,
                )
                return True

            logger.error(
                "Auto-refill payment failed for account %s: %s",
                account_id, payment_intent.status,
            )
            return False

        except Exception as stripe_error:
            logger.error(
                "Stripe error during auto-refill for account %s: %s",
                account_id, stripe_error,
            )
            return False

    except Exception as exc:
        logger.error(
            "Error in auto-refill trigger for account %s: %s", account_id, exc,
        )
        return False


async def _notify_agent_webhook(
    agent_id: uuid.UUID,
    payload: dict,
    *,
    db: asyncpg.Connection,
) -> None:
    """Fire-and-forget POST to the agent's configured webhook URL."""
    try:
        agent = await get_agent_webhook_info(db, agent_id)
        if not agent or not agent["webhook_url"]:
            return

        logger.info(
            "Sending webhook to %s for agent %s",
            agent["webhook_url"], agent["name"],
        )
        async with aiohttp.ClientSession() as session:
            async with session.post(
                agent["webhook_url"], json=payload, timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status != 200:
                    logger.error("Webhook notification failed: %s", resp.status)
                else:
                    logger.info("Webhook notification succeeded: %s", resp.status)
    except Exception as exc:
        logger.error("Error in webhook notification: %s", exc)


async def _enable_call_recording(
    call_sid: str,
    callback_url: str,
    account_id: uuid.UUID,
    db: asyncpg.Connection,
) -> str | None:
    """Enable recording for a call using the Twilio REST API.

    Waits briefly for the call to be established, then starts recording.
    Returns the recording SID on success, None on failure.
    """
    try:
        await asyncio.sleep(1.5)

        client = await _get_twilio_client(db, account_id)
        app_callback_url = _get_callback_url()

        recording = client.calls(call_sid).recordings.create(
            recording_status_callback=callback_url,
            recording_status_callback_event=["completed"],
        )

        # Ensure call has our status callback so we get completion events
        client.calls(call_sid).update(
            status_callback=f"{app_callback_url}/call_status_callback",
            status_callback_method="POST",
        )

        # Persist recording SID in the DB
        async with db.transaction():
            await set_call_recording_sid(db, call_sid, recording.sid)

        logger.info(
            "Recording started for call %s, recording SID: %s",
            call_sid, recording.sid,
        )
        return recording.sid
    except Exception as exc:
        logger.error("Failed to start recording for call %s: %s", call_sid, exc)
        return None


async def _build_contact_campaign_payload(
    conn: asyncpg.Connection,
    call_sid: str,
    call_details: asyncpg.Record,
) -> dict:
    """Build the contact and campaign portions of a webhook payload.

    Returns a dict with optional ``contact`` and ``campaign`` keys.
    """
    result: dict = {}

    # Try to find contact by call_sid first, then by phone number for outbound
    contact_info = await get_contact_by_call_sid(conn, call_sid)
    if not contact_info and call_details.get("call_type") == "outbound":
        contact_info = await get_contact_by_phone_latest(
            conn, call_details.get("to_number"),
        )

    campaign_info = None

    if contact_info:
        result["contact"] = {
            "contact_id": str(contact_info["contact_id"]),
            "list_id": str(contact_info["list_id"]),
            "account_id": str(contact_info["account_id"]),
            "phone_number": contact_info["phone_number"],
            "template_variables": contact_info["template_variables"],
            "status": contact_info["status"],
            "error_message": contact_info["error_message"],
            "processed_at": (
                contact_info["processed_at"].isoformat()
                if contact_info["processed_at"]
                else None
            ),
            "created_at": (
                contact_info["created_at"].isoformat()
                if contact_info["created_at"]
                else None
            ),
        }
        # Find campaign via contact's list_id
        campaign_info = await get_campaign_with_list_by_list_id(
            conn, contact_info["list_id"], contact_info["account_id"],
        )
    else:
        # Fallback: find campaign by sender phone number
        call_type = call_details.get("call_type", "outbound")
        search_number = (
            call_details.get("from_number")
            if call_type == "outbound"
            else call_details.get("to_number")
        )
        campaign_info = await get_campaign_with_list_by_phone(
            conn, search_number, call_details.get("account_id"),
        )

    if campaign_info:
        result["campaign"] = {
            "campaign_id": str(campaign_info["campaign_id"]),
            "name": campaign_info["name"],
            "description": campaign_info["description"],
            "status": campaign_info["status"],
            "sender_phone_number": campaign_info["sender_phone_number"],
            "total_contacts": campaign_info["total_contacts"],
            "completed_calls": campaign_info["completed_calls"],
            "failed_calls": campaign_info["failed_calls"],
            "created_at": (
                campaign_info["created_at"].isoformat()
                if campaign_info.get("created_at")
                else None
            ),
            "started_at": (
                campaign_info["started_at"].isoformat()
                if campaign_info.get("started_at")
                else None
            ),
            "completed_at": (
                campaign_info["completed_at"].isoformat()
                if campaign_info.get("completed_at")
                else None
            ),
            "delay_between_calls": campaign_info["delay_between_calls"],
            "scheduled_time": (
                campaign_info["scheduled_time"].isoformat()
                if campaign_info.get("scheduled_time")
                else None
            ),
            "timezone": campaign_info["timezone"],
            "status_changed_at": (
                campaign_info["status_changed_at"].isoformat()
                if campaign_info.get("status_changed_at")
                else None
            ),
            "list": {
                "list_id": str(campaign_info["list_id"]),
                "name": campaign_info["list_name"],
                "description": campaign_info["list_description"],
            },
        }

    return result


async def _upload_recording_to_s3(
    audio_data: bytes,
    call_sid: str,
) -> str:
    """Upload recording bytes to S3 and return the public URL.

    Uses ``aiobotocore`` (async) instead of the synchronous ``boto3`` client
    used in the old backend.
    """
    import aiobotocore.session as aio_session

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    s3_key = f"recordingskallabot/recordings_{call_sid}_{timestamp}.wav"

    session = aio_session.get_session()
    async with session.create_client(
        "s3",
        region_name=settings.aws_region,
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
    ) as s3_client:
        await s3_client.put_object(
            Bucket=settings.s3_bucket_name,
            Key=s3_key,
            Body=audio_data,
            ContentType="audio/wav",
        )

    s3_url = (
        f"https://{settings.s3_bucket_name}.s3.{settings.aws_region}.amazonaws.com/{s3_key}"
    )
    return s3_url


# ---------------------------------------------------------------------------
# Twilio status mapping
# ---------------------------------------------------------------------------

_TWILIO_STATUS_MAP: dict[str, str] = {
    "queued": "initiated",
    "ringing": "ringing",
    "in-progress": "in_progress",
    "completed": "completed",
    "busy": "rejected",
    "no-answer": "rejected",
    "failed": "failed",
    "canceled": "canceled",
}

_TERMINAL_STATUSES = frozenset({"completed", "failed", "rejected", "canceled"})


# ===========================================================================
# POST /twilio_callback
# ===========================================================================

@router.post("/twilio_callback")
async def twilio_callback(
    ws_url: str = Query(...),
    agent_id: str = Query(...),
    account_id: str = Query(...),
    redis: aioredis.Redis = Depends(get_redis),
):
    """Return TwiML that connects the Twilio call to our WebSocket agent.

    This is the URL Twilio POSTs to after the call is answered; the TwiML
    response instructs Twilio to open a bi-directional media stream to the
    agent's WebSocket endpoint.
    """
    try:
        # If a temporary agent was used, verify its config exists in Redis
        is_temp_agent = await redis.get(f"is_temp_agent:{agent_id}")
        if is_temp_agent:
            agent_config = await redis.get(f"temp_agent_config:{agent_id}")
            if not agent_config:
                logger.error("Temporary agent config not found for %s", agent_id)
                raise HTTPException(
                    status_code=404,
                    detail="Temporary agent configuration not found",
                )

        response = VoiceResponse()
        connect = Connect()

        websocket_url = _get_websocket_url()
        websocket_twilio_route = f"{websocket_url}/v1/chat/{agent_id}"
        connect.stream(url=websocket_twilio_route)
        logger.info("Setting up Twilio WebSocket connection to %s", websocket_twilio_route)
        response.append(connect)

        return PlainTextResponse(
            str(response), status_code=200, media_type="text/xml",
        )

    except HTTPException as he:
        logger.error("HTTP Exception in twilio_callback: %s", he.detail)
        return PlainTextResponse(
            str(he.detail), status_code=he.status_code, media_type="text/plain",
        )
    except Exception as exc:
        logger.error("Exception occurred in twilio_callback: %s", exc)
        return PlainTextResponse(str(exc), status_code=500, media_type="text/plain")


# ===========================================================================
# POST /call_status_callback
# ===========================================================================

@router.post("/call_status_callback")
async def call_status_callback(
    request: Request,
    db: asyncpg.Connection = Depends(get_db),
    redis: aioredis.Redis = Depends(get_redis),
):
    """Handle Twilio call status updates (queued, ringing, in-progress, completed, etc.).

    For terminal statuses the handler:
      1. Saves the transcription from Redis (if not compliant).
      2. Calculates cost and updates call/agent/account/org records.
      3. Deducts minutes and checks auto-refill.
      4. Fires the agent's webhook with a rich payload.
    """
    try:
        form_data = await request.form()
        call_sid = form_data.get("CallSid")
        call_status = form_data.get("CallStatus")
        call_duration = form_data.get("CallDuration")  # seconds (string)

        internal_status = _TWILIO_STATUS_MAP.get(call_status, call_status)
        is_terminal = internal_status in _TERMINAL_STATUSES

        # For terminal statuses wait for other callbacks (e.g. recording) to land
        if is_terminal:
            await asyncio.sleep(15)

        # Check compliance flag
        is_compliant = await redis.get(f"agent_compliant:{call_sid}")

        # Get transcription from Redis for completed calls (unless compliant)
        transcription_json_str: str | None = None
        redis_key = f"transcription:{call_sid}"
        if internal_status == "completed" and not is_compliant:
            transcription_json_str = await redis.get(redis_key)

        # Check for temporary-agent mapping
        temp_agent_id = await redis.get(f"temp_agent_mapping:{call_sid}")

        async with db.transaction():
            # For early statuses of temp-agent calls, create a skeleton record
            if call_status in ("initiated", "ringing") and temp_agent_id:
                if not await call_exists(db, call_sid):
                    try:
                        await insert_skeleton_call(db, call_sid, internal_status)
                        logger.info(
                            "Created call record for temporary agent call %s", call_sid,
                        )
                    except Exception as exc:
                        logger.error("Error creating call record: %s", exc)

            # Fetch full call detail with agent info
            call_details = await get_call_with_agent_info(db, call_sid)

            if not call_details:
                logger.warning(
                    "Call record not found for %s with status %s. "
                    "This is expected for new calls.",
                    call_sid, call_status,
                )
                return PlainTextResponse("OK")

            # ---- Calculate cost ------------------------------------------------
            call_cost: float = 0
            if call_duration and internal_status == "completed":
                call_cost = await _calculate_call_cost(
                    db, int(call_duration), call_details["account_id"],
                )

            # ---- Update call record --------------------------------------------
            await update_call_fields(
                db, call_sid, internal_status,
                duration=float(call_duration) if call_duration and internal_status == "completed" else None,
                cost=call_cost if call_duration and internal_status == "completed" else None,
                transcription_json=transcription_json_str if transcription_json_str and internal_status == "completed" and not is_compliant else None,
            )

            # ---- Update agent / account stats ----------------------------------
            if internal_status == "completed" and call_duration:
                exact_duration = float(call_duration)

                # Agent totals
                await update_agent_call_stats(
                    db, call_details["agent_id"],
                    exact_duration, call_cost,
                )

                # Account totals
                await update_account_call_stats(
                    db, call_details["account_id"],
                    exact_duration, call_cost,
                )

                # Deduct minutes from Organisation
                minutes_to_deduct = exact_duration / 60.0
                current_org = await get_org_minutes(db, call_details["account_id"])
                current_balance = current_org["minutes"] if current_org else 0
                new_balance = current_balance - minutes_to_deduct

                await deduct_org_minutes(
                    db, call_details["account_id"], minutes_to_deduct,
                )

                # Auto-refill check
                refill_triggered = await _trigger_auto_refill(
                    db, call_details["account_id"], new_balance,
                )
                if refill_triggered:
                    logger.info(
                        "Auto-refill triggered after call %s for account %s",
                        call_sid, call_details["account_id"],
                    )

            # Clean up Redis keys after successful DB update
            if transcription_json_str and internal_status == "completed":
                await redis.delete(redis_key)

            if internal_status == "completed":
                await redis.delete(f"agent_compliant:{call_sid}")

            # ---- Fire webhook for terminal statuses ----------------------------
            if is_terminal and call_details["webhook_url"]:
                # Re-read the call to pick up any recording data written by
                # the recording_callback that may have landed before our
                # 15-second sleep.
                updated_call = await get_call_by_sid(db, call_sid)
                recording_url = (
                    updated_call.get("recording_url") if updated_call else None
                )
                recording_sid = (
                    updated_call.get("recording_sid") if updated_call else None
                )

                logger.info(
                    "Sending webhook for call %s with recording data: %s, %s",
                    call_sid, recording_url, recording_sid,
                )

                webhook_payload: dict = {
                    "event_type": "call_completed",
                    "call_sid": call_sid,
                    "status": internal_status,
                    "call_type": call_details.get("call_type"),
                    "from_number": call_details.get("from_number"),
                    "to_number": call_details.get("to_number"),
                    "duration": (
                        round(float(call_duration), 1) if call_duration else 0
                    ),
                    "cost": call_cost,
                    "created_at": (
                        call_details.get("created_at").isoformat()
                        if call_details.get("created_at")
                        else None
                    ),
                    "agent": {
                        "agent_id": (
                            str(call_details.get("agent_id"))
                            if call_details.get("agent_id")
                            else None
                        ),
                        "name": call_details.get("agent_name"),
                    },
                    "recording": {
                        "url": recording_url,
                        "sid": recording_sid,
                    },
                    "transcription": (
                        json.loads(transcription_json_str)
                        if transcription_json_str
                        else None
                    ),
                    "transfer": {
                        "transferred": call_details.get("transferred", False),
                        "department": call_details.get("transfer_department"),
                        "number": call_details.get("transfer_number"),
                        "time": (
                            call_details.get("transfer_time").isoformat()
                            if call_details.get("transfer_time")
                            else None
                        ),
                    },
                }

                # Enrich with contact / campaign / account data
                extra = await _build_contact_campaign_payload(
                    db, call_sid, call_details,
                )
                webhook_payload.update(extra)

                # Account info
                account_info = await get_account_full(
                    db, call_details.get("account_id"),
                )
                if account_info:
                    account_data: dict = {
                        "account_id": (
                            str(account_info.get("account_id"))
                            if account_info.get("account_id")
                            else None
                        ),
                    }
                    if account_info.get("total_calls") is not None:
                        account_data["total_calls"] = account_info["total_calls"]
                    if account_info.get("total_duration") is not None:
                        account_data["total_duration"] = account_info["total_duration"]
                    if account_info.get("total_cost") is not None:
                        account_data["total_cost"] = account_info["total_cost"]
                    if account_info.get("name"):
                        account_data["name"] = account_info["name"]
                    webhook_payload["account"] = account_data

                # Fire asynchronously so we don't block the response
                if call_details.get("agent_id"):
                    asyncio.create_task(
                        _notify_agent_webhook(
                            call_details["agent_id"],
                            webhook_payload,
                            db=db,
                        )
                    )

        return PlainTextResponse("OK")

    except Exception as exc:
        logger.error("Error in call_status_callback: %s", exc)
        return PlainTextResponse("Error", status_code=500)


# ===========================================================================
# POST /recording_callback
# ===========================================================================

@router.post("/recording_callback")
async def recording_callback(
    request: Request,
    db: asyncpg.Connection = Depends(get_db),
):
    """Handle Twilio recording status updates.

    When a recording is ``completed``, the handler:
      1. Downloads the ``.wav`` from Twilio (using per-account credentials).
      2. Uploads it to S3.
      3. Updates the call record with the S3 URL and recording SID.
    """
    try:
        form_data = await request.form()
        recording_url = form_data.get("RecordingUrl")
        call_sid = form_data.get("CallSid")
        recording_status = form_data.get("RecordingStatus")
        recording_sid = form_data.get("RecordingSid")

        if recording_status != "completed":
            return PlainTextResponse("OK")

        # Give Twilio a moment to finalise the recording file
        await asyncio.sleep(7)

        # Look up the call to get account credentials
        call_info = await get_call_account_id(db, call_sid)
        if not call_info:
            logger.error("Call not found for call_sid: %s", call_sid)
            return PlainTextResponse("Call not found", status_code=404)

        account_id: uuid.UUID = call_info["account_id"]

        # Fetch sub-account credentials for authenticated download
        account_info = await get_twilio_credentials(db, account_id)
        if not account_info:
            logger.error("Account not found for account_id: %s", account_id)
            return PlainTextResponse("Account not found", status_code=404)

        auth = aiohttp.BasicAuth(
            account_info["twilio_subaccount_sid"],
            account_info["twilio_subaccount_auth_token"],
        )

        async with aiohttp.ClientSession(auth=auth) as session:
            async with session.get(f"{recording_url}.wav") as resp:
                if resp.status != 200:
                    logger.error("Failed to download recording: %s", resp.status)
                    return PlainTextResponse(
                        "Error downloading recording", status_code=500,
                    )
                audio_data = await resp.read()

        # Upload to S3
        try:
            s3_url = await _upload_recording_to_s3(audio_data, call_sid)
        except Exception as s3_err:
            logger.error("S3 upload error: %s", s3_err)
            return PlainTextResponse("Error uploading to S3", status_code=500)

        # Update DB with the S3 URL and recording SID
        await update_call_recording_with_sid(db, call_sid, s3_url, recording_sid)
        logger.info("Successfully uploaded recording to S3: %s", s3_url)

        # TODO: schedule Twilio recording deletion after 7 days
        # (requires a task scheduler -- old code used APScheduler)

        return PlainTextResponse("OK")

    except Exception as exc:
        logger.error("Error in recording_callback: %s", exc)
        return PlainTextResponse("Error", status_code=500)


# ===========================================================================
# POST /inbound_call
# ===========================================================================

@router.post("/inbound_call")
async def inbound_call_handler(
    request: Request,
    agent_id: str = Query(None),
    account_id: str = Query(None),
    db: asyncpg.Connection = Depends(get_db),
    redis: aioredis.Redis = Depends(get_redis),
):
    """Return TwiML that connects an inbound call to the agent WebSocket.

    Also creates the initial call record, bumps counters, and optionally
    enables call recording (unless the agent is marked compliant).
    """
    # Pre-flight: check org plan and minutes
    if account_id:
        try:
            await _check_org_plan_and_minutes(db, uuid.UUID(account_id))
        except HTTPException as he:
            if he.status_code == 403:
                response = VoiceResponse()
                response.say("This service is not available at the moment.")
                return PlainTextResponse(
                    str(response), status_code=200, media_type="text/xml",
                )
            raise

    try:
        form_data = await request.form()
        call_sid = form_data.get("CallSid")
        from_number = form_data.get("From")
        to_number = form_data.get("To")

        logger.info(
            "Inbound call received: From=%s, To=%s, CallSid=%s",
            from_number, to_number, call_sid,
        )

        if not agent_id or not account_id:
            logger.error("Missing agent_id or account_id in request params")
            response = VoiceResponse()
            response.say(
                "Sorry, this number is not configured for inbound calls. "
                "Use query parameters in your webhook URL."
            )
            return PlainTextResponse(str(response), media_type="text/xml")

        # Validate phone number status callback configuration
        try:
            await _validate_phone_number_status_callback(
                db, uuid.UUID(account_id), to_number,
            )
        except HTTPException as validation_error:
            logger.error(
                "Call rejected due to invalid configuration: %s",
                validation_error.detail,
            )
            response = VoiceResponse()
            response.reject()
            return PlainTextResponse(
                str(response), status_code=200, media_type="text/xml",
            )

        logger.info("Using agent_id=%s, account_id=%s", agent_id, account_id)

        # Check agent compliance (controls recording / transcription)
        is_compliant = False
        try:
            agent_info = await get_agent_compliance(db, uuid.UUID(agent_id))
            is_compliant = bool(agent_info and agent_info["is_compliant"])
        except Exception as db_err:
            logger.error("Database error checking compliance (non-fatal): %s", db_err)

        # Build TwiML response with <Connect><Stream> to our WebSocket
        app_callback_url = _get_callback_url()
        websocket_url = _get_websocket_url()

        # Enable recording (async task) if agent is not compliant
        if not is_compliant:
            try:
                asyncio.create_task(
                    _enable_call_recording(
                        call_sid,
                        f"{app_callback_url}/recording_callback",
                        uuid.UUID(account_id),
                        db,
                    )
                )
                logger.info("Initiated call recording task for %s", call_sid)
            except Exception as rec_err:
                logger.error("Failed to initiate recording task: %s", rec_err)
        else:
            logger.info(
                "Agent is compliant, recording disabled for call %s", call_sid,
            )

        response = VoiceResponse()
        connect = Connect()
        websocket_twilio_route = f"{websocket_url}/v1/chat/{agent_id}"
        connect.stream(url=websocket_twilio_route)
        response.append(connect)

        logger.info("Connecting call to WebSocket: %s", websocket_twilio_route)

        # Persist call record and update counters (best-effort)
        try:
            if is_compliant:
                await redis.set(
                    f"agent_compliant:{call_sid}", "1", ex=86400,
                )
                logger.info(
                    "Agent is compliant, transcription will not be stored for call %s",
                    call_sid,
                )

            if not await call_exists(db, call_sid):
                await insert_call(
                    db, call_sid,
                    uuid.UUID(account_id),
                    from_number, to_number, "in_progress",
                    agent_id=uuid.UUID(agent_id),
                    call_type="inbound",
                )
                await increment_agent_total_calls(db, uuid.UUID(agent_id))
                await increment_account_total_calls(db, uuid.UUID(account_id))
        except Exception as db_err:
            logger.error("Database error (non-fatal): %s", db_err)

        return PlainTextResponse(
            str(response), status_code=200, media_type="text/xml",
        )

    except Exception as exc:
        logger.error("Error in inbound call handler: %s", exc)
        response = VoiceResponse()
        response.say("An error occurred processing your call.")
        return PlainTextResponse(
            str(response), status_code=500, media_type="text/xml",
        )


async def _validate_phone_number_status_callback(
    db: asyncpg.Connection,
    account_id: uuid.UUID,
    phone_number: str,
) -> bool:
    """Validate that the Twilio phone number has our call_status_callback set.

    Raises HTTPException if configuration is invalid.
    """
    try:
        client = await _get_twilio_client(db, account_id)
        app_callback_url = _get_callback_url()
        expected_callback_url = f"{app_callback_url}/call_status_callback"

        numbers = client.incoming_phone_numbers.list(phone_number=phone_number)
        if not numbers:
            logger.error(
                "Phone number %s not found in Twilio account %s",
                phone_number, account_id,
            )
            raise HTTPException(
                status_code=400,
                detail=f"Phone number {phone_number} not found in your Twilio account",
            )

        number = numbers[0]

        # Check if voice_url and status_callback are properly configured
        if not number.status_callback:
            logger.warning(
                "Phone number %s missing status_callback, updating...",
                phone_number,
            )
            number.update(
                status_callback=expected_callback_url,
                status_callback_method="POST",
            )

        return True

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Error validating phone number callback for %s: %s",
            phone_number, exc,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to validate phone number configuration: {exc}",
        )


# ===========================================================================
# POST /trigger_webhook/{call_sid}
# ===========================================================================

@router.post("/trigger_webhook/{call_sid}")
async def trigger_webhook(
    call_sid: str,
    org: OrgDep,
    db: DBDep,
):
    """Manually trigger a webhook for a specific call SID.

    Useful for testing or when the automatic webhook failed.
    This endpoint **does** require Bearer auth (``OrgDep``).
    """
    try:
        call_details = await get_call_with_all_fields(db, call_sid)

        if not call_details:
            raise HTTPException(status_code=404, detail="Call not found")

        if not call_details["webhook_url"]:
            raise HTTPException(
                status_code=400,
                detail="No webhook URL configured for this agent",
            )

        # Build payload
        webhook_payload: dict = {
            "event_type": "call_completed",
            "call_sid": call_sid,
            "status": call_details.get("status"),
            "call_type": call_details.get("call_type"),
            "from_number": call_details.get("from_number"),
            "to_number": call_details.get("to_number"),
            "duration": (
                float(call_details.get("duration"))
                if call_details.get("duration")
                else 0
            ),
            "cost": (
                float(call_details.get("cost"))
                if call_details.get("cost")
                else 0
            ),
            "created_at": (
                call_details.get("created_at").isoformat()
                if call_details.get("created_at")
                else None
            ),
            "agent": {
                "agent_id": (
                    str(call_details.get("agent_id"))
                    if call_details.get("agent_id")
                    else None
                ),
                "name": call_details.get("agent_name"),
            },
            "recording": {
                "url": call_details.get("recording_url"),
                "sid": call_details.get("recording_sid"),
            },
            "transcription": call_details.get("transcription"),
            "transfer": {
                "transferred": call_details.get("transferred", False),
                "department": call_details.get("transfer_department"),
                "number": call_details.get("transfer_number"),
                "time": (
                    call_details.get("transfer_time").isoformat()
                    if call_details.get("transfer_time")
                    else None
                ),
            },
            "manually_triggered": True,
        }

        # Enrich with contact / campaign info
        extra = await _build_contact_campaign_payload(db, call_sid, call_details)
        webhook_payload.update(extra)

        # Account info
        account_info = await get_account_full(
            db, call_details.get("account_id"),
        )
        if account_info:
            webhook_payload["account"] = {
                "account_id": str(account_info["account_id"]),
                "name": account_info["name"],
                "total_calls": account_info["total_calls"],
                "total_duration": account_info["total_duration"],
                "total_cost": account_info["total_cost"],
            }

        # Actually send the webhook
        async with aiohttp.ClientSession() as session:
            async with session.post(
                call_details.get("webhook_url", ""),
                json=webhook_payload,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status != 200:
                    logger.error("Webhook notification failed: %s", resp.status)
                    return {
                        "success": False,
                        "message": (
                            f"Webhook notification failed with status {resp.status}"
                        ),
                        "call_sid": call_sid,
                    }

        return {
            "success": True,
            "message": "Webhook triggered successfully",
            "call_sid": call_sid,
            "webhook_url": call_details.get("webhook_url", ""),
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error triggering webhook: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to trigger webhook: {exc}",
        )
