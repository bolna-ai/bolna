"""
Call transfer and whisper TwiML routes.

main_server.py reference: lines ~6876-7005.

Routes (exact paths from main_server.py):
  POST   /transfer-call   -- transfer an active call to another number
  POST   /whisper-twiml   -- generate TwiML for whisper/pre-call message (Twilio webhook, no auth)
"""

from __future__ import annotations

import json
import logging
import urllib.parse
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from twilio.base.exceptions import TwilioRestException
from twilio.twiml.voice_response import Dial, VoiceResponse

from app.config import get_settings
from app.dependencies import DBDep, RedisDep

logger = logging.getLogger(__name__)
settings = get_settings()

# No prefix -- old routes live at root: /transfer-call, /whisper-twiml
router = APIRouter(tags=["transfers"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _get_sub_client(db, account_id):
    """Return a Twilio Client configured with the org's sub-account creds."""
    from twilio.rest import Client  # lazy import

    row = await db.fetchrow(
        "SELECT twilio_subaccount_sid, twilio_subaccount_auth_token "
        "FROM accounts WHERE account_id = $1",
        account_id,
    )
    if not row or not row["twilio_subaccount_sid"] or not row["twilio_subaccount_auth_token"]:
        raise HTTPException(
            status_code=400,
            detail="Twilio credentials not configured for this account",
        )
    return Client(row["twilio_subaccount_sid"], row["twilio_subaccount_auth_token"])


# ---------------------------------------------------------------------------
# POST /whisper-twiml
# ---------------------------------------------------------------------------
# Called by Twilio as a webhook URL during warm transfers -- no Bearer auth.

@router.post("/whisper-twiml")
async def whisper_twiml(
    summary: str = Query("No summary provided."),
):
    """
    Generates the TwiML for the whisper message in a warm transfer.

    Twilio calls this URL before connecting the transfer recipient to the
    caller.  The ``summary`` query-param carries a brief context message so
    the human agent knows why the call was transferred.
    """
    response = VoiceResponse()
    response.say(
        f"This is a transfer from your AI assistant. "
        f"Here is a summary of the customer's request: {summary}"
    )
    response.pause(length=1)
    response.say("Connecting you to the customer now.")
    return PlainTextResponse(str(response), media_type="text/xml")


# ---------------------------------------------------------------------------
# POST /transfer-call
# ---------------------------------------------------------------------------

class TransferCallRequest(BaseModel):
    call_sid: str
    call_transfer_number: str
    department: str = "customer_service"
    provider: str = "twilio"
    transfer_type: str = "cold"  # "cold" or "warm"
    summary: Optional[str] = None


@router.post("/transfer-call")
async def transfer_call(
    request: Request,
    db: DBDep,
    redis: RedisDep,
):
    """
    Transfer an active Twilio call to another phone number.

    Supports two transfer modes:

    * **cold** (default) -- the caller is immediately connected to the
      transfer number with a simple ``<Dial>`` TwiML.
    * **warm** -- the transfer number first hears a whisper message
      (via ``/whisper-twiml``) summarising the conversation before being
      connected to the caller.

    The call's account is resolved from the ``calls`` table so the correct
    Twilio sub-account credentials are used for the API call.
    """
    try:
        payload = await request.json()
        call_sid = payload.get("call_sid")
        call_transfer_number = payload.get("call_transfer_number")
        department = payload.get("department", "customer_service")
        provider = payload.get("provider", "twilio")
        transfer_type = payload.get("transfer_type", "cold")
        summary = payload.get("summary")

        # ------------------------------------------------------------------
        # Validate required fields
        # ------------------------------------------------------------------
        if not call_sid or not call_transfer_number:
            return _json(
                {"status": "error", "message": "Missing required parameters: call_sid or call_transfer_number"},
                400,
            )

        logger.info(
            "Transferring call %s to %s (department: %s, type: %s)",
            call_sid, call_transfer_number, department, transfer_type,
        )

        # ------------------------------------------------------------------
        # Look up the owning account from the call record
        # ------------------------------------------------------------------
        call_info = await db.fetchrow(
            "SELECT account_id FROM calls WHERE call_sid = $1",
            call_sid,
        )
        if not call_info:
            return _json({"status": "error", "message": "Call not found"}, 404)

        account_id = call_info["account_id"]

        # ------------------------------------------------------------------
        # Build the Twilio client for this account
        # ------------------------------------------------------------------
        client = await _get_sub_client(db, account_id)

        if provider != "twilio":
            return _json(
                {"status": "error", "message": f"Provider {provider} not supported for transfers"},
                400,
            )

        # ------------------------------------------------------------------
        # Build TwiML
        # ------------------------------------------------------------------
        twiml_response = VoiceResponse()

        if transfer_type == "warm" and summary:
            # Warm transfer: the recipient hears a whisper summary first
            encoded_summary = urllib.parse.quote(summary)
            whisper_url = f"{settings.base_url}/whisper-twiml?summary={encoded_summary}"

            dial = Dial()
            dial.number(call_transfer_number, url=whisper_url)
            twiml_response.append(dial)
        else:
            # Cold transfer (or warm without a summary -- fall back to cold)
            twiml_response.dial(call_transfer_number)

        # ------------------------------------------------------------------
        # Update the in-progress call with the new TwiML
        # ------------------------------------------------------------------
        updated_call = client.calls(call_sid).update(twiml=str(twiml_response))
        logger.info("Call transfer initiated: %s", updated_call.sid)

        # ------------------------------------------------------------------
        # Track the transfer in Redis (24-hour TTL)
        # ------------------------------------------------------------------
        await redis.set(
            f"call_transfer:{call_sid}",
            json.dumps({
                "transfer_number": call_transfer_number,
                "transfer_time": datetime.now().isoformat(),
                "original_call_sid": call_sid,
                "department": department,
                "type": transfer_type,
            }),
            ex=86400,
        )

        # ------------------------------------------------------------------
        # Persist transfer metadata in the calls table
        # ------------------------------------------------------------------
        try:
            await db.execute(
                """
                UPDATE calls
                SET    transferred       = TRUE,
                       transfer_department = $1,
                       transfer_number   = $2,
                       transfer_time     = CURRENT_TIMESTAMP
                WHERE  call_sid = $3
                """,
                department,
                call_transfer_number,
                call_sid,
            )
        except Exception as db_error:
            logger.error("Error updating call record: %s", db_error)

        return _json({
            "status": "success",
            "message": f"Call transfer ({transfer_type}) initiated",
            "call_sid": call_sid,
            "department": department,
            "transfer_number": call_transfer_number,
        })

    except TwilioRestException as exc:
        logger.error("Twilio API error during transfer: %s", exc)
        return _json({"status": "error", "message": f"Twilio API error: {exc}"}, exc.status)

    except Exception as exc:
        logger.error("Error transferring call: %s", exc)
        return _json({"status": "error", "message": f"Internal server error: {exc}"}, 500)


# ---------------------------------------------------------------------------
# Tiny helper for JSONResponse without an extra import at module level.
# ---------------------------------------------------------------------------

def _json(body: dict, status_code: int = 200):
    from fastapi.responses import JSONResponse
    return JSONResponse(body, status_code=status_code)
