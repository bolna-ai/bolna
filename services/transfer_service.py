"""
services/transfer_service.py -- call transfer execution via Twilio.

Ported from main_server.py:
  - POST /transfer-call handler (~lines 6892-7005)
  - POST /whisper-twiml handler (~lines 6876-6885)

Supports both cold transfers (immediate redirect) and warm transfers
(pre-call whisper summarising the conversation before connecting).
"""

from __future__ import annotations

import json
import logging
import urllib.parse
from datetime import datetime

import asyncpg
from redis.asyncio import Redis
from twilio.base.exceptions import TwilioRestException
from twilio.twiml.voice_response import Dial, VoiceResponse

from db.queries.calls import get_call_by_sid
from services.call_service import _get_callback_urls, get_twilio_client

logger = logging.getLogger(__name__)

# Redis TTL for transfer tracking metadata
_TRANSFER_TTL = 86400  # 24 hours


# ---------------------------------------------------------------------------
# TwiML generation
# ---------------------------------------------------------------------------

def generate_whisper_twiml(summary: str = "No summary provided.") -> str:
    """Build TwiML XML for a pre-call whisper message in a warm transfer.

    The whisper is played to the *transfer target* (e.g. a human agent)
    before they are connected to the caller, giving them context about
    the conversation so far.

    Ported from main_server.py POST /whisper-twiml (~line 6876).

    Parameters
    ----------
    summary : str
        A natural-language summary of the conversation to relay to the
        transfer target.

    Returns
    -------
    str
        TwiML XML string.
    """
    response = VoiceResponse()
    response.say(
        f"This is a transfer from your AI assistant. "
        f"Here is a summary of the customer's request: {summary}"
    )
    response.pause(length=1)
    response.say("Connecting you to the customer now.")
    return str(response)


# ---------------------------------------------------------------------------
# Call transfer
# ---------------------------------------------------------------------------

async def transfer_call(
    db: asyncpg.Connection,
    redis: Redis,
    call_sid: str,
    transfer_number: str,
    transfer_type: str = "cold",
    department: str = "customer_service",
    summary: str | None = None,
    provider: str = "twilio",
) -> dict:
    """Execute a call transfer via the Twilio API.

    Supports two modes:

    * **Cold transfer** -- the caller is immediately redirected to the
      ``transfer_number`` with a standard ``<Dial>`` verb.
    * **Warm transfer** -- a whisper URL is attached so the transfer
      target hears a spoken summary of the conversation before being
      bridged to the caller.

    Ported from main_server.py POST /transfer-call (~lines 6892-7005).

    Parameters
    ----------
    db : asyncpg.Connection
    redis : Redis
    call_sid : str
        Twilio Call SID of the active call to transfer.
    transfer_number : str
        E.164 destination number for the transfer.
    transfer_type : str
        ``"cold"`` (default) or ``"warm"``.
    department : str
        Logical department label stored for analytics.
    summary : str | None
        Conversation summary (required for warm transfers).
    provider : str
        Telephony provider.  Currently only ``"twilio"`` is supported.

    Returns
    -------
    dict
        ``{"status": "success", "call_sid": ..., "department": ..., ...}``

    Raises
    ------
    ValueError
        When required parameters are missing or the provider is unsupported.
    TwilioRestException
        When Twilio rejects the transfer request.
    """
    if not call_sid or not transfer_number:
        raise ValueError("Missing required parameters: call_sid and transfer_number")

    if provider != "twilio":
        raise ValueError(f"Provider '{provider}' is not supported for transfers")

    # Look up the call to find the account
    call_row = await get_call_by_sid(db, call_sid)
    if not call_row:
        raise ValueError(f"Call {call_sid} not found")

    account_id = call_row["account_id"]

    logger.info(
        "Transferring call %s to %s (department=%s, type=%s)",
        call_sid, transfer_number, department, transfer_type,
    )

    # Build TwiML for the transfer
    twilio_client = await get_twilio_client(db, account_id)
    twiml_response = VoiceResponse()

    if transfer_type == "warm" and summary:
        app_callback_url, _ = _get_callback_urls()
        encoded_summary = urllib.parse.quote(summary)
        whisper_url = f"{app_callback_url}/whisper-twiml?summary={encoded_summary}"

        dial = Dial()
        dial.number(transfer_number, url=whisper_url)
        twiml_response.append(dial)
    else:
        # Cold transfer (or warm without summary falls back to cold)
        twiml_response.dial(transfer_number)

    # Update the in-progress call with the new TwiML
    try:
        updated_call = twilio_client.calls(call_sid).update(
            twiml=str(twiml_response)
        )
    except TwilioRestException as exc:
        logger.error("Twilio error transferring call %s: %s", call_sid, exc)
        raise

    logger.info("Call transfer initiated: %s", updated_call.sid)

    # Store transfer metadata in Redis for tracking/analytics
    transfer_meta = {
        "transfer_number": transfer_number,
        "transfer_time": datetime.utcnow().isoformat(),
        "original_call_sid": call_sid,
        "department": department,
        "type": transfer_type,
    }
    try:
        await redis.set(
            f"call_transfer:{call_sid}",
            json.dumps(transfer_meta),
            ex=_TRANSFER_TTL,
        )
    except Exception:
        logger.exception("Failed to store transfer metadata for %s", call_sid)

    # Update the call record in the database
    try:
        await db.execute(
            """UPDATE calls
               SET transferred = TRUE,
                   transfer_department = $1,
                   transfer_number = $2,
                   transfer_time = CURRENT_TIMESTAMP
               WHERE call_sid = $3""",
            department,
            transfer_number,
            call_sid,
        )
    except Exception:
        logger.exception("Failed to update call record for transfer %s", call_sid)

    return {
        "status": "success",
        "message": f"Call transfer ({transfer_type}) initiated",
        "call_sid": call_sid,
        "department": department,
        "transfer_number": transfer_number,
    }
