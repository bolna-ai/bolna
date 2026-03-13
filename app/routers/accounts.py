"""
Account / org management routes.

Routes (exact paths matching main_server.py):
  POST   /account                           — create org (no auth)
  POST   /account/renew-api-key            — rotate API key (auth)
  DELETE /account                           — delete org + close Twilio (auth)
  GET    /account                           — org detail + agents list (auth)
  GET    /account/validate-call-configuration — Twilio call config check (auth)
  GET    /account/test-phone-validation     — test phone number callback config (auth)
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from app.config import get_settings
from app.dependencies import DBDep, OrgDep, RedisDep
from db.queries.organizations import (
    create_org,
    delete_account_cascade,
    get_account_agent_ids,
    get_account_agents_list,
    get_account_detail,
    get_account_twilio_sid,
    get_org_by_id,
    update_org_api_key,
)

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/account", tags=["accounts"])


def _master_twilio_client():
    from twilio.rest import Client  # lazy import — only needed when called
    return Client(settings.twilio_account_sid, settings.twilio_auth_token)


# ── POST /account ─────────────────────────────────────────────────────────────

@router.post("")
async def create_account(db: DBDep) -> dict:
    """
    Create a new org: provision a Twilio sub-account, store credentials,
    generate an API key, and return the three values the caller needs.

    No auth is required — this is the registration endpoint.
    """
    try:
        master = _master_twilio_client()
        sub = master.api.v2010.accounts.create(friendly_name="kallabot-subaccount")
    except Exception as exc:
        logger.exception("Twilio sub-account creation failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"Twilio error: {exc}")

    try:
        result = await create_org(
            db,
            twilio_subaccount_sid=sub.sid,
            twilio_subaccount_auth_token=sub.auth_token,
        )
    except Exception as exc:
        logger.exception("DB create_org failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    return result  # {account_id, api_key, twilio_subaccount_sid}


# ── POST /account/renew-api-key ───────────────────────────────────────────────

@router.post("/renew-api-key")
async def renew_api_key(org: OrgDep, db: DBDep) -> dict:
    """Rotate the API key for the authenticated org."""
    result = await update_org_api_key(db, org)
    if not result:
        raise HTTPException(status_code=404, detail="Account not found")
    return result  # {account_id, api_key}


# ── DELETE /account ───────────────────────────────────────────────────────────

@router.delete("")
async def delete_account(org: OrgDep, db: DBDep, redis: RedisDep) -> dict:
    """
    Cascade-delete the org from the DB, close its Twilio sub-account,
    and evict all agent keys from Redis.
    """
    # Fetch Twilio SID and agent IDs before deletion (rows gone after cascade).
    twilio_row = await get_account_twilio_sid(db, org)
    twilio_sid = twilio_row["twilio_subaccount_sid"] if twilio_row else None

    agent_id_rows = await get_account_agent_ids(db, org)

    await delete_account_cascade(db, org)

    # Close the Twilio sub-account (best-effort — non-fatal on error).
    if twilio_sid:
        try:
            _master_twilio_client().api.v2010.accounts(twilio_sid).update(
                status="closed"
            )
        except Exception as exc:
            logger.warning(
                "Could not close Twilio sub-account %s: %s", twilio_sid, exc
            )

    # Evict every agent's config blob from Redis.
    for row in agent_id_rows:
        try:
            await redis.delete(str(row["agent_id"]))
        except Exception as exc:
            logger.warning(
                "Redis delete failed for agent %s: %s", row["agent_id"], exc
            )

    return {"message": "Account and all associated data deleted successfully"}


# ── GET /account ──────────────────────────────────────────────────────────────

@router.get("")
async def get_account(org: OrgDep, db: DBDep) -> dict:
    """Return account details and the list of agents."""
    detail = await get_account_detail(db, org)
    if not detail:
        raise HTTPException(status_code=404, detail="Account not found")

    agents = await get_account_agents_list(db, org)

    return {
        "account_id": str(detail["account_id"]),
        "number_of_agents": detail["number_of_agents"],
        "total_calls": detail["total_calls"],
        "total_duration_minutes": round(
            float(detail["total_duration"] or 0) / 60, 2
        ),
        "total_cost": round(float(detail["total_cost"] or 0), 2),
        "agents": [
            {"agent_id": str(a["agent_id"]), "name": a["name"]}
            for a in (agents or [])
        ],
    }


# ── GET /account/validate-call-configuration ─────────────────────────────────

@router.get("/validate-call-configuration")
async def validate_call_configuration(
    org: OrgDep,
    db: DBDep,
    phone_number: Optional[str] = Query(None),
) -> dict:
    """
    Check whether the Twilio call status-callback is properly configured
    for this account.  Optionally scope to a specific inbound phone number.
    """
    # Need sub-account credentials — get_org_by_id returns the full row.
    full_row = await get_org_by_id(db, str(org))
    if not full_row:
        raise HTTPException(status_code=404, detail="Account not found")

    twilio_sid = full_row["twilio_subaccount_sid"]
    twilio_auth = full_row["twilio_subaccount_auth_token"]

    result: dict = {
        "account_id": str(org),
        "twilio_subaccount_sid": twilio_sid,
        "phone_number": phone_number,
        "callback_configured": False,
        "details": [],
    }

    if not (twilio_sid and twilio_auth):
        return result

    try:
        from twilio.rest import Client

        sub_client = Client(twilio_sid, twilio_auth)

        if phone_number:
            incoming = sub_client.incoming_phone_numbers.list(
                phone_number=phone_number
            )
        else:
            incoming = sub_client.incoming_phone_numbers.list()

        callback_url_base = f"{settings.base_url}/call/inbound"
        any_configured = False

        for number in incoming:
            configured = bool(
                number.voice_url and number.voice_url.startswith(callback_url_base)
            )
            if configured:
                any_configured = True
            result["details"].append(
                {
                    "phone_number": number.phone_number,
                    "voice_url": number.voice_url or "",
                    "configured": configured,
                }
            )

        result["callback_configured"] = any_configured

    except Exception as exc:
        logger.exception("validate-call-configuration failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    return result


# ── GET /account/test-phone-validation ────────────────────────────────────────

@router.get("/test-phone-validation")
async def test_phone_validation(
    org: OrgDep,
    db: DBDep,
    phone_number: str = Query(..., description="Phone number to test (E.164 format)"),
) -> dict:
    """
    Simulate what happens when an inbound call is received on *phone_number*.
    Returns whether the number has a properly configured status callback so
    the call would be connected.
    """
    full_row = await get_org_by_id(db, str(org))
    if not full_row:
        raise HTTPException(status_code=404, detail="Account not found")

    twilio_sid = full_row["twilio_subaccount_sid"]
    twilio_auth = full_row["twilio_subaccount_auth_token"]

    if not (twilio_sid and twilio_auth):
        return {
            "status": "invalid",
            "message": "No Twilio credentials configured",
            "call_would_connect": False,
            "test_result": "FAIL - No Twilio sub-account",
        }

    try:
        from twilio.rest import Client

        sub_client = Client(twilio_sid, twilio_auth)
        incoming = sub_client.incoming_phone_numbers.list(phone_number=phone_number)
        if not incoming:
            return {
                "status": "invalid",
                "message": f"Phone number {phone_number} not found on this account",
                "call_would_connect": False,
                "test_result": "FAIL - Number not found",
            }

        number = incoming[0]
        expected_base = f"{settings.base_url}/call_status_callback"
        callback_ok = bool(
            number.status_callback
            and expected_base in number.status_callback
        )

        if callback_ok:
            return {
                "status": "valid",
                "message": f"Phone number {phone_number} has valid status callback configuration",
                "call_would_connect": True,
                "test_result": "PASS - This call would be connected normally",
            }
        else:
            return {
                "status": "invalid",
                "message": f"Status callback missing or incorrect on {phone_number}",
                "call_would_connect": False,
                "test_result": "FAIL - This call would be rejected with TwiML <Reject>",
            }

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("test-phone-validation failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to test phone validation")
