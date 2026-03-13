"""
services/twilio_helpers.py -- shared Twilio utilities used across services.

Extracted from call_service.py to break a circular import between
call_service and agent_service.
"""

from __future__ import annotations

import uuid

import asyncpg
from twilio.rest import Client

from app.config import get_settings

settings = get_settings()


def get_callback_urls() -> tuple[str, str]:
    """Return (app_callback_url, websocket_url) derived from settings.base_url.

    In production ``base_url`` is ``https://api.kallabot.com`` and the
    WebSocket URL is derived by replacing the scheme with ``wss://`` and
    swapping the ``api.`` prefix for ``ws.``.  In dev the base URL is used
    as-is for both (Twilio connects through ngrok).
    """
    base = settings.base_url.rstrip("/")
    if "kallabot.com" in base:
        app_callback_url = "https://api.kallabot.com"
        websocket_url = "wss://ws.kallabot.com"
    else:
        # Local / ngrok: same host for both
        app_callback_url = base
        websocket_url = base.replace("https://", "wss://").replace("http://", "ws://")
    return app_callback_url, websocket_url


async def get_twilio_client(
    db: asyncpg.Connection,
    account_id: str | uuid.UUID,
) -> Client:
    """Return a Twilio ``Client`` using the sub-account credentials stored
    in the ``accounts`` table for *account_id*.

    Raises ``ValueError`` when credentials are missing.
    """
    aid = uuid.UUID(account_id) if isinstance(account_id, str) else account_id
    row = await db.fetchrow(
        "SELECT twilio_subaccount_sid, twilio_subaccount_auth_token "
        "FROM accounts WHERE account_id = $1",
        aid,
    )
    if not row or not row["twilio_subaccount_sid"] or not row["twilio_subaccount_auth_token"]:
        raise ValueError(f"Twilio credentials not configured for account {aid}")

    return Client(row["twilio_subaccount_sid"], row["twilio_subaccount_auth_token"])
