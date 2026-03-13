"""
services/agent_service.py -- agent configuration loading, caching, and
Twilio inbound webhook setup.

Ported from main_server.py:
  - Agent config loading via Redis cache -> DB fallback (various places)
  - Redis cache set/invalidate for agent configs
  - Inbound Twilio phone number webhook configuration (~lines 1126-1161)
"""

from __future__ import annotations

import json
import logging
import uuid

import asyncpg
from redis.asyncio import Redis

from app.config import get_settings
from db.queries.agents import get_agent
from services.twilio_helpers import get_callback_urls, get_twilio_client

logger = logging.getLogger(__name__)
settings = get_settings()

# TTL for cached agent configs in Redis
_AGENT_CACHE_TTL = 3600  # 1 hour


# ---------------------------------------------------------------------------
# Agent config cache
# ---------------------------------------------------------------------------

async def get_agent_config(
    db: asyncpg.Connection,
    redis: Redis,
    org_id: str,
    agent_id: str,
) -> dict | None:
    """Load agent config from Redis cache, falling back to the DB.

    The cache key is the ``agent_id`` string.  The value is the full
    ``agent_config`` JSON blob stored in the ``agents`` table.

    Returns ``None`` when the agent does not exist (or has been soft-deleted).
    """
    # 1. Try Redis cache first
    cached = await redis.get(agent_id)
    if cached:
        try:
            config = json.loads(cached)
            logger.debug("Agent config cache hit for %s", agent_id)
            return config
        except (json.JSONDecodeError, TypeError):
            logger.warning("Corrupt agent cache for %s, falling back to DB", agent_id)
            await redis.delete(agent_id)

    # 2. DB fallback
    aid = uuid.UUID(agent_id) if isinstance(agent_id, str) else agent_id
    oid = uuid.UUID(org_id) if isinstance(org_id, str) else org_id

    row = await get_agent(db, aid, oid)
    if not row:
        logger.warning("Agent %s not found in DB for org %s", agent_id, org_id)
        return None

    # The agent_config column is stored as jsonb; asyncpg returns it as str
    raw_config = row["agent_config"]
    if isinstance(raw_config, str):
        config = json.loads(raw_config)
    elif isinstance(raw_config, dict):
        config = raw_config
    else:
        logger.error("Unexpected agent_config type for %s: %s", agent_id, type(raw_config))
        return None

    # Merge top-level DB fields the caller may need
    config["is_compliant"] = row.get("is_compliant", False)
    config["webhook_url"] = row.get("webhook_url")
    config["call_direction"] = row.get("call_direction")
    config["inbound_phone_number"] = row.get("inbound_phone_number")

    # 3. Populate cache
    await cache_agent_config(redis, agent_id, config)

    return config


async def cache_agent_config(
    redis: Redis,
    agent_id: str,
    config: dict,
    ttl: int = _AGENT_CACHE_TTL,
) -> None:
    """Store an agent config dict in Redis.

    Called after loading from DB and also after creating/updating an agent
    so the cache stays warm.
    """
    try:
        await redis.set(agent_id, json.dumps(config), ex=ttl)
        logger.debug("Cached agent config for %s (ttl=%ds)", agent_id, ttl)
    except Exception:
        logger.exception("Failed to cache agent config for %s", agent_id)


async def invalidate_agent_cache(
    redis: Redis,
    agent_id: str,
) -> None:
    """Remove the agent config from Redis.

    Call this after updating or deleting an agent so stale config is not
    served to subsequent calls.
    """
    await redis.delete(agent_id)
    logger.info("Invalidated agent cache for %s", agent_id)


# ---------------------------------------------------------------------------
# Inbound Twilio webhook configuration
# ---------------------------------------------------------------------------

async def configure_inbound_twilio(
    db: asyncpg.Connection,
    redis: Redis,
    org_id: str,
    agent_id: str,
    inbound_phone_number: str,
) -> dict:
    """Configure the Twilio webhook for an inbound agent's phone number.

    This sets both the ``voice_url`` (for receiving calls) and the
    ``status_callback`` (for billing/call tracking) on the Twilio phone
    number resource.

    Also stores the ``inbound_mapping:{phone_number} -> agent_id`` key
    in Redis so the inbound call handler can quickly resolve which agent
    to use without hitting the DB.

    Ported from main_server.py create_agent handler (~lines 1126-1161).

    Parameters
    ----------
    db : asyncpg.Connection
    redis : Redis
    org_id : str
        Account UUID.
    agent_id : str
        Agent UUID.
    inbound_phone_number : str
        E.164 formatted phone number to configure.

    Returns
    -------
    dict
        ``{"status": "configured", "phone_number": ..., "voice_url": ...}``

    Raises
    ------
    ValueError
        When the phone number is not found in the Twilio account.
    """
    from twilio.base.exceptions import TwilioRestException

    account_id = uuid.UUID(org_id) if isinstance(org_id, str) else org_id
    twilio_client = await get_twilio_client(db, account_id)
    app_callback_url, websocket_url = get_callback_urls()

    # Store inbound mapping so the callback handler can resolve the agent
    mapping_key = f"inbound_mapping:{inbound_phone_number}"
    await redis.set(mapping_key, agent_id)

    # Find the phone number in Twilio's account
    try:
        numbers = twilio_client.incoming_phone_numbers.list(
            phone_number=inbound_phone_number
        )
    except TwilioRestException as exc:
        logger.error(
            "Twilio error listing phone numbers for %s: %s",
            inbound_phone_number, exc,
        )
        raise

    if not numbers:
        raise ValueError(
            f"Phone number {inbound_phone_number} not found in Twilio account for org {org_id}"
        )

    incoming = numbers[0]
    voice_url = (
        f"{app_callback_url}/inbound_call"
        f"?agent_id={agent_id}&account_id={org_id}"
    )
    status_callback = f"{app_callback_url}/call_status_callback"

    try:
        incoming.update(
            voice_url=voice_url,
            voice_method="POST",
            status_callback=status_callback,
            status_callback_method="POST",
        )
    except TwilioRestException as exc:
        logger.error(
            "Twilio error configuring webhook for %s: %s",
            inbound_phone_number, exc,
        )
        raise

    logger.info(
        "Configured inbound number %s with voice_url=%s for agent %s",
        inbound_phone_number, voice_url, agent_id,
    )

    return {
        "status": "configured",
        "phone_number": inbound_phone_number,
        "voice_url": voice_url,
        "status_callback": status_callback,
    }
