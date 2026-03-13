"""
FastAPI dependency injection helpers.

Usage inside a route handler:

    @router.get("/example")
    async def example(
        org: OrgDep,
        db: DBDep,
        redis: RedisDep,
    ):
        row = await db.fetchrow("SELECT ...")
        ...
"""

from __future__ import annotations

import logging
import uuid
from typing import Annotated, AsyncGenerator

import asyncpg
import redis.asyncio as aioredis
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import get_settings
from db.pool import db_pool
from db.queries.organizations import get_org_by_api_key

logger = logging.getLogger(__name__)
settings = get_settings()

# ── DB lifecycle ──────────────────────────────────────────────────────────────

async def startup_db() -> None:
    """Initialise the asyncpg connection pool.  Called from lifespan()."""
    await db_pool.init(
        dsn=settings.database_url,
        min_size=settings.db_min_connections,
        max_size=settings.db_max_connections,
    )
    logger.info("Database pool initialised")


async def shutdown_db() -> None:
    """Close the asyncpg connection pool.  Called from lifespan()."""
    await db_pool.close()
    logger.info("Database pool closed")


# ── Redis lifecycle ───────────────────────────────────────────────────────────

_redis_client: aioredis.Redis | None = None


async def startup_redis() -> None:
    """Initialise the Redis client.  Called from lifespan()."""
    global _redis_client
    _redis_client = aioredis.from_url(
        settings.redis_url,
        encoding="utf-8",
        decode_responses=True,
    )
    await _redis_client.ping()
    logger.info("Redis connected")


async def shutdown_redis() -> None:
    """Close the Redis client.  Called from lifespan()."""
    global _redis_client
    if _redis_client:
        await _redis_client.aclose()
        _redis_client = None
    logger.info("Redis closed")


# ── Per-request dependency functions ─────────────────────────────────────────

async def get_db() -> AsyncGenerator[asyncpg.Connection, None]:
    """Yield a single checked-out connection from the pool."""
    async with db_pool.pool.acquire() as conn:
        yield conn


async def get_redis() -> aioredis.Redis:
    """Return the shared Redis client (already connected)."""
    if _redis_client is None:
        raise RuntimeError("Redis client not initialised — startup_redis() not called")
    return _redis_client


# ── Bearer-token auth ─────────────────────────────────────────────────────────

_bearer = HTTPBearer(auto_error=False)


async def get_current_org(
    credentials: HTTPAuthorizationCredentials | None = Security(_bearer),
    db: asyncpg.Connection = Depends(get_db),
) -> uuid.UUID:
    """
    Validate the Bearer token and return the matching account_id (UUID).

    The token must be a key that starts with ``sk_``.
    Raises HTTP 401 if the token is missing, malformed, or unknown.
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Bearer token",
        )

    token = credentials.credentials
    if not token.startswith("sk_"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key format",
        )

    row = await get_org_by_api_key(db, token)
    if not row:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

    return row["account_id"]  # uuid.UUID


# ── Annotated type aliases ────────────────────────────────────────────────────
# Use these in route signatures for cleaner code.

DBDep = Annotated[asyncpg.Connection, Depends(get_db)]
RedisDep = Annotated[aioredis.Redis, Depends(get_redis)]
OrgDep = Annotated[uuid.UUID, Depends(get_current_org)]
