"""
Rate limiter — enforces per-org concurrent call limits.

Usage (in routers/calls.py):
    from app.middleware.rate_limiter import check_call_limit, release_call_limit

    @router.post("/call")
    async def initiate_call(...):
        await check_call_limit(redis, org_id, max_concurrent)
        try:
            ...
        finally:
            await release_call_limit(redis, org_id)

Redis key: ``concurrent_calls:{org_id}`` (integer counter)

main_server.py reference: search "max_concurrent" / "organisation_limits"
"""

from __future__ import annotations

from fastapi import HTTPException, status
from redis.asyncio import Redis

_KEY_PREFIX = "concurrent_calls"


def _key(org_id: str) -> str:
    return f"{_KEY_PREFIX}:{org_id}"


async def get_concurrent_calls(redis: Redis, org_id: str) -> int:
    """Return the current concurrent call count for an org."""
    val = await redis.get(_key(org_id))
    return int(val) if val else 0


async def check_call_limit(
    redis: Redis,
    org_id: str,
    max_concurrent: int,
) -> None:
    """
    Atomically increment the counter and reject if over the limit.

    Uses a Lua script for atomic check-and-increment to avoid races.
    Raises HTTP 429 if the limit would be exceeded.
    """
    lua = """
    local current = redis.call('INCR', KEYS[1])
    if current > tonumber(ARGV[1]) then
        redis.call('DECR', KEYS[1])
        return -1
    end
    redis.call('EXPIRE', KEYS[1], 3600)  -- safety TTL: auto-expire after 1h
    return current
    """
    result = await redis.eval(lua, 1, _key(org_id), max_concurrent)  # type: ignore[arg-type]
    if result == -1:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Concurrent call limit ({max_concurrent}) reached for this account",
        )


async def release_call_limit(redis: Redis, org_id: str) -> None:
    """Decrement the counter when a call ends (always call in a finally block)."""
    key = _key(org_id)
    current = await redis.decr(key)
    if current < 0:
        # Guard against going negative (e.g. if startup missed a cleanup)
        await redis.set(key, 0)
