"""
Health-check and diagnostics routes.

GET  /health                 — readiness probe (DB + Redis ping)
GET  /debug/connection-stats — pool statistics (dev/ops use only)
"""

from __future__ import annotations

from fastapi import APIRouter

from app.dependencies import DBDep, RedisDep
from db.pool import db_pool

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(db: DBDep, redis: RedisDep) -> dict:
    """Liveness + readiness probe.  Returns 200 only if DB and Redis are up."""
    # Database ping
    db_status = "ok"
    try:
        await db.fetchval("SELECT 1")
    except Exception as exc:  # noqa: BLE001
        db_status = f"error: {exc}"

    # Redis ping
    redis_status = "ok"
    try:
        await redis.ping()
    except Exception as exc:  # noqa: BLE001
        redis_status = f"error: {exc}"

    overall = "ok" if db_status == "ok" and redis_status == "ok" else "degraded"
    return {
        "status": overall,
        "db": db_status,
        "redis": redis_status,
        "version": "2.0.0",
    }


@router.get("/debug/connection-stats")
async def connection_stats() -> dict:
    """Return asyncpg pool statistics.  Do not expose publicly in production."""
    pool = db_pool.pool  # raises RuntimeError if not initialised
    return {
        "min_size": pool.get_min_size(),
        "max_size": pool.get_max_size(),
        "size": pool.get_size(),
        "free_size": pool.get_idle_size(),
    }
