"""
asyncpg connection pool singleton.

The pool is initialised once during application startup (via
``startup_db()`` in ``app/dependencies.py``) and closed during
shutdown.  All query functions receive a checked-out connection via
FastAPI's ``Depends(get_db)`` dependency.
"""

from __future__ import annotations

import logging

import asyncpg

logger = logging.getLogger(__name__)


class DatabasePool:
    """Thin wrapper around an asyncpg pool giving a clear init/close API."""

    def __init__(self) -> None:
        self._pool: asyncpg.Pool | None = None

    @property
    def pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError(
                "Database pool not initialised.  "
                "Call await db_pool.init(...) during app startup."
            )
        return self._pool

    async def init(
        self,
        dsn: str,
        min_size: int = 2,
        max_size: int = 20,
    ) -> None:
        """Create the pool.  Idempotent — safe to call multiple times."""
        if self._pool is not None:
            return
        self._pool = await asyncpg.create_pool(
            dsn=dsn,
            min_size=min_size,
            max_size=max_size,
            command_timeout=30,
        )
        logger.info(
            "asyncpg pool created (min=%d, max=%d)",
            min_size,
            max_size,
        )

    async def close(self) -> None:
        """Gracefully close all connections in the pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            logger.info("asyncpg pool closed")


# Module-level singleton — import this everywhere
db_pool = DatabasePool()
