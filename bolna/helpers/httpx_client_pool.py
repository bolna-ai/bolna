"""Shared httpx.AsyncClient pool — one client per (origin, key) to reuse TCP+TLS connections."""

import atexit
import asyncio
from typing import Dict, Tuple, Optional
from urllib.parse import urlparse

import httpx

from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)


def _normalize_origin(url: Optional[str]) -> str:
    if not url:
        return "_default"
    parsed = urlparse(url)
    return parsed.hostname or parsed.netloc or url


class HttpxClientPool:
    """Singleton pool of httpx.AsyncClient instances keyed by (origin, key).

    Any service that makes repeated HTTP calls to the same host can use this
    to avoid redundant TCP+TLS handshakes and leverage HTTP/2 multiplexing.
    """

    _clients: Dict[Tuple[str, str, bool], httpx.AsyncClient] = {}

    @classmethod
    def get_client(
        cls,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        http2: bool = True,
    ) -> httpx.AsyncClient:
        """Return a cached or new httpx.AsyncClient for the given origin+key+protocol."""
        origin = _normalize_origin(base_url)
        key = (origin, api_key or "", http2)

        client = cls._clients.get(key)
        if client is not None:
            return client

        limits = httpx.Limits(
            max_connections=100,
            max_keepalive_connections=100,
            keepalive_expiry=120,
        )
        client = httpx.AsyncClient(
            limits=limits,
            timeout=httpx.Timeout(600.0, connect=10.0),
            http2=http2,
        )
        cls._clients[key] = client
        proto = "h2" if http2 else "h1.1"
        logger.info(f"HttpxClientPool: new {proto} client for {origin} (pool size: {len(cls._clients)})")
        return client

    @classmethod
    async def close_all(cls) -> None:
        """Close every pooled client."""
        for _, client in list(cls._clients.items()):
            try:
                await client.aclose()
            except Exception:
                pass
        cls._clients.clear()
        logger.info("HttpxClientPool: all clients closed")


def _atexit_cleanup() -> None:
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(HttpxClientPool.close_all())
        else:
            loop.run_until_complete(HttpxClientPool.close_all())
    except Exception:
        pass


atexit.register(_atexit_cleanup)
