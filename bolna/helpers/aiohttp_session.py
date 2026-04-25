import aiohttp
import asyncio

_session: aiohttp.ClientSession | None = None
_lock = asyncio.Lock()


async def get_shared_aiohttp_session() -> aiohttp.ClientSession:
    global _session
    if _session is None or _session.closed:
        async with _lock:
            if _session is None or _session.closed:
                connector = aiohttp.TCPConnector(limit=200, limit_per_host=100, keepalive_timeout=30)
                _session = aiohttp.ClientSession(connector=connector, cookie_jar=aiohttp.DummyCookieJar())
    return _session
