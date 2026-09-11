import threading
import httpx

_pool: dict[tuple, httpx.AsyncClient] = {}
_lock = threading.Lock()


def get_shared_http_client(base_url: str | None = None, http2: bool = True) -> httpx.AsyncClient:
    key = (base_url, http2)
    client = _pool.get(key)
    if client is None:
        with _lock:
            client = _pool.get(key)
            if client is None:
                # 30s let idle connections go stale and hang silently on reuse; retries=1 is
                # the backstop if one still slips through.
                limits = httpx.Limits(max_connections=200, max_keepalive_connections=200, keepalive_expiry=12)
                transport = httpx.AsyncHTTPTransport(http2=http2, limits=limits, retries=1)
                client = httpx.AsyncClient(transport=transport, timeout=httpx.Timeout(600.0, connect=10.0))
                _pool[key] = client
    return client


_sync_pool: dict[tuple, httpx.Client] = {}


def get_shared_sync_http_client(base_url: str | None = None, http2: bool = True) -> httpx.Client:
    """Sync twin of get_shared_http_client, for SDK clients that reject an AsyncClient."""
    key = (base_url, http2)
    client = _sync_pool.get(key)
    if client is None:
        with _lock:
            client = _sync_pool.get(key)
            if client is None:
                limits = httpx.Limits(max_connections=200, max_keepalive_connections=200, keepalive_expiry=30)
                client = httpx.Client(limits=limits, timeout=httpx.Timeout(600.0, connect=10.0), http2=http2)
                _sync_pool[key] = client
    return client
