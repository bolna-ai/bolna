import threading
import httpx

from bolna.helpers.function_calling_helpers import guarded_transport

_pool: dict[tuple, httpx.AsyncClient] = {}
_lock = threading.Lock()


def get_shared_http_client(base_url: str | None = None, http2: bool = True, guarded: bool = False) -> httpx.AsyncClient:
    """``guarded`` for a customer-supplied base_url: the client then dials only allowed addresses."""
    key = (base_url, http2, guarded)
    client = _pool.get(key)
    if client is None:
        with _lock:
            client = _pool.get(key)
            if client is None:
                limits = httpx.Limits(max_connections=200, max_keepalive_connections=200, keepalive_expiry=30)
                timeout = httpx.Timeout(600.0, connect=10.0)
                if guarded:
                    transport = guarded_transport(limits=limits, http2=http2)
                    client = httpx.AsyncClient(timeout=timeout, transport=transport)
                else:
                    client = httpx.AsyncClient(limits=limits, timeout=timeout, http2=http2)
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
