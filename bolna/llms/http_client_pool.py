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
                limits = httpx.Limits(max_connections=200, max_keepalive_connections=200, keepalive_expiry=30)
                client = httpx.AsyncClient(limits=limits, timeout=httpx.Timeout(600.0, connect=10.0), http2=http2)
                _pool[key] = client
    return client
