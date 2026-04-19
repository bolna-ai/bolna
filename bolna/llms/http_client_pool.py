import httpx

_pool: dict[tuple, httpx.AsyncClient] = {}


def get_shared_http_client(base_url: str | None = None, http2: bool = True) -> httpx.AsyncClient:
    key = (base_url, http2)
    if key not in _pool:
        limits = httpx.Limits(max_connections=200, max_keepalive_connections=200, keepalive_expiry=30)
        _pool[key] = httpx.AsyncClient(limits=limits, timeout=httpx.Timeout(600.0, connect=10.0), http2=http2)
    return _pool[key]
