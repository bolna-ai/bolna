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
                # A pool entry keyed on a low-traffic base_url (e.g. a PTU deployment mostly hit
                # by a high-volume channel elsewhere) can go idle for minutes between requests on
                # this key specifically. 30s let a connection outlive whatever silently drops it
                # first, and a request handed that connection just hung with no exception until
                # something eventually unstuck it. 10s keeps connections fresher than any idle
                # timeout we've seen; retries=1 is the backstop for a stale one slipping through
                # anyway — it fails fast onto a new connection instead of hanging.
                limits = httpx.Limits(max_connections=200, max_keepalive_connections=200, keepalive_expiry=10)
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
