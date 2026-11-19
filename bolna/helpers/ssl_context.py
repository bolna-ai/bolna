import ssl
import threading

import aiohttp
import certifi

_ssl_context: ssl.SSLContext | None = None
_lock = threading.Lock()


def _build_ssl_context() -> ssl.SSLContext:
    """System trust store plus certifi's roots, so a stale OS bundle cannot reject a current public CA."""
    context = ssl.create_default_context()
    context.load_verify_locations(cafile=certifi.where())
    return context


def get_ssl_context(url: str | None = None) -> ssl.SSLContext | None:
    if url is not None and not url.startswith("wss://"):
        return None

    global _ssl_context
    if _ssl_context is None:
        with _lock:
            if _ssl_context is None:
                _ssl_context = _build_ssl_context()
    return _ssl_context


def tcp_connector(**kwargs) -> aiohttp.TCPConnector:
    """aiohttp connector that verifies against the same trust store as the websocket and tool-call clients."""
    return aiohttp.TCPConnector(ssl=get_ssl_context(), **kwargs)
