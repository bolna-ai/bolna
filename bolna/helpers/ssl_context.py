import ssl
import threading

_ssl_context: ssl.SSLContext | None = None
_lock = threading.Lock()


def get_ssl_context() -> ssl.SSLContext:
    global _ssl_context
    if _ssl_context is None:
        with _lock:
            if _ssl_context is None:
                _ssl_context = ssl.create_default_context()
    return _ssl_context
