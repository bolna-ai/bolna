"""Google Cloud OAuth2 tokens via ADC for Vertex endpoints, which take a bearer token rather
than the generativelanguage API key the other Gemini integrations use."""

import asyncio
import threading

import google.auth
import google.auth.transport.requests

_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]

_lock = threading.Lock()
_credentials = None
_adc_project = None


def _refresh_locked():
    global _credentials, _adc_project
    if _credentials is None:
        _credentials, _adc_project = google.auth.default(scopes=_SCOPES)
    # `valid` applies its own clock skew, so refreshing when it flips covers expiry.
    if not _credentials.valid:
        _credentials.refresh(google.auth.transport.requests.Request())
    return _credentials.token, _adc_project


async def get_gcp_credentials():
    """(access_token, adc_project) from ADC, in a thread since load and refresh block."""

    def _sync():
        with _lock:
            return _refresh_locked()

    return await asyncio.get_event_loop().run_in_executor(None, _sync)
