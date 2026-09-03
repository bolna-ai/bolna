"""Unit tests for per-call Deepgram host/protocol override used by multi-cloud routing."""

from urllib.parse import urlparse

from bolna.transcriber.deepgram_transcriber import DeepgramTranscriber


def _make(model="nova-2", language="en", **kwargs):
    return DeepgramTranscriber(
        telephony_provider="plivo",
        model=model,
        language=language,
        stream=True,
        **kwargs,
    )


def test_default_host_when_unset(monkeypatch):
    monkeypatch.delenv("DEEPGRAM_HOST", raising=False)
    monkeypatch.delenv("DEEPGRAM_FLUX_HOST", raising=False)
    monkeypatch.delenv("DEEPGRAM_HOST_PROTOCOL", raising=False)
    nova = urlparse(_make(model="nova-2")._get_nova_ws_url())
    flux = urlparse(_make(model="flux-general-en")._get_flux_ws_url())
    assert (nova.scheme, nova.netloc) == ("wss", "api.deepgram.com")
    assert (flux.scheme, flux.netloc) == ("wss", "api.deepgram.com")


def test_kwarg_host_overrides_env(monkeypatch):
    monkeypatch.setenv("DEEPGRAM_HOST", "env-host:8080")
    monkeypatch.setenv("DEEPGRAM_FLUX_HOST", "env-flux:8080")
    monkeypatch.setenv("DEEPGRAM_HOST_PROTOCOL", "wss")
    nova = urlparse(_make(model="nova-2", deepgram_host="kwarg-host:8080", deepgram_host_protocol="ws")._get_nova_ws_url())
    flux = urlparse(
        _make(model="flux-general-en", deepgram_flux_host="kwarg-flux:8080", deepgram_host_protocol="ws")._get_flux_ws_url()
    )
    assert (nova.scheme, nova.netloc) == ("ws", "kwarg-host:8080")
    assert (flux.scheme, flux.netloc) == ("ws", "kwarg-flux:8080")


def test_env_used_when_no_kwarg(monkeypatch):
    monkeypatch.setenv("DEEPGRAM_HOST", "env-host:8080")
    monkeypatch.setenv("DEEPGRAM_HOST_PROTOCOL", "ws")
    nova = urlparse(_make(model="nova-2")._get_nova_ws_url())
    assert (nova.scheme, nova.netloc) == ("ws", "env-host:8080")


def test_flux_host_independent_of_nova_kwarg(monkeypatch):
    monkeypatch.delenv("DEEPGRAM_HOST", raising=False)
    monkeypatch.delenv("DEEPGRAM_FLUX_HOST", raising=False)
    monkeypatch.delenv("DEEPGRAM_HOST_PROTOCOL", raising=False)
    # deepgram_host set but deepgram_flux_host omitted -> flux still resolves to its own default.
    flux = urlparse(_make(model="flux-general-en", deepgram_host="nova-only:8080")._get_flux_ws_url())
    assert flux.netloc == "api.deepgram.com"
