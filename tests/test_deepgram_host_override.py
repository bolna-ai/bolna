"""The per-call Deepgram endpoint is a name resolved against DEEPGRAM_ENDPOINT_<NAME>_* env, never a raw host."""

from urllib.parse import urlparse

import pytest

from bolna.transcriber.deepgram_transcriber import DeepgramTranscriber


def _make(model="nova-2", language="en", **kwargs):
    return DeepgramTranscriber(
        telephony_provider="plivo",
        model=model,
        language=language,
        stream=True,
        **kwargs,
    )


def _nova(**kwargs):
    url = urlparse(_make(model="nova-2", **kwargs)._get_nova_ws_url())
    return url.scheme, url.netloc


def _flux(**kwargs):
    url = urlparse(_make(model="flux-general-en", **kwargs)._get_flux_ws_url())
    return url.scheme, url.netloc


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("DEEPGRAM_HOST", "default:8080")
    monkeypatch.setenv("DEEPGRAM_FLUX_HOST", "default-flux:8080")
    monkeypatch.setenv("DEEPGRAM_HOST_PROTOCOL", "wss")
    monkeypatch.setenv("DEEPGRAM_ENDPOINT_GCP_HOST", "gcp:8080")
    monkeypatch.setenv("DEEPGRAM_ENDPOINT_GCP_FLUX_HOST", "gcp-flux:8080")
    monkeypatch.setenv("DEEPGRAM_ENDPOINT_GCP_PROTOCOL", "ws")
    monkeypatch.setenv("DEEPGRAM_ENDPOINT_AWS_HOST", "aws:8080")


def test_default_host_without_an_endpoint():
    assert _nova() == ("wss", "default:8080")
    assert _flux() == ("wss", "default-flux:8080")


def test_api_deepgram_com_when_nothing_is_configured(monkeypatch):
    for name in ("DEEPGRAM_HOST", "DEEPGRAM_FLUX_HOST", "DEEPGRAM_HOST_PROTOCOL"):
        monkeypatch.delenv(name)
    assert _nova() == ("wss", "api.deepgram.com")
    assert _flux() == ("wss", "api.deepgram.com")


def test_named_endpoint_resolves_from_env():
    assert _nova(deepgram_endpoint="gcp") == ("ws", "gcp:8080")
    assert _flux(deepgram_endpoint="gcp") == ("ws", "gcp-flux:8080")


def test_endpoint_without_flux_host_or_protocol_uses_its_host_and_wss():
    assert _nova(deepgram_endpoint="aws") == ("wss", "aws:8080")
    assert _flux(deepgram_endpoint="aws") == ("wss", "aws:8080")


@pytest.mark.parametrize("name", ["unknown", "", None, 7, ["gcp"]])
def test_an_unconfigured_or_malformed_name_uses_the_default(name):
    assert _nova(deepgram_endpoint=name) == ("wss", "default:8080")


def test_a_customer_key_keeps_the_default_host():
    assert _nova(deepgram_endpoint="gcp", transcriber_key="customer-key") == ("wss", "default:8080")


def test_a_raw_host_kwarg_is_not_read():
    assert _nova(deepgram_host="elsewhere:443", deepgram_host_protocol="ws") == ("wss", "default:8080")
    assert _flux(deepgram_flux_host="elsewhere:443") == ("wss", "default-flux:8080")
