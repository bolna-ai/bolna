"""Deepgram Model Improvement Program opt-out reaches every request URL."""

from urllib.parse import parse_qs, urlparse

import pytest

from bolna.models import DeepgramConfig, Transcriber
from bolna.synthesizer.deepgram_synthesizer import DeepgramSynthesizer
from bolna.transcriber.deepgram_transcriber import DeepgramTranscriber


def _transcriber(model="nova-3", stream=True, **kwargs):
    return DeepgramTranscriber(telephony_provider="plivo", model=model, language="en", stream=stream, **kwargs)


def _synth(**kwargs):
    return DeepgramSynthesizer(voice_id="asteria", voice="Asteria", model="aura-2", **kwargs)


def _query(url):
    return parse_qs(urlparse(url).query)


@pytest.fixture(autouse=True)
def _no_env(monkeypatch):
    monkeypatch.delenv("DEEPGRAM_MIP_OPT_OUT", raising=False)


@pytest.mark.parametrize("model", ["nova-3", "flux-general-en"])
def test_streaming_url_has_opt_out(model):
    assert _query(_transcriber(model=model, mip_opt_out=True).get_deepgram_ws_url())["mip_opt_out"] == ["true"]


@pytest.mark.parametrize("model", ["nova-3", "flux-general-en"])
def test_streaming_url_omits_param_by_default(model):
    assert "mip_opt_out" not in _query(_transcriber(model=model).get_deepgram_ws_url())


async def test_prerecorded_url_has_opt_out():
    t = _transcriber(stream=False, mip_opt_out=True)
    try:
        assert _query(t.api_url)["mip_opt_out"] == ["true"]
    finally:
        await t.session.close()


def test_env_default_applies_when_unset(monkeypatch):
    monkeypatch.setenv("DEEPGRAM_MIP_OPT_OUT", "true")
    assert "mip_opt_out" in _query(_transcriber().get_deepgram_ws_url())


def test_explicit_false_overrides_env(monkeypatch):
    monkeypatch.setenv("DEEPGRAM_MIP_OPT_OUT", "true")
    assert "mip_opt_out" not in _query(_transcriber(mip_opt_out=False).get_deepgram_ws_url())


def test_synthesizer_urls_have_opt_out():
    assert _query(_synth(mip_opt_out=True).ws_url)["mip_opt_out"] == ["true"]
    assert "mip_opt_out" not in _query(_synth().ws_url)


def test_config_models_accept_field():
    assert Transcriber(provider="deepgram", mip_opt_out=True).mip_opt_out is True
    assert DeepgramConfig(voice_id="a", voice="A", model="aura-2", mip_opt_out=True).mip_opt_out is True
    assert Transcriber(provider="deepgram").mip_opt_out is None
