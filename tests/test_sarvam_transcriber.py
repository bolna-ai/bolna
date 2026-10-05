"""Unit tests for SarvamTranscriber endpoint routing.

saaras models split across two Sarvam endpoints: v3+ transcribe directly on
/speech-to-text (and need language-code), older saaras goes to the legacy
/speech-to-text-translate. Getting this wrong is silent — the connection
succeeds against the wrong endpoint — so the URLs are pinned here.
"""

import json
from urllib.parse import parse_qs, urlparse

import aiohttp
import pytest

from bolna.constants import SARVAM_MAX_KEYTERM_CHARACTERS, SARVAM_MAX_KEYTERMS
from bolna.transcriber.sarvam_transcriber import SarvamTranscriber


def _transcriber(model, language="hi-IN", **kwargs):
    # telephony_provider="twilio" exercises the 8k->16k telephony branch; no
    # network happens until run(), and stream=False would open an aiohttp session.
    return SarvamTranscriber(
        telephony_provider="twilio",
        model=model,
        language=language,
        stream=True,
        transcriber_key="test-key",
        **kwargs,
    )


def _ws_keyterms(transcriber):
    values = parse_qs(urlparse(transcriber.ws_url).query).get("keyterms")
    return json.loads(values[0]) if values else None


@pytest.mark.parametrize("model", ["saaras:v3", "saaras:v4", "saarika:v2.5"])
def test_transcribe_models_use_speech_to_text(model):
    t = _transcriber(model)
    assert t.api_url == "https://api.sarvam.ai/speech-to-text"
    assert t.ws_url.startswith("wss://api.sarvam.ai/speech-to-text/ws?")
    assert "/speech-to-text-translate" not in t.ws_url


@pytest.mark.parametrize("model", ["saaras:v3", "saaras:v4", "saarika:v2.5"])
def test_transcribe_models_send_language_code(model):
    # The translate branch omits language-code entirely; a model landing there by
    # mistake transcribes with no language at all.
    assert "language-code=hi-IN" in _transcriber(model).ws_url


def test_legacy_saaras_still_uses_translate_endpoint():
    t = _transcriber("saaras:v2.5")
    assert t.api_url == "https://api.sarvam.ai/speech-to-text-translate"
    assert t.ws_url.startswith("wss://api.sarvam.ai/speech-to-text-translate/ws?")
    assert "language-code" not in t.ws_url


def test_mode_is_sent_for_v3_only():
    # Sarvam documents `mode` as a saaras:v3-only parameter; v4 defaults to
    # transcribe on the WS, so sending it there risks a rejection for no gain.
    assert "mode=transcribe" in _transcriber("saaras:v3").ws_url
    assert "mode=" not in _transcriber("saaras:v4").ws_url


@pytest.mark.parametrize("model", ["saaras:v3", "saaras:v4"])
def test_vad_params_present(model):
    # Turn tracking (turn_counter, turn_latencies, speech_started) is driven
    # entirely by the VAD "events" messages these two params enable.
    ws_url = _transcriber(model).ws_url
    assert "high_vad_sensitivity=true" in ws_url
    assert "vad_signals=true" in ws_url


def test_unknown_language_passthrough():
    # Auto-detect mode: language-code=unknown is what makes saaras return
    # language_code per segment.
    assert "language-code=unknown" in _transcriber("saaras:v4", language="unknown").ws_url


def test_v4_sends_keyterms_as_a_json_array_without_weights():
    # Multilingual legs inherit the base transcriber's keywords, Deepgram weights included.
    t = _transcriber("saaras:v4", keywords="Bolna:5, New Delhi, 3:30 pm")
    assert _ws_keyterms(t) == ["Bolna", "New Delhi", "3:30 pm"]


@pytest.mark.parametrize("model", ["saaras:v3", "saarika:v2.5", "saaras:v2.5"])
def test_keyterms_are_sent_to_v4_only(model):
    assert "keyterms" not in _transcriber(model, keywords="Bolna").ws_url


def test_keyterms_are_held_to_sarvams_limits():
    too_long = "x" * (SARVAM_MAX_KEYTERM_CHARACTERS + 1)
    keywords = ",".join([too_long, "Bolna", "Bolna"] + [f"term{i}" for i in range(SARVAM_MAX_KEYTERMS)])
    keyterms = _ws_keyterms(_transcriber("saaras:v4", keywords=keywords))
    assert len(keyterms) == SARVAM_MAX_KEYTERMS
    assert keyterms[:2] == ["Bolna", "term0"]
    assert too_long not in keyterms


@pytest.mark.parametrize("keywords", [None, "", " , "])
def test_no_keyterms_without_keywords(keywords):
    assert "keyterms" not in _transcriber("saaras:v4", keywords=keywords).ws_url


class _RecordingForm:
    def __init__(self):
        self.fields = {}

    def add_field(self, name, value, **kwargs):
        self.fields[name] = value


class _FakeResponse:
    status = 200

    async def text(self):
        return json.dumps({"transcript": ""})

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeSession:
    closed = False

    def post(self, url, **kwargs):
        return _FakeResponse()


async def test_rest_sends_keyterms_as_one_json_form_field(monkeypatch):
    form = _RecordingForm()
    monkeypatch.setattr(aiohttp, "FormData", lambda: form)
    t = _transcriber("saaras:v4", keywords="Bolna:5, New Delhi")
    t.session = _FakeSession()

    await t._get_http_transcription(b"\xff" * 320)

    assert json.loads(form.fields["keyterms"]) == ["Bolna", "New Delhi"]
