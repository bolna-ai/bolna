"""Unit tests for SarvamTranscriber endpoint routing.

saaras models split across two Sarvam endpoints: v3+ transcribe directly on
/speech-to-text (and need language-code), older saaras goes to the legacy
/speech-to-text-translate. Getting this wrong is silent — the connection
succeeds against the wrong endpoint — so the URLs are pinned here.
"""

import pytest

from bolna.transcriber.sarvam_transcriber import SarvamTranscriber


def _transcriber(model, language="hi-IN"):
    # telephony_provider="twilio" exercises the 8k->16k telephony branch; no
    # network happens until run(), and stream=False would open an aiohttp session.
    return SarvamTranscriber(
        telephony_provider="twilio",
        model=model,
        language=language,
        stream=True,
        transcriber_key="test-key",
    )


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
