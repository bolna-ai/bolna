"""SonioxLID: tokens fold into ONE segment per utterance (flushed on <end>) so
segment cadence — and the short-audio substance gate driven by segment duration —
matches the Sarvam backend. Language = dominant final-token tag; Soniox reports no
language probability so prob is None."""

from bolna.constants import SONIOX_ENDPOINT_TOKEN
from bolna.lid.provider import LIDProvider
from bolna.lid.sarvam import SarvamLID
from bolna.lid.soniox import SonioxLID


def _detector():
    # on_language=None, empty config → no network; we only exercise message handling.
    return SonioxLID(on_language=None, config={})


def _tok(text, lang=None, is_final=True, start_ms=None, end_ms=None):
    t = {"text": text, "is_final": is_final}
    if lang:
        t["language"] = lang
    if start_ms is not None:
        t["start_ms"] = start_ms
    if end_ms is not None:
        t["end_ms"] = end_ms
    return t


def test_finals_accumulate_and_flush_on_endpoint():
    d = _detector()
    d._handle_message({"tokens": [_tok("Travel", "en", start_ms=0, end_ms=500), _tok(" again", "en", end_ms=1100)]})
    assert d.take_turn_transcript() == ("", None)  # nothing buffered until <end>
    d._handle_message({"tokens": [_tok(SONIOX_ENDPOINT_TOKEN)]})
    text, lang = d.take_turn_transcript()
    assert text == "Travel again"
    assert lang == "en"


def test_segment_duration_spans_utterance_not_tokens():
    # The substance gate needs utterance-level audio_s: 0→1800ms = 1.8s, not per-token slivers.
    d = _detector()
    d._handle_message({"tokens": [_tok("ఇంకేం", "te", start_ms=0, end_ms=700)]})
    d._handle_message({"tokens": [_tok(" సంగతులు", "te", end_ms=1800), _tok(SONIOX_ENDPOINT_TOKEN)]})
    assert d.buffer_max_segment_seconds() == 1.8
    segs = d.buffer_segments()
    assert len(segs) == 1
    assert segs[0]["lang"] == "te"
    assert segs[0]["prob"] is None  # soniox has no language probability
    assert segs[0]["ts"] is not None


def test_dominant_language_wins_segment_tag():
    d = _detector()
    d._handle_message(
        {
            "tokens": [
                _tok("mera", "hi", start_ms=0, end_ms=300),
                _tok(" number", "en", end_ms=600),
                _tok(" nau", "hi", end_ms=900),
                _tok(" aath", "hi", end_ms=1200),
                _tok(SONIOX_ENDPOINT_TOKEN),
            ]
        }
    )
    text, lang = d.take_turn_transcript()
    assert lang == "hi"  # 3 hi vs 1 en


def test_non_final_tokens_ignored():
    d = _detector()
    d._handle_message({"tokens": [_tok("hell", "en", is_final=False), _tok("Hello", "en", end_ms=500)]})
    d._handle_message({"tokens": [_tok(SONIOX_ENDPOINT_TOKEN)]})
    text, _ = d.take_turn_transcript()
    assert text == "Hello"


def test_multiple_utterances_become_multiple_segments():
    d = _detector()
    d._handle_message({"tokens": [_tok("హలో", "te", start_ms=0, end_ms=600), _tok(SONIOX_ENDPOINT_TOKEN)]})
    d._handle_message({"tokens": [_tok("Why English", "en", start_ms=2000, end_ms=3600), _tok(SONIOX_ENDPOINT_TOKEN)]})
    segs = d.buffer_segments()
    assert [s["lang"] for s in segs] == ["te", "en"]
    assert d.buffer_language() == "en"  # latest
    assert d.buffer_language_confidence() is None


def test_region_tagged_language_normalized():
    d = _detector()
    d._handle_message({"tokens": [_tok("नमस्ते", "hi-IN", end_ms=500), _tok(SONIOX_ENDPOINT_TOKEN)]})
    assert d.take_turn_transcript()[1] == "hi"


def test_stop_flush_preserves_pending_utterance():
    # Finals without a trailing <end> at teardown must still reach the buffer —
    # exercised via the flush helper directly (stop() also closes network state).
    d = _detector()
    d._handle_message({"tokens": [_tok("bye", "en", start_ms=0, end_ms=400)]})
    d._flush_pending_segment()
    assert d.take_turn_transcript() == ("bye", "en")


def test_server_error_raises_connection_error():
    d = _detector()
    try:
        d._handle_message({"error_code": 401, "error_message": "bad key"})
        raise AssertionError("expected ConnectionError")
    except ConnectionError as e:
        assert "401" in str(e)


def test_factory_creates_soniox_and_falls_back_to_sarvam():
    assert isinstance(LIDProvider.create("soniox", None, {}), SonioxLID)
    assert isinstance(LIDProvider.create("sarvam", None, {}), SarvamLID)
    # Removed/unknown providers (azure, elevenlabs_scribe) fall back to sarvam.
    assert isinstance(LIDProvider.create("azure", None, {}), SarvamLID)
    assert isinstance(LIDProvider.create("elevenlabs_scribe", None, {}), SarvamLID)


def test_soniox_inherits_turn_buffer_api():
    # The pool's new-flow inertness check requires the per-turn buffer API.
    d = _detector()
    for method in (
        "take_turn_transcript",
        "buffer_age_seconds",
        "buffer_language",
        "buffer_event",
        "buffer_language_streak",
        "buffer_language_confidence",
        "buffer_segments",
        "buffer_max_segment_seconds",
    ):
        assert callable(getattr(d, method))


def test_telephony_audio_params():
    plivo = SonioxLID(None, {"telephony_provider": "plivo"})
    assert (plivo._audio_format, plivo._input_sr) == ("pcm_s16le", 8000)
    twilio = SonioxLID(None, {"telephony_provider": "twilio"})
    assert (twilio._audio_format, twilio._input_sr) == ("mulaw", 8000)
    web = SonioxLID(None, {})
    assert (web._audio_format, web._input_sr) == ("pcm_s16le", 16000)
