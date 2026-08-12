"""utterance_end_ms drives the UtteranceEnd fallback and is measured on WORD gaps, not
silence — with endpointing=200 it used to be pinned to 1000ms with no way to lengthen it
except by wrecking the speech_final fast path. Now configurable per agent; unconfigured
agents must keep the old derivation byte-for-byte."""

from urllib.parse import urlparse, parse_qs

from bolna.transcriber.deepgram_transcriber import DeepgramTranscriber
from bolna.models import Transcriber


def _make_transcriber(**kwargs):
    return DeepgramTranscriber(
        telephony_provider="plivo",
        model="nova-3",
        language="hi",
        stream=True,
        **kwargs,
    )


def _params(url):
    return parse_qs(urlparse(url).query, keep_blank_values=True)


def test_default_unchanged_low_endpointing():
    # endpointing=200 → floored to 1000, exactly the old `1000 if ep < 1000 else ep`
    assert _make_transcriber(endpointing="200").utterance_end_ms == 1000


def test_default_unchanged_high_endpointing():
    assert _make_transcriber(endpointing="1500").utterance_end_ms == 1500


def test_override_wins_over_endpointing():
    t = _make_transcriber(endpointing="200", utterance_end_ms=1500)
    assert t.utterance_end_ms == 1500
    assert t.endpointing_ms == 200  # fast path untouched


def test_override_clamped_to_deepgram_minimum():
    assert _make_transcriber(endpointing="200", utterance_end_ms=400).utterance_end_ms == 1000


def test_none_override_falls_back():
    # transcriber_config splat passes utterance_end_ms=None for unconfigured agents
    assert _make_transcriber(endpointing="200", utterance_end_ms=None).utterance_end_ms == 1000


def test_string_override_coerced():
    # agent configs arrive as JSON; values may be strings like endpointing itself
    assert _make_transcriber(endpointing="200", utterance_end_ms="2000").utterance_end_ms == 2000


def test_value_reaches_ws_url():
    t = _make_transcriber(endpointing="200", utterance_end_ms=1500)
    params = _params(t._get_nova_ws_url())
    assert params["utterance_end_ms"] == ["1500"]
    assert params["endpointing"] == ["200"]


def test_transcriber_model_accepts_field():
    cfg = Transcriber(provider="deepgram", endpointing=200, utterance_end_ms=1500)
    assert cfg.utterance_end_ms == 1500
    assert Transcriber(provider="deepgram").utterance_end_ms is None
