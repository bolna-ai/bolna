"""Unit tests for Deepgram Flux URL building, language hints, and tuning defaults."""

import os
from urllib.parse import urlparse, parse_qs

import pytest

from bolna.transcriber.deepgram_transcriber import DeepgramTranscriber


def _make_transcriber(model="flux-general-en", language="en", **kwargs):
    return DeepgramTranscriber(
        telephony_provider="plivo",
        model=model,
        language=language,
        stream=True,
        **kwargs,
    )


def _params(url):
    return parse_qs(urlparse(url).query, keep_blank_values=True)


def test_flux_url_uses_v2_endpoint():
    t = _make_transcriber()
    url = t._get_flux_ws_url()
    assert "/v2/listen?" in url


def test_flux_general_en_omits_language_hint():
    t = _make_transcriber(model="flux-general-en", language="en")
    p = _params(t._get_flux_ws_url())
    assert "language_hint" not in p


def test_flux_general_multi_with_single_language():
    t = _make_transcriber(model="flux-general-multi", language="hi")
    p = _params(t._get_flux_ws_url())
    assert p["language_hint"] == ["hi"]


def test_flux_general_multi_with_multi_prefix():
    t = _make_transcriber(model="flux-general-multi", language="multi-hi")
    p = _params(t._get_flux_ws_url())
    assert p["language_hint"] == ["hi"]


def test_flux_general_multi_with_bare_multi_omits_hint():
    t = _make_transcriber(model="flux-general-multi", language="multi")
    p = _params(t._get_flux_ws_url())
    assert "language_hint" not in p


def test_flux_general_multi_with_explicit_hints_list():
    t = _make_transcriber(
        model="flux-general-multi",
        language="multi",
        language_hints=["en", "hi"],
    )
    p = _params(t._get_flux_ws_url())
    assert p["language_hint"] == ["en", "hi"]


def test_flux_defaults_present_in_url():
    t = _make_transcriber()
    p = _params(t._get_flux_ws_url())
    assert p["eot_threshold"] == ["0.7"]
    assert p["eot_timeout_ms"] == ["500"]
    # eager_eot_threshold is only emitted when explicitly configured (no default sent).
    assert "eager_eot_threshold" not in p


def test_flux_kwargs_override_defaults():
    t = _make_transcriber(
        eot_threshold=0.8,
        eager_eot_threshold=0.4,
        eot_timeout_ms=2000,
    )
    p = _params(t._get_flux_ws_url())
    assert p["eot_threshold"] == ["0.8"]
    assert p["eager_eot_threshold"] == ["0.4"]
    assert p["eot_timeout_ms"] == ["2000"]


def test_resolve_language_hints_helper():
    cases = [
        ("multi-general", "multi", None),
        ("multi-general", "multi-hi", ["hi"]),
        ("multi-general", "hi", ["hi"]),
        ("multi-general", None, None),
    ]
    for _, lang, expected in cases:
        t = _make_transcriber(model="flux-general-multi", language=lang or "en")
        if lang is None:
            t.language = None
        assert t._resolve_language_hints() == expected, f"for language={lang}"

    # explicit hints override derivation
    t = _make_transcriber(model="flux-general-multi", language="hi", language_hints=["en", "es"])
    assert t._resolve_language_hints() == ["en", "es"]


def test_nova_url_unchanged_by_flux_changes():
    t = _make_transcriber(model="nova-2", language="en")
    url = t._get_nova_ws_url()
    assert "/v1/listen?" in url
    p = _params(url)
    # Flux params must not leak into Nova URL
    assert "eot_threshold" not in p
    assert "eager_eot_threshold" not in p
    assert "language_hint" not in p


def test_is_flux_model_detection():
    assert _make_transcriber(model="flux-general-en").is_flux_model is True
    assert _make_transcriber(model="flux-general-multi").is_flux_model is True
    assert _make_transcriber(model="flux-general-multi").is_flux_multi is True
    assert _make_transcriber(model="flux-general-en").is_flux_multi is False
    assert _make_transcriber(model="nova-2").is_flux_model is False
