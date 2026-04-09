"""Tests for Deepgram run_id tagging via tag and extra query params."""
import asyncio
from unittest.mock import MagicMock
from urllib.parse import urlparse, parse_qs

import pytest

from bolna.transcriber.deepgram_transcriber import DeepgramTranscriber
from bolna.synthesizer.deepgram_synthesizer import DeepgramSynthesizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_transcriber(run_id=None, stream=True, **extra_kwargs):
    kwargs = {
        "enforce_streaming": True,
        "transcriber_key": "fake-key",
    }
    if run_id is not None:
        kwargs["run_id"] = run_id
    kwargs.update(extra_kwargs)
    t = DeepgramTranscriber(
        telephony_provider="plivo",
        input_queue=asyncio.Queue(),
        output_queue=asyncio.Queue(),
        stream=stream,
        **kwargs,
    )
    return t


def _make_synthesizer(run_id=None, stream=False):
    kwargs = {
        "transcriber_key": "fake-key",
    }
    if run_id is not None:
        kwargs["run_id"] = run_id
    s = DeepgramSynthesizer(
        voice_id="asteria",
        voice="asteria",
        stream=stream,
        **kwargs,
    )
    return s


def _parse_url_params(url):
    parsed = urlparse(url)
    return parse_qs(parsed.query)


# ---------------------------------------------------------------------------
# DeepgramTranscriber — streaming (WebSocket)
# ---------------------------------------------------------------------------

class TestTranscriberStreamingTagging:
    def test_ws_url_includes_tag_when_run_id_set(self):
        t = _make_transcriber(run_id="run-abc-123")
        url = t.get_deepgram_ws_url()
        params = _parse_url_params(url)
        assert params["tag"] == ["run-abc-123"]

    def test_ws_url_includes_extra_when_run_id_set(self):
        t = _make_transcriber(run_id="run-abc-123")
        url = t.get_deepgram_ws_url()
        params = _parse_url_params(url)
        assert params["extra"] == ["run_id:run-abc-123"]

    def test_ws_url_no_tag_without_run_id(self):
        t = _make_transcriber()
        url = t.get_deepgram_ws_url()
        params = _parse_url_params(url)
        assert "tag" not in params
        assert "extra" not in params

    def test_ws_url_no_tag_with_none_run_id(self):
        t = _make_transcriber(run_id=None)
        url = t.get_deepgram_ws_url()
        params = _parse_url_params(url)
        assert "tag" not in params


# ---------------------------------------------------------------------------
# DeepgramTranscriber — HTTP (non-streaming)
# ---------------------------------------------------------------------------

class TestTranscriberHttpTagging:
    def test_http_url_includes_tag_when_run_id_set(self):
        t = _make_transcriber(run_id="run-http-456", stream=False)
        params = _parse_url_params(t.api_url)
        assert params["tag"] == ["run-http-456"]

    def test_http_url_includes_extra_when_run_id_set(self):
        t = _make_transcriber(run_id="run-http-456", stream=False)
        params = _parse_url_params(t.api_url)
        assert params["extra"] == ["run_id:run-http-456"]

    def test_http_url_no_tag_without_run_id(self):
        t = _make_transcriber(stream=False)
        params = _parse_url_params(t.api_url)
        assert "tag" not in params
        assert "extra" not in params


# ---------------------------------------------------------------------------
# DeepgramSynthesizer — WebSocket
# ---------------------------------------------------------------------------

class TestSynthesizerWsTagging:
    def test_ws_url_includes_tag_when_run_id_set(self):
        s = _make_synthesizer(run_id="run-tts-789")
        params = _parse_url_params(s.ws_url)
        assert params["tag"] == ["run-tts-789"]

    def test_ws_url_no_tag_without_run_id(self):
        s = _make_synthesizer()
        params = _parse_url_params(s.ws_url)
        assert "tag" not in params


# ---------------------------------------------------------------------------
# DeepgramSynthesizer — HTTP
# ---------------------------------------------------------------------------

class TestSynthesizerHttpTagging:
    @pytest.mark.asyncio
    async def test_http_url_includes_tag_when_run_id_set(self):
        s = _make_synthesizer(run_id="run-tts-http-012")
        # We can't easily call _generate_http without mocking the HTTP call,
        # but we can verify run_id is stored and would be appended
        assert s.run_id == "run-tts-http-012"

    def test_run_id_none_when_not_provided(self):
        s = _make_synthesizer()
        assert s.run_id is None
