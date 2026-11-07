"""Unit tests for DeepgramTranscriber.deepgram_connect handshake-rejection messages."""

import pytest
from websockets.datastructures import Headers
from websockets.exceptions import InvalidStatus
from websockets.http11 import Response

from bolna.transcriber import deepgram_transcriber
from bolna.transcriber.deepgram_transcriber import DeepgramTranscriber

KEYTERM_LIMIT = "Bad Request: Keyterm limit exceeded. The maximum number of tokens across all keyterms is 500."


def _make():
    return DeepgramTranscriber(telephony_provider="plivo", model="nova-3", language="hi", stream=True)


def _reject_with(monkeypatch, headers=None, body=b""):
    async def fake_connect(*args, **kwargs):
        raise InvalidStatus(Response(400, "Bad Request", Headers(headers or {}), body))

    monkeypatch.setattr(deepgram_transcriber.websockets, "connect", fake_connect)


async def test_rejection_message_includes_dg_error(monkeypatch):
    _reject_with(monkeypatch, headers={"dg-error": KEYTERM_LIMIT}, body=b'{"err_msg": "ignored"}')
    with pytest.raises(ConnectionError) as exc:
        await _make().deepgram_connect()
    assert str(exc.value) == (
        f"Invalid handshake during Deepgram websocket connection: "
        f"server rejected WebSocket connection: HTTP 400 ({KEYTERM_LIMIT})"
    )


async def test_rejection_message_falls_back_to_body(monkeypatch):
    _reject_with(monkeypatch, body=b'{"err_code":"Bad Request","err_msg":"bad"}\n')
    with pytest.raises(ConnectionError) as exc:
        await _make().deepgram_connect()
    assert str(exc.value).endswith('HTTP 400 ({"err_code":"Bad Request","err_msg":"bad"})')


async def test_rejection_message_without_reason_is_unchanged(monkeypatch):
    _reject_with(monkeypatch)
    with pytest.raises(ConnectionError) as exc:
        await _make().deepgram_connect()
    assert str(exc.value) == (
        "Invalid handshake during Deepgram websocket connection: server rejected WebSocket connection: HTTP 400"
    )
