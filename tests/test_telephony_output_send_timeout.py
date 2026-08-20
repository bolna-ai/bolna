"""A telephony provider's media WebSocket can go half-dead: the underlying TCP
connection stops delivering ACKs without ever sending a close frame, so
`websocket.send_text()` never raises and never returns.

`__cleanup_downstream_tasks()` awaits `output.handle_interruption()` first,
ahead of task cancellation and history sync, with nothing above it bounding
the wait. Production incident: execution 88eeeb50-dadb-4abc-8f79-fd34cb6bac61
(Plivo, 2026-08-17) froze there for ~10.5 minutes with a lost transcript
because Plivo's own idle-stream detection was the only thing that ever ended
the call.

These tests simulate a send that never resolves and assert the call site
still finishes within bounds instead of hanging indefinitely.
"""

import asyncio

import pytest

import bolna.output_handlers.telephony as telephony_module
from bolna.helpers.mark_event_meta_data import MarkEventMetaData
from bolna.output_handlers.telephony_providers.exotel import ExotelOutputHandler
from bolna.output_handlers.telephony_providers.plivo import PlivoOutputHandler
from bolna.output_handlers.telephony_providers.twilio import TwilioOutputHandler
from bolna.output_handlers.telephony_providers.vobiz import VobizOutputHandler


class _HangingWebSocket:
    """A socket that never completes a send and never raises — the half-dead case."""

    async def send_text(self, payload):
        await asyncio.Event().wait()  # never set: hangs forever, never raises


class _RecordingWebSocket:
    def __init__(self):
        self.sent = []

    async def send_text(self, payload):
        self.sent.append(payload)


@pytest.fixture(autouse=True)
def _fast_send_timeout(monkeypatch):
    # Pre-fix, this constant doesn't exist yet and the outer wait_for below is
    # what catches the hang. Post-fix, shrink the real timeout so the test
    # runs fast and deterministically instead of riding the outer bound.
    if hasattr(telephony_module, "OUTPUT_SEND_TIMEOUT_S"):
        monkeypatch.setattr(telephony_module, "OUTPUT_SEND_TIMEOUT_S", 0.05)


@pytest.mark.parametrize(
    "handler_cls",
    [PlivoOutputHandler, TwilioOutputHandler, ExotelOutputHandler, VobizOutputHandler],
)
async def test_handle_interruption_does_not_hang_on_a_dead_socket(handler_cls):
    handler = handler_cls(websocket=_HangingWebSocket(), mark_event_meta_data=MarkEventMetaData())
    handler.stream_sid = "test-stream"

    # If handle_interruption's send has no timeout, this outer wait_for is what
    # actually catches the hang in CI; a real dead call would just never return.
    await asyncio.wait_for(handler.handle_interruption(), timeout=1.0)

    assert handler.is_closed() is True


async def test_handle_interruption_still_sends_normally_on_a_healthy_socket():
    ws = _RecordingWebSocket()
    handler = PlivoOutputHandler(websocket=ws, mark_event_meta_data=MarkEventMetaData())
    handler.stream_sid = "test-stream"

    await asyncio.wait_for(handler.handle_interruption(), timeout=1.0)

    assert handler.is_closed() is False
    assert len(ws.sent) == 1


def _audio_packet():
    return {
        "data": b"\x01\x02" * 200,
        "meta_info": {
            "stream_sid": "test-stream",
            "sequence_id": 1,
            "turn_id": "t1",
            "response_uid": "r1",
            "response_group_uid": "g1",
            "cached": False,
            "format": "wav",
        },
    }


async def test_handle_does_not_hang_sending_audio_on_a_dead_socket():
    handler = PlivoOutputHandler(websocket=_HangingWebSocket(), mark_event_meta_data=MarkEventMetaData())

    await asyncio.wait_for(handler.handle(_audio_packet()), timeout=1.0)

    assert handler.is_closed() is True


async def test_handle_still_sends_normally_on_a_healthy_socket():
    ws = _RecordingWebSocket()
    handler = PlivoOutputHandler(websocket=ws, mark_event_meta_data=MarkEventMetaData())

    await asyncio.wait_for(handler.handle(_audio_packet()), timeout=1.0)

    assert handler.is_closed() is False
    assert len(ws.sent) == 3  # pre-mark, media, post-mark
