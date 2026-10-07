"""The telephony input forwards caller audio in its 10-message batches unless the transcriber asks for a chunk."""

import asyncio
import base64
import json

import pytest

from bolna.input_handlers.telephony_providers.plivo import PlivoInputHandler
from bolna.input_handlers.telephony_providers.twilio import TwilioInputHandler


class _FakeWebSocket:
    def __init__(self, frames):
        self.messages = [
            json.dumps(
                {
                    "event": "media",
                    "media": {"payload": base64.b64encode(f).decode(), "timestamp": str(20 * i), "track": "inbound"},
                }
            )
            for i, f in enumerate(frames)
        ] + [json.dumps({"event": "stop"})]

    async def receive_text(self):
        return self.messages.pop(0)


async def _forwarded(handler_class, frames, audio_chunk_ms=None):
    queues = {"transcriber": asyncio.Queue()}
    handler = handler_class(queues, websocket=_FakeWebSocket(frames), input_types={"audio": 0})
    handler.audio_chunk_ms = audio_chunk_ms
    await handler._listen()
    sizes = []
    while not queues["transcriber"].empty():
        packet = queues["transcriber"].get_nowait()
        if packet["data"]:
            sizes.append(len(packet["data"]))
    return sizes


@pytest.mark.asyncio
async def test_without_a_transcriber_chunk_ten_provider_messages_make_one_packet():
    frames = [b"\x01" * 320] * 25
    assert await _forwarded(PlivoInputHandler, frames) == [3200, 3200]


@pytest.mark.asyncio
async def test_a_40_ms_chunk_forwards_40_ms_of_linear16_audio_per_packet():
    frames = [b"\x01" * 320] * 10
    assert await _forwarded(PlivoInputHandler, frames, audio_chunk_ms=40) == [640] * 5


@pytest.mark.asyncio
async def test_a_40_ms_chunk_counts_mulaw_at_one_byte_per_sample():
    frames = [b"\x01" * 160] * 10
    assert await _forwarded(TwilioInputHandler, frames, audio_chunk_ms=40) == [320] * 5
