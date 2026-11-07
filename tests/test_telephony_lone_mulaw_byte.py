"""Streaming TTS (Maya) emits frames of a few samples that resample to one mu-law byte. The output
handler pads a lone byte to two; padding with 0x00 played a full-scale negative sample, heard as a
click every ~85 ms. The lone sample is repeated instead; the end sentinel still sends nothing."""

import audioop
import base64
import json

import pytest

from bolna.helpers.mark_event_meta_data import MarkEventMetaData
from bolna.output_handlers.telephony_providers.plivo import PlivoOutputHandler
from bolna.output_handlers.telephony_providers.twilio import TwilioOutputHandler


class RecordingWebSocket:
    def __init__(self):
        self.sent = []

    async def send_text(self, payload):
        self.sent.append(json.loads(payload))


def media_payloads(ws):
    out = []
    for msg in ws.sent:
        media = msg.get("media") or {}
        if "payload" in media:
            out.append(base64.b64decode(media["payload"]))
    return out


async def send(handler_cls, chunk, audio_format="mulaw"):
    ws = RecordingWebSocket()
    handler = handler_cls(websocket=ws, mark_event_meta_data=MarkEventMetaData())
    handler.stream_sid = "test-stream"
    meta_info = {"format": audio_format, "sequence_id": 1, "stream_sid": "test-stream"}
    await handler.handle({"data": chunk, "meta_info": meta_info})
    return media_payloads(ws)


@pytest.mark.parametrize("handler_cls", [PlivoOutputHandler, TwilioOutputHandler])
async def test_a_lone_mulaw_byte_is_repeated_not_padded_with_a_full_scale_sample(handler_cls):
    sample = bytes([0x9C])  # an ordinary mid-level mu-law sample
    (payload,) = await send(handler_cls, sample)

    assert payload == sample + sample
    first, pad = (int.from_bytes(audioop.ulaw2lin(payload, 2)[i : i + 2], "little", signed=True) for i in (0, 2))
    assert pad == first  # the old 0x00 pad decoded to -32124


async def test_a_lone_pcm_byte_keeps_the_zero_high_byte():
    # Half a 16-bit sample: repeating it could be loud, a zero high byte keeps it near silence.
    # Plivo forwards PCM as is (Twilio re-encodes it, so its wire bytes differ by design).
    assert await send(PlivoOutputHandler, bytes([0x9C]), audio_format="pcm") == [bytes([0x9C, 0x00])]


@pytest.mark.parametrize("handler_cls", [PlivoOutputHandler, TwilioOutputHandler])
async def test_the_end_sentinel_still_sends_no_audio(handler_cls):
    assert await send(handler_cls, b"\x00") == []


@pytest.mark.parametrize("handler_cls", [PlivoOutputHandler, TwilioOutputHandler])
async def test_ordinary_chunks_are_sent_unchanged(handler_cls):
    chunk = bytes([0x9C, 0xA0, 0xFF, 0x80])
    assert await send(handler_cls, chunk) == [chunk]
