"""DefaultOutputHandler.handle registers a mark duration so web calls feed the same playout
estimate as telephony.

The duration call sits inside handle's try, whose except sets _closed and stops all further
audio, so a throw there would silently kill the call. These assert it does not throw and that
the wire format the browser sees is unchanged.
"""

import asyncio
import json

from bolna.helpers.mark_event_meta_data import MarkEventMetaData
from bolna.output_handlers.default import DefaultOutputHandler


class _FakeWebSocket:
    def __init__(self):
        self.text = []
        self.json = []

    async def send_text(self, payload):
        self.text.append(payload)

    async def send_json(self, payload):
        self.json.append(payload)


def _packet(audio, fmt="pcm"):
    return {
        "data": audio,
        "meta_info": {"type": "audio", "format": fmt, "sequence_id": 4, "message_category": "agent_response"},
    }


def _run(sampling_rate=24000, audio=b"\x01\x02" * 24000, fmt="pcm"):
    mark = MarkEventMetaData()
    ws = _FakeWebSocket()
    handler = DefaultOutputHandler(websocket=ws, mark_event_meta_data=mark, sampling_rate=sampling_rate)
    asyncio.run(handler.handle(_packet(audio, fmt)))
    return handler, mark, ws


def test_audio_send_does_not_close_the_handler():
    handler, _, ws = _run()
    assert handler.is_closed() is False
    assert len(ws.json) == 1  # the audio frame still went out


def test_duration_is_registered_and_feeds_the_playout_estimate():
    # 48000 bytes of 16-bit PCM at 24kHz is exactly 1 second.
    _, mark, _ = _run()
    assert mark.get_audio_playing_until() > 0
    durations = [m["duration"] for m in mark.get_chunk_marks()]
    assert durations == [1.0]


def test_mark_wire_format_unchanged():
    # The browser contract is {"type": "mark", "name": <id>}; duration is server-side only.
    _, _, ws = _run()
    marks = [json.loads(t) for t in ws.text]
    assert all(set(m) == {"type", "name"} for m in marks), marks


def test_single_byte_chunk_does_not_throw():
    handler, mark, _ = _run(audio=b"\x01")
    assert handler.is_closed() is False
    assert mark.get_audio_playing_until() > 0
