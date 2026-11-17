"""
Web calls with Smallest TTS sounded choppy after the welcome message.

__listen_synthesizer sleeps synthesizer.get_sleep_time() after every streamed chunk.
Smallest inherited BaseSynthesizer's 0.2s while each of its chunks holds 160ms of 24kHz
audio, so playout ran dry for ~40ms on every chunk. The cached welcome clip skips that loop.
"""

import base64
import json

import websockets

from bolna.constants import WEBCALL_TTS_SAMPLE_RATE
from bolna.synthesizer.smallest_synthesizer import SmallestSynthesizer

CHUNK_BYTES = 7680  # size seen on prod web calls: 160ms of 16-bit PCM at 24kHz


class FakeSmallestSocket:
    state = websockets.protocol.State.OPEN

    def __init__(self, chunks):
        self.messages = [
            json.dumps({"status": "chunk", "data": {"audio": base64.b64encode(c).decode()}}) for c in chunks
        ]
        self.messages.append(json.dumps({"status": "complete"}))

    async def recv(self):
        if not self.messages:
            raise websockets.exceptions.ConnectionClosedOK(None, None)
        return self.messages.pop(0)


async def test_streamed_audio_reaches_output_faster_than_it_plays():
    synth = SmallestSynthesizer(
        voice_id="candela", stream=True, synthesizer_key="test", sampling_rate=WEBCALL_TTS_SAMPLE_RATE
    )
    synth.websocket = FakeSmallestSocket([b"\x01\x00" * (CHUNK_BYTES // 2)] * 10)

    audio_bytes = 0
    listener_sleep_s = 0.0
    async for packet in synth.generate():
        if packet["data"] != b"\x00":
            audio_bytes += len(packet["data"])
        listener_sleep_s += synth.get_sleep_time()  # what __listen_synthesizer awaits per packet

    audio_s = audio_bytes / 2 / WEBCALL_TTS_SAMPLE_RATE
    assert audio_s == 1.6
    assert listener_sleep_s < audio_s / 2, f"listener sleeps {listener_sleep_s:.2f}s to forward {audio_s:.2f}s of audio"


def test_http_mode_keeps_default_sleep():
    synth = SmallestSynthesizer(voice_id="candela", stream=False, synthesizer_key="test")
    assert synth.get_sleep_time() == 0.2
