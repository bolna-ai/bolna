"""The OpenAI transcriber ends a turn at its speech_stopped commit, and transcript deltas only arrive
after it. The user's start is the speech_started event, on the same clock as that commit."""

import asyncio
import json
import time as real_time

from bolna.transcriber import openai_transcriber
from bolna.transcriber.openai_transcriber import OpenAITranscriber


class ReplaySocket:
    def __init__(self, events):
        self.events = [json.dumps(event) for event in events]

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.events:
            raise StopAsyncIteration
        return self.events.pop(0)


class SteppingTime:
    """time stand-in whose clock moves 0.3 s on every read, so event order shows in the stamps."""

    def __init__(self, start):
        self.now = start

    def time(self):
        self.now += 0.3
        return self.now

    def __getattr__(self, name):
        return getattr(real_time, name)


def test_server_vad_turn_starts_before_it_ends(monkeypatch):
    monkeypatch.setattr(openai_transcriber, "time", SteppingTime(1_700_000_000.0))
    transcriber = OpenAITranscriber("twilio", output_queue=asyncio.Queue())
    transcriber.meta_info = {}
    events = [
        {"type": "input_audio_buffer.speech_started"},
        {"type": "input_audio_buffer.speech_stopped"},
        {"type": "conversation.item.input_audio_transcription.delta", "delta": "yes"},
    ]

    async def receive():
        return [packet async for packet in transcriber.receiver(ReplaySocket(events))]

    asyncio.run(receive())

    assert transcriber.meta_info["user_start_ts_wall"] < transcriber.meta_info["user_stop_ts_wall"]
