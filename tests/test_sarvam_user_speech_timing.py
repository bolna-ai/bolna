"""Sarvam sends a turn's transcript only after its END_SPEECH, so stamping the user's start when the
transcript arrived always put it after the end. The start is the START_SPEECH event, on the same
clock as the END_SPEECH used for the end."""

import asyncio
import json
import time as real_time

from bolna.transcriber import sarvam_transcriber
from bolna.transcriber.sarvam_transcriber import SarvamTranscriber


class ReplaySocket:
    def __init__(self, messages):
        self.messages = [json.dumps(message) for message in messages]

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.messages:
            raise StopAsyncIteration
        return self.messages.pop(0)


class SteppingTime:
    """time stand-in whose clock moves 0.3 s on every read, so event order shows in the stamps."""

    def __init__(self, start):
        self.now = start

    def time(self):
        self.now += 0.3
        return self.now

    def __getattr__(self, name):
        return getattr(real_time, name)


def vad(signal):
    return {"type": "events", "data": {"signal_type": signal}}


def test_the_reply_starts_before_it_ends(monkeypatch):
    monkeypatch.setattr(sarvam_transcriber, "time", SteppingTime(1_700_000_000.0))
    transcriber = SarvamTranscriber(
        telephony_provider="twilio", model="saarika:v2.5", language="hi-IN", stream=True, transcriber_key="k"
    )
    transcriber.meta_info = {}
    messages = [vad("START_SPEECH"), vad("END_SPEECH"), {"type": "data", "data": {"transcript": "haan ji"}}]

    async def receive():
        return [packet async for packet in transcriber.receiver(ReplaySocket(messages))]

    packets = asyncio.run(receive())
    interim = next(
        p for p in packets if isinstance(p["data"], dict) and p["data"]["type"] == "interim_transcript_received"
    )

    assert interim["meta_info"]["user_start_ts_wall"] < interim["meta_info"]["user_stop_ts_wall"]
