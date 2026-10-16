"""Soniox user speech timing: both ends of a turn come from token positions on the same clock.

The start used to be when the first interim arrived and the end now minus the endpoint delay, which is
only Soniox's maximum. A short reply then ended before it started.
"""

import asyncio
import json

from bolna.constants import SONIOX_ENDPOINT_TOKEN
from bolna.transcriber.soniox_transcriber import SonioxTranscriber

STREAM_START_MS = 1_700_000_000_000.0


class ReplaySocket:
    def __init__(self, responses):
        self.responses = [json.dumps(response) for response in responses]

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.responses:
            raise StopAsyncIteration
        return self.responses.pop(0)


def soniox_leg_with_live_audio(seconds):
    """A leg that has sent `seconds` of audio in 100 ms frames, each sent as it filled."""
    leg = SonioxTranscriber("web_based_call", output_queue=asyncio.Queue())
    leg.meta_info = {}
    for index in range(int(seconds * 10)):
        leg.record_audio_frame(0.1, STREAM_START_MS + (index + 1) * 100)
    return leg


async def receive(leg, responses):
    return [packet async for packet in leg.receiver(ReplaySocket(responses))]


def test_a_short_reply_starts_before_it_ends():
    leg = soniox_leg_with_live_audio(10)
    words = [
        {"text": "ha", "start_ms": 4000, "end_ms": 4200, "is_final": True},
        {"text": "an", "start_ms": 4200, "end_ms": 4500, "is_final": True},
    ]

    packets = asyncio.run(receive(leg, [{"tokens": words}, {"tokens": [{"text": SONIOX_ENDPOINT_TOKEN}]}]))

    interim = next(
        p for p in packets if isinstance(p["data"], dict) and p["data"]["type"] == "interim_transcript_received"
    )
    transcript = next(p for p in packets if isinstance(p["data"], dict) and p["data"]["type"] == "transcript")
    user_start = interim["meta_info"]["user_start_ts_wall"]
    user_end = transcript["meta_info"]["user_stop_ts_wall"]

    assert abs(user_start - (STREAM_START_MS / 1000 + 4.0)) < 0.001
    assert abs(user_end - (STREAM_START_MS / 1000 + 4.5)) < 0.001


def test_the_next_turn_does_not_inherit_the_previous_onset():
    leg = soniox_leg_with_live_audio(10)
    words = [{"text": "yes", "start_ms": 1000, "end_ms": 1300, "is_final": True}]

    asyncio.run(receive(leg, [{"tokens": words}, {"tokens": [{"text": SONIOX_ENDPOINT_TOKEN}]}]))

    assert "user_start_ts_wall" not in leg.meta_info
    assert leg.turn_last_word_end_ms is None
