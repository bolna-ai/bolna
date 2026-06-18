"""Regression: TranscriberPool.turn_latencies must return turns in true conversation
order even when a mid-call language switch spread them across per-language instances.

Each instance only records the turns it was active for; aggregating by dict (label)
order interleaved them (e.g. a post-switch turn_3 ahead of turn_1/turn_2), which
mis-paired the user transcript with the wrong agent turn in the latency dict.
"""

import asyncio
from unittest.mock import AsyncMock

from bolna.transcriber.transcriber_pool import TranscriberPool


class FakeTranscriber:
    def __init__(self, turn_latencies):
        self.run = AsyncMock()
        self.input_queue = asyncio.Queue()
        self.transcription_task = None
        self.turn_latencies = turn_latencies


def _pool(transcribers, active):
    return TranscriberPool(
        transcribers=transcribers,
        shared_input_queue=asyncio.Queue(),
        output_queue=asyncio.Queue(),
        active_label=active,
        multilingual_config={},
    )


def test_turn_latencies_sorted_across_switched_instances():
    # 'hi' was active for turns 1 & 2, then the call switched to 'en' which recorded
    # turn 3. Dict order lists 'en' first, so without the sort the aggregate would be
    # turn_3, turn_1, turn_2 (the bug observed on QA 4c3dd300).
    transcribers = {
        "en": FakeTranscriber([{"turn_id": "turn_3", "asr_start_epoch_ms": 3000}]),
        "hi": FakeTranscriber(
            [
                {"turn_id": "turn_1", "asr_start_epoch_ms": 1000},
                {"turn_id": "turn_2", "asr_start_epoch_ms": 2000},
            ]
        ),
    }
    ids = [t["turn_id"] for t in _pool(transcribers, "en").turn_latencies]
    assert ids == ["turn_1", "turn_2", "turn_3"]


def test_turn_latencies_missing_asr_start_does_not_crash():
    # A turn without asr_start_epoch_ms (e.g. final_transcript never finalized) sorts as
    # 0 rather than raising on the None comparison.
    transcribers = {
        "hi": FakeTranscriber(
            [
                {"turn_id": "turn_2", "asr_start_epoch_ms": 2000},
                {"turn_id": "turn_early"},  # no asr_start_epoch_ms
            ]
        ),
    }
    ids = [t["turn_id"] for t in _pool(transcribers, "hi").turn_latencies]
    assert ids == ["turn_early", "turn_2"]


def test_turn_latencies_empty_when_no_turns():
    transcribers = {"hi": FakeTranscriber([]), "en": FakeTranscriber([])}
    assert _pool(transcribers, "hi").turn_latencies == []
