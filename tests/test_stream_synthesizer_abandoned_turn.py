"""A TTS turn cut off before its eos must not hand its start to the next sequence.

The turn timer only opened when none was running and only closed at an eos, so a sequence cancelled
mid-stream left it running: the next sequence reused its tts_start and first-result latency (one call
showed seq 5's TTS starting before its own LLM, with 1545 ms first audio against ~165 ms). Sarvam
also still ends an abandoned, already-flushed turn with its `final`, which must not close the next turn.
"""

import asyncio
from unittest.mock import MagicMock

import pytest

from bolna.synthesizer.stream_synthesizer import StreamSynthesizer
from bolna.synthesizer.synthesizer_pool import SynthesizerPool

EOS = b"\x00"
AUDIO = b"\x01\x02"


class ScriptedSynth(StreamSynthesizer):
    """Streams scripted frames; flushing a turn sets last_text_sent like Sarvam's sender does."""

    def __init__(self, delivers_eos_after_interruption=True):
        task_manager = MagicMock()
        task_manager.conversation_start_init_ts = 0
        self.retired = set()
        task_manager.is_sequence_id_in_current_ids.side_effect = lambda seq: seq not in self.retired
        super().__init__(provider_name="scripted", task_manager_instance=task_manager)
        self.delivers_eos_after_interruption = delivers_eos_after_interruption
        self.frames = []

    async def sender(self, text, sequence_id, end_of_llm_stream=False):
        if self.ws_send_time is None:
            self.ws_send_time = 0.0
        if end_of_llm_stream:
            self.last_text_sent = True

    async def receiver(self):
        for frame in self.frames:
            yield frame

    def _get_audio_format(self):
        return "pcm"

    def _process_audio_chunk(self, chunk):
        return chunk


async def push(synth, sequence_id, tts_start_ms, flush=True):
    meta = {"sequence_id": sequence_id, "turn_id": 5, "tts_start_ms": tts_start_ms, "end_of_llm_stream": flush}
    await synth.push({"data": "text", "meta_info": meta})
    await synth.sender_task


async def play(synth, frames):
    synth.frames = frames
    return [packet async for packet in synth.generate()]


def entry(synth, sequence_id):
    return next(e for e in synth.turn_latencies if e["sequence_id"] == sequence_id)


def test_a_continuation_does_not_hand_the_cancelled_turn_to_the_next_sequence():
    async def scenario():
        synth = ScriptedSynth()
        await push(synth, 4, tts_start_ms=11379.59)
        synth.retired.add(4)
        synth.abandon_open_turn()  # interruption cleanup
        await push(synth, 5, tts_start_ms=12900.0)
        # The cancelled seq 4 still ends with its own eos, then seq 5 plays and ends.
        await play(synth, [EOS, AUDIO, EOS])
        return synth

    synth = asyncio.run(scenario())

    assert entry(synth, 4)["tts_start_ms"] == 11379.59
    assert "cancelled_at_ms" in entry(synth, 4)
    assert entry(synth, 5)["tts_start_ms"] == 12900.0
    assert "total_stream_duration_ms" in entry(synth, 5)  # closed by its own eos, not seq 4's
    assert synth.current_turn_start_time is None and synth.stale_eos_pending == 0


def test_a_retired_open_turn_is_abandoned_by_the_next_push_alone():
    async def scenario():
        synth = ScriptedSynth()
        await push(synth, 4, tts_start_ms=11379.59)
        synth.retired.add(4)
        await push(synth, 5, tts_start_ms=12900.0)
        return synth

    synth = asyncio.run(scenario())

    assert "cancelled_at_ms" in entry(synth, 4)
    assert entry(synth, 5)["tts_start_ms"] == 12900.0


@pytest.mark.parametrize(
    "delivers, flushed",
    [pytest.param(False, True, id="provider-drops-abandoned-eos"), pytest.param(True, False, id="never-flushed")],
)
def test_no_stale_eos_is_expected_when_none_will_come(delivers, flushed):
    async def scenario():
        synth = ScriptedSynth(delivers_eos_after_interruption=delivers)
        await push(synth, 4, tts_start_ms=1.0, flush=flushed)
        synth.abandon_open_turn()
        return synth

    synth = asyncio.run(scenario())

    assert synth.stale_eos_pending == 0


def test_the_pool_abandons_the_active_synthesizer_turn():
    active = MagicMock()
    pool = SynthesizerPool.__new__(SynthesizerPool)
    pool.synthesizers = {"hi": active}
    pool.active_label = "hi"

    pool.abandon_open_turn()

    active.abandon_open_turn.assert_called_once()


def test_a_stale_eos_neither_takes_the_next_turns_meta_nor_goes_downstream():
    async def scenario():
        synth = ScriptedSynth()
        await push(synth, 4, tts_start_ms=1.0)
        synth.retired.add(4)
        synth.abandon_open_turn()
        await push(synth, 5, tts_start_ms=2.0)
        packets = await play(synth, [EOS, AUDIO])
        return synth, packets

    synth, packets = asyncio.run(scenario())

    assert [p["data"] for p in packets] == [AUDIO]
    assert packets[0]["meta_info"]["sequence_id"] == 5


def test_an_owed_eos_that_never_comes_stops_being_expected(monkeypatch):
    async def scenario():
        synth = ScriptedSynth()
        await push(synth, 4, tts_start_ms=1.0)
        synth.retired.add(4)
        synth.abandon_open_turn()
        await push(synth, 5, tts_start_ms=2.0)
        synth.stale_eos_deadline = 0.0  # the window has passed with no eos from seq 4
        await play(synth, [AUDIO, EOS])
        return synth

    synth = asyncio.run(scenario())

    assert "total_stream_duration_ms" in entry(synth, 5)  # seq 5's own eos closed it
    assert synth.stale_eos_pending == 0
