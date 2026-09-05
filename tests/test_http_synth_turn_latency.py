"""The non-streaming HTTP synth loop records a turn-latency entry, so HTTP-only providers
(gemini, openai) report synthesizer latency the same way the streaming synths do."""

import asyncio
from unittest.mock import MagicMock

from bolna.synthesizer.base_synthesizer import BaseSynthesizer


class _HttpSynth(BaseSynthesizer):
    def __init__(self, audio=b"\x01\x02", **kwargs):
        super().__init__(kwargs.get("task_manager_instance"))
        self.caching = False
        self._audio = audio

    async def _generate_http(self, text):
        return self._audio

    def _get_http_audio_format(self):
        return "pcm"


def _synth(**kwargs):
    tm = MagicMock()
    tm.is_sequence_id_in_current_ids.return_value = True
    return _HttpSynth(task_manager_instance=tm, **kwargs)


def _drive_one(synth, text, meta):
    synth.internal_queue.put_nowait({"data": text, "meta_info": meta})

    async def run():
        async for _ in synth._generate_http_loop():
            return

    asyncio.run(run())


def test_http_loop_records_one_turn_latency():
    s = _synth()
    _drive_one(s, "hello there", {"sequence_id": 3, "turn_id": 3, "message_category": "", "tts_start_ms": 1234.5})
    assert len(s.turn_latencies) == 1
    rec = s.turn_latencies[0]
    assert (rec["sequence_id"], rec["turn_id"]) == (3, 3)
    assert rec["characters"] == len("hello there")
    assert rec["tts_start_ms"] == 1234.5
    # Non-streaming: the whole utterance renders at once, so first_result == total.
    assert rec["first_result_latency_ms"] == rec["total_stream_duration_ms"] >= 0


def test_failed_render_still_records_latency():
    """A None render becomes the b\"\\x00\" sentinel, but the time it took is still recorded."""
    s = _synth(audio=None)
    _drive_one(s, "hi", {"sequence_id": 1, "turn_id": 1})
    assert len(s.turn_latencies) == 1
    assert s.turn_latencies[0]["first_result_latency_ms"] >= 0
