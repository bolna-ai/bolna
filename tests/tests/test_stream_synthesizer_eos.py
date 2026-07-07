"""
Regression tests for StreamSynthesizer._generate_ws_loop end-of-stream tagging.

Bug: ElevenLabs buffers short replies into fewer audio messages than there were
LLM pushes, so the eos sentinel (b"\\x00") pops a non-final chunk's meta_info via
text_queue.popleft(). is_final_chunk = end_of_llm_stream AND end_of_synthesizer_stream,
so the eos mark came out final=False and is_audio_being_played latched True (caller
replies dropped as false interruptions). Fix: when the turn's LLM stream has ended
(last_text_sent), stamp end_of_llm_stream on the eos packet regardless of which meta
was popped.
"""

from collections import deque

import pytest

from bolna.synthesizer.stream_synthesizer import StreamSynthesizer


class FakeStreamSynth:
    """Minimal self for driving the real StreamSynthesizer._generate_ws_loop."""

    def __init__(self, recv_items, text_metas, last_text_sent):
        self.recv_items = recv_items  # list of (audio_bytes, extra_meta)
        self.text_queue = deque(text_metas)
        self.last_text_sent = last_text_sent
        self.connection_error = None
        self.meta_info = None
        self.provider_name = "elevenlabs"
        self.first_chunk_generated = False

    async def receiver(self):
        for item in self.recv_items:
            yield item

    def _unpack_receiver_message(self, raw_item):
        return raw_item

    def _compute_first_result_latency(self):
        pass

    def _get_audio_format(self):
        return "mulaw"

    def _stamp_first_chunk(self, meta_info):
        pass

    def _process_audio_chunk(self, audio):
        return audio

    def _stamp_mark_id(self, meta_info):
        pass

    def _record_turn_latency(self):
        pass


async def drain(fake):
    return [pkt async for pkt in StreamSynthesizer._generate_ws_loop(fake)]


@pytest.mark.asyncio
async def test_eos_carries_end_of_llm_stream_when_turn_ended():
    # 3 LLM pushes (eol on the last), but only 2 receiver yields (audio + eos):
    # popleft gives the eos packet chunk2's meta (eol=False), stranding chunk3.
    metas = [
        {"sequence_id": 2, "end_of_llm_stream": False},
        {"sequence_id": 2, "end_of_llm_stream": False},
        {"sequence_id": 2, "end_of_llm_stream": True},
    ]
    recv = [(b"audio", {}), (b"\x00", {})]
    packets = await drain(FakeStreamSynth(recv, metas, last_text_sent=True))

    eos = packets[-1]
    assert eos["data"] == b"\x00"
    assert eos["meta_info"]["end_of_synthesizer_stream"] is True
    assert eos["meta_info"]["end_of_llm_stream"] is True  # <- final mark now fires


@pytest.mark.asyncio
async def test_eos_does_not_force_end_of_llm_stream_when_turn_not_ended():
    # Interrupted/incomplete turn: end_of_llm_stream was never sent, so the eos
    # must NOT be forced final (no premature is_final_chunk).
    metas = [{"sequence_id": 2, "end_of_llm_stream": False}]
    recv = [(b"\x00", {})]
    packets = await drain(FakeStreamSynth(recv, metas, last_text_sent=False))

    eos = packets[-1]
    assert eos["meta_info"]["end_of_synthesizer_stream"] is True
    assert eos["meta_info"].get("end_of_llm_stream", False) is False
