"""
Regression tests for bolna-ai/bolna#1126: a WebSocket that closes mid-reply must
still produce an end-of-synthesizer-stream packet, otherwise is_final_chunk
never fires and the agent stays marked as speaking for the rest of the call.

Provider receivers (Cartesia, Deepgram, Smallest, Maya, Pixa) break out of
receiver() on ConnectionClosed without yielding the EOS sentinel, so
_generate_ws_loop used to end the turn silently.
"""

import time
from collections import deque

from bolna.synthesizer.stream_synthesizer import StreamSynthesizer

EOS_SENTINEL = b"\x00"


class FakeDroppedSocketSynth:
    """Minimal self for driving the real StreamSynthesizer._generate_ws_loop,
    mirroring tests/test_stream_synthesizer_eos.py with turn bookkeeping."""

    def __init__(self, recv_items, text_metas, last_text_sent, turn_open):
        self.recv_items = recv_items  # list of (audio_bytes, extra_meta)
        self.text_queue = deque(text_metas)
        self.last_text_sent = last_text_sent
        self.connection_error = None
        self.meta_info = None
        self.provider_name = "fake"
        self.first_chunk_generated = False
        # Open turn only when unsettled — the real class clears it on eos.
        self.current_turn_start_time = time.perf_counter() if turn_open else None
        self.turn_latency_recorded = False

    async def receiver(self):
        for item in self.recv_items:
            yield item
        # Returns without yielding the EOS sentinel — like `break` on
        # ConnectionClosed in provider receivers.

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
        self.turn_latency_recorded = True
        self.current_turn_start_time = None


async def drain(fake):
    return [pkt async for pkt in StreamSynthesizer._generate_ws_loop(fake)]


async def test_receiver_early_return_emits_synthetic_eos():
    """Socket closed after text was sent but before the provider's end frame:
    the loop must emit end_of_synthesizer_stream so the turn settles."""
    metas = [{"sequence_id": 5, "end_of_llm_stream": True}]
    recv = [(b"audio", {})]
    synth = FakeDroppedSocketSynth(recv, metas, last_text_sent=True, turn_open=True)

    packets = await drain(synth)

    last = packets[-1]
    assert last["data"] == EOS_SENTINEL
    assert last["meta_info"]["end_of_synthesizer_stream"] is True
    # Text was fully sent (end_of_llm_stream), so the synthetic eos is final.
    assert last["meta_info"]["end_of_llm_stream"] is True
    assert synth.turn_latency_recorded is True


async def test_no_synthetic_eos_when_no_open_turn():
    """Receiver ending between turns must not emit a spurious eos packet."""
    recv = [(b"audio", {})]
    synth = FakeDroppedSocketSynth(recv, [], last_text_sent=False, turn_open=False)

    packets = await drain(synth)

    assert all(p["meta_info"].get("end_of_synthesizer_stream") is not True for p in packets)


async def test_receiver_early_return_incomplete_turn_not_final():
    """Interrupted turn (LLM never finished): synthetic eos must not mark final."""
    metas = [{"sequence_id": 7, "end_of_llm_stream": False}]
    recv = [(b"audio", {})]
    synth = FakeDroppedSocketSynth(recv, metas, last_text_sent=False, turn_open=True)

    packets = await drain(synth)

    last = packets[-1]
    assert last["meta_info"]["end_of_synthesizer_stream"] is True
    assert last["meta_info"].get("end_of_llm_stream", False) is False
