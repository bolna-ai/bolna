"""
Regression tests for SarvamSynthesizer.receiver() surviving a mid-call socket drop.

Bug (prod run 03d94c42, 2026-09-04): receiver() did `break` on ConnectionClosed.
SynthesizerPool iterates generate() exactly once, so ending the receiver killed the
audio path for the rest of the call while monitor_connection cheerfully redialled a
socket nobody read. No error, no hangup — 56s of dead air with the LLM still answering.

Second half of the same bug: the turn that died with the socket never got its 'final'
event, so no b"\\x00" reached _generate_ws_loop, so _record_turn_latency() never ran and
current_turn_start_time stayed latched — which disables the whole new-turn block in
_stamp_turn_start (including the stale text_queue drop) for every later turn.
"""

from collections import deque

import pytest
import websockets
from websockets.exceptions import ConnectionClosedError

from bolna.synthesizer.sarvam_synthesizer import SarvamSynthesizer
from bolna.synthesizer.stream_synthesizer import StreamSynthesizer

OPEN = websockets.protocol.State.OPEN
CLOSED = websockets.protocol.State.CLOSED


class FakeSocket:
    """Yields queued frames, then raises whatever `then` holds (default: closed)."""

    def __init__(self, frames, then=None, state=OPEN):
        self.frames = deque(frames)
        self.then = then or ConnectionClosedError(None, None)
        self.state = state

    async def recv(self):
        if self.frames:
            return self.frames.popleft()
        self.state = CLOSED
        raise self.then


def audio_frame(b64="YWJj"):
    return '{"type": "audio", "data": {"audio": "%s"}}' % b64


FINAL_FRAME = '{"data": {"event_type": "final"}}'


def make_synth():
    synth = SarvamSynthesizer(
        voice_id="simran",
        model="bulbul:v3",
        language="en-IN",
        stream=True,
        synthesizer_key="test-key",
    )
    return synth


async def take(gen, n):
    """Pull at most n items, stopping early if the generator finishes."""
    out = []
    for _ in range(n):
        try:
            out.append(await gen.__anext__())
        except StopAsyncIteration:
            break
    return out


async def test_receiver_survives_socket_close_and_resumes_on_new_socket():
    """The incident: socket dies mid-turn, monitor_connection redials, audio must flow again."""
    synth = make_synth()
    dead = FakeSocket([])  # raises ConnectionClosed immediately
    synth.websocket = dead
    synth.current_turn_start_time = 1.0  # a turn is in flight (seq 21)

    gen = synth.receiver()

    # The dying socket settles the lost turn rather than ending the generator.
    assert await gen.__anext__() == b"\x00"

    # Consumer would have run _record_turn_latency(); mimic that, then reconnect.
    synth.current_turn_start_time = None
    synth.websocket = FakeSocket([audio_frame(), FINAL_FRAME])

    assert await take(gen, 2) == [b"abc", b"\x00"]
    await gen.aclose()


async def test_socket_dropped_while_idle_emits_no_sentinel():
    """No turn in flight → a spurious eos would falsely end playback / flush the turn."""
    synth = make_synth()
    synth.websocket = FakeSocket([])
    synth.current_turn_start_time = None  # idle

    gen = synth.receiver()
    synth.websocket = FakeSocket([audio_frame("eHl6")])

    assert await take(gen, 1) == [b"xyz"]
    await gen.aclose()


async def test_socket_dying_between_recvs_settles_the_turn():
    """The close surfaces at the top-of-loop connectivity check, not via ConnectionClosed."""
    synth = make_synth()
    synth.websocket = FakeSocket([], state=CLOSED)
    synth.current_turn_start_time = 1.0

    gen = synth.receiver()
    assert await gen.__anext__() == b"\x00"
    await gen.aclose()


async def test_frame_from_replaced_socket_is_dropped():
    """A dying socket drains buffered frames; they must not play as the next turn."""
    synth = make_synth()
    fresh = FakeSocket([audio_frame("ZnJlc2g=")])

    class StaleSocket(FakeSocket):
        async def recv(self):
            # Replaced mid-recv, exactly as monitor_connection would.
            synth.websocket = fresh
            return await super().recv()

    synth.websocket = StaleSocket([audio_frame("c3RhbGU=")])

    gen = synth.receiver()
    # The stale frame is discarded, so the first audio out is the fresh socket's.
    assert await take(gen, 1) == [b"fresh"]
    await gen.aclose()


async def test_receiver_still_gives_up_on_connection_error():
    """The give-up path must stay reachable — it raises, which ends the call loudly."""
    synth = make_synth()
    synth.websocket = FakeSocket([])
    synth.current_turn_start_time = None
    synth.connection_error = "boom"

    gen = synth.receiver()
    assert await take(gen, 3) == []  # returns immediately


async def test_receiver_returns_when_conversation_ended():
    synth = make_synth()
    synth.websocket = FakeSocket([])
    synth.conversation_ended = True

    gen = synth.receiver()
    assert await take(gen, 3) == []


# ----------------------------------------------------------------------
# The latch: settling the lost turn must clear current_turn_start_time so the
# NEXT turn is stamped fresh (and drops the dead turn's stale text_queue entries).
# ----------------------------------------------------------------------


class FakeStreamSynth:
    """Minimal self for driving the real StreamSynthesizer._generate_ws_loop."""

    def __init__(self, recv_items, text_metas):
        self.recv_items = recv_items
        self.text_queue = deque(text_metas)
        self.last_text_sent = True
        self.connection_error = None
        self.meta_info = None
        self.provider_name = "sarvam"
        self.first_chunk_generated = False
        self.current_turn_start_time = 1.0
        self.current_turn_id = 20
        self.current_sequence_id = 21
        self.current_message_category = None
        self.current_tts_start_ms = 100
        self.ws_send_time = None
        self.current_turn_ttfb = None
        self.current_sequence_chars = 38
        self.turn_latencies = []

    async def receiver(self):
        for item in self.recv_items:
            yield item

    def _unpack_receiver_message(self, raw_item):
        return raw_item, {}

    def _compute_first_result_latency(self):
        pass

    def _get_audio_format(self):
        return "wav"

    def _stamp_first_chunk(self, meta_info):
        pass

    def _process_audio_chunk(self, audio):
        return audio

    def _stamp_mark_id(self, meta_info):
        pass

    _upsert_turn_latency = StreamSynthesizer._upsert_turn_latency
    _record_turn_latency = StreamSynthesizer._record_turn_latency
    has_unsettled_turn = StreamSynthesizer.has_unsettled_turn


async def test_lost_turn_sentinel_unlatches_turn_state():
    fake = FakeStreamSynth([b"\x00"], [{"sequence_id": 21, "end_of_llm_stream": True}])
    assert fake.has_unsettled_turn() is True

    packets = [pkt async for pkt in StreamSynthesizer._generate_ws_loop(fake)]

    assert packets[-1]["meta_info"]["end_of_synthesizer_stream"] is True
    # Latch cleared → _stamp_turn_start will treat the next push as a new turn.
    assert fake.has_unsettled_turn() is False
    assert fake.current_sequence_id is None


async def test_next_turn_is_stamped_fresh_after_a_lost_turn():
    """End-to-end of the second-order bug: stale meta must not leak into the next turn."""
    synth = make_synth()
    # Dead turn left an entry behind in the text_queue.
    synth.text_queue = deque([{"sequence_id": 21}])
    synth.current_turn_start_time = None  # settled by the lost-turn sentinel
    synth.last_text_sent = True

    synth._stamp_turn_start({"sequence_id": 22, "turn_id": 21, "text": "hello", "tts_start_ms": 5})

    # Stale seq-21 entry dropped, so the next turn's first chunk cannot inherit a dead
    # sequence_id and get filtered out by __listen_synthesizer.
    assert [m["sequence_id"] for m in synth.text_queue] == []
    assert synth.current_sequence_id == 22
    assert synth.last_text_sent is False


@pytest.mark.parametrize("latched", [True, False])
async def test_has_unsettled_turn_tracks_turn_start_time(latched):
    synth = make_synth()
    synth.current_turn_start_time = 1.0 if latched else None
    assert synth.has_unsettled_turn() is latched
