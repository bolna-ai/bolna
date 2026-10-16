"""A turn whose socket closes before its end-of-stream is ended by StreamSynthesizer.

Without it the turn stays open: current_turn_start_time stays latched, so the next turn keeps the
dead turn's text_queue metas and its first audio goes out tagged with the dead sequence_id, which
__listen_synthesizer skips. Covers every provider whose receiver() does not settle such turns itself."""

import asyncio
import base64
import json
from collections import deque
from unittest.mock import MagicMock

import websockets
import websockets.protocol
from websockets.exceptions import ConnectionClosedError

from bolna.synthesizer.cartesia_synthesizer import CartesiaSynthesizer
from bolna.synthesizer.soniox_synthesizer import SonioxSynthesizer
from bolna.synthesizer.stream_synthesizer import StreamSynthesizer

OPEN = websockets.protocol.State.OPEN
CLOSED = websockets.protocol.State.CLOSED


class FakeSocket:
    """Yields queued frames, then closes."""

    def __init__(self, frames=(), state=OPEN):
        self.frames = deque(frames)
        self.state = state
        self.sent = []

    async def send(self, message):
        self.sent.append(json.loads(message))

    async def recv(self):
        if self.frames:
            return self.frames.popleft()
        self.state = CLOSED
        raise ConnectionClosedError(None, None)


class BreakingSynth(StreamSynthesizer):
    """Receiver shaped like Cartesia/Deepgram/Rime/Smallest/Maya: ends on ConnectionClosed."""

    def __init__(self):
        super().__init__(stream=True, provider_name="fake", task_manager_instance=MagicMock())
        self.task_manager_instance.is_sequence_id_in_current_ids.return_value = True

    async def sender(self, text, sequence_id, end_of_llm_stream=False):
        await self._send_json({"text": text})
        if end_of_llm_stream:
            self.last_text_sent = True

    async def receiver(self):
        while True:
            if not self._is_ws_connected():
                await asyncio.sleep(0.01)
                continue
            try:
                frame = await self.websocket.recv()
            except websockets.exceptions.ConnectionClosed:
                break
            yield b"\x00" if frame == "done" else frame.encode()


async def push(synth, text, seq, end=False):
    await synth.push({"data": text, "meta_info": {"sequence_id": seq, "turn_id": seq, "end_of_llm_stream": end}})
    await synth.sender_task


def summary(packets):
    return [
        (p["data"], p["meta_info"].get("sequence_id"), bool(p["meta_info"].get("end_of_synthesizer_stream")))
        for p in packets
    ]


async def test_socket_dropped_mid_turn_ends_the_turn():
    synth = BreakingSynth()
    synth.websocket = FakeSocket(["A1"])
    await push(synth, "Sure, let me check", 10)
    await push(synth, " that for you.", 10, end=True)

    packets = [p async for p in synth.generate()]

    assert summary(packets) == [(b"A1", 10, False), (b"\x00", 10, True)]
    assert packets[-1]["meta_info"]["end_of_llm_stream"] is True  # so is_final_chunk fires
    assert synth.current_turn_start_time is None
    assert synth.turn_latencies[-1]["sequence_id"] == 10


async def test_socket_closed_while_suspended_at_a_yield_ends_the_turn():
    """Receivers skip a closed socket, so the turn's own end-of-stream can never arrive."""
    synth = BreakingSynth()
    ws = FakeSocket(["A1"])
    synth.websocket = ws
    await push(synth, "Hello there.", 10, end=True)

    gen = synth.generate()
    assert (await gen.__anext__())["data"] == b"A1"
    ws.state = CLOSED  # closes while the consumer still holds A1

    eos = await asyncio.wait_for(gen.__anext__(), 1)
    assert summary([eos]) == [(b"\x00", 10, True)]
    await gen.aclose()


async def test_next_reply_is_not_tagged_with_the_dropped_turn():
    synth = CartesiaSynthesizer(
        voice="v",
        voice_id="vid",
        task_manager_instance=MagicMock(),
        use_mulaw=True,
        caching=False,
        synthesizer_key="k",
        stream=True,
    )
    synth.task_manager_instance.is_sequence_id_in_current_ids.return_value = True

    def audio(payload):
        return json.dumps({"context_id": synth.context_id, "data": base64.b64encode(payload).decode()})

    synth.websocket = FakeSocket()
    await push(synth, "Sure, let me check", 10)
    await push(synth, " that for you.", 10, end=True)
    synth.websocket.frames.append(audio(b"A1"))
    assert summary([p async for p in synth.generate()]) == [(b"A1", 10, False), (b"\x00", 10, True)]

    synth.websocket = FakeSocket()
    await push(synth, "Your order shipped", 11)
    await push(synth, " yesterday.", 11, end=True)
    synth.websocket.frames.extend(
        [audio(b"B1"), audio(b"B2"), json.dumps({"context_id": synth.context_id, "done": True})]
    )

    assert summary([p async for p in synth.generate()]) == [(b"B1", 11, False), (b"B2", 11, False), (b"\x00", 11, True)]


async def test_socket_dropped_while_idle_emits_no_end_of_stream():
    synth = BreakingSynth()
    synth.websocket = FakeSocket(["A1", "done"])
    await push(synth, "Hello there.", 10, end=True)

    assert summary([p async for p in synth.generate()]) == [(b"A1", 10, False), (b"\x00", 10, True)]


async def test_turn_not_yet_sent_on_the_dead_socket_is_not_ended():
    """A turn pushed while redialling has sent nothing on this socket; its audio comes later."""
    synth = BreakingSynth()
    synth.websocket = FakeSocket()
    synth.current_turn_start_time = 1.0
    synth.current_turn_socket = None

    assert [p async for p in synth.generate()] == []


async def test_provider_that_settles_its_own_lost_turns_is_left_alone():
    synth = BreakingSynth()
    synth.settles_lost_turns = True
    synth.websocket = FakeSocket(["A1"])
    await push(synth, "Hello there.", 10, end=True)

    assert summary([p async for p in synth.generate()]) == [(b"A1", 10, False)]


async def test_send_claims_the_socket_only_for_an_open_turn():
    synth = BreakingSynth()
    ws = FakeSocket()
    synth.websocket = ws

    await synth._send_json({"type": "keepalive"})
    assert synth.current_turn_socket is None

    await push(synth, "Hello there.", 10)
    assert synth.current_turn_socket is ws


async def test_soniox_socket_dropped_mid_turn_ends_the_turn():
    """Soniox's receiver keeps looping on a lost socket, so it ends the turn itself."""
    synth = SonioxSynthesizer(
        voice="Adrian",
        use_mulaw=True,
        caching=False,
        synthesizer_key="k",
        stream=True,
        task_manager_instance=MagicMock(),
    )
    synth.task_manager_instance.is_sequence_id_in_current_ids.return_value = True
    ws = FakeSocket()
    synth.websocket = ws
    await push(synth, "Hello there.", 10, end=True)
    ws.frames.append(json.dumps({"stream_id": synth.stream_id, "audio": base64.b64encode(b"A1").decode()}))

    gen = synth.generate()
    first = await gen.__anext__()
    eos = await asyncio.wait_for(gen.__anext__(), 1)

    assert summary([first, eos]) == [(b"A1", 10, False), (b"\x00", 10, True)]
    assert synth.current_turn_start_time is None
    await gen.aclose()
