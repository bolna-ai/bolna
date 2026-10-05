"""ElevenLabs multi-stream receiver surviving a socket that closes mid-reply.

websockets keeps returning frames queued before a close, and the turbo consumer reads one
message per 0.2s, so a long reply's tail and its real isFinal are often still queued when the
server closes. The receiver must read every socket until ConnectionClosed, never stop on its
state, and only then end a turn that got no isFinal. A send racing the close is not fatal, and
an interruption on a dead socket must not leave the next turn on an ignored context.
"""

import asyncio
import base64
import json
import logging
from collections import deque

import websockets
from websockets.exceptions import ConnectionClosedError

from bolna.synthesizer.elevenlabs_synthesizer import ElevenlabsSynthesizer

OPEN = websockets.protocol.State.OPEN
CLOSING = websockets.protocol.State.CLOSING
CLOSED = websockets.protocol.State.CLOSED
CTX = "ctx-turn"


def _audio(text="ab", ctx=CTX):
    return json.dumps(
        {"audio": base64.b64encode(b"\x01\x02").decode(), "alignment": {"chars": list(text)}, "contextId": ctx}
    )


def _final(ctx=CTX):
    return json.dumps({"isFinal": True, "contextId": ctx})


class FakeSocket:
    """websockets 15 semantics: queued frames are returned whatever the state; recv raises
    ConnectionClosed only once the queue is empty and the socket is CLOSED."""

    def __init__(self, frames=(), state=CLOSED, send_raises=False, close_code=1008, close_reason="policy violation"):
        self.frames = deque(frames)
        self.state = state
        self.send_raises = send_raises
        self.close_code = close_code
        self.close_reason = close_reason
        self.sent = []

    async def recv(self):
        while not self.frames and self.state is not CLOSED:
            await asyncio.sleep(0.01)
        if self.frames:
            return self.frames.popleft()
        raise ConnectionClosedError(None, None)

    async def send(self, data):
        if self.send_raises:
            self.state = CLOSED  # the close handshake completes
            raise ConnectionClosedError(None, None)
        self.sent.append(json.loads(data))


class StubTaskManager:
    def is_sequence_id_in_current_ids(self, sequence_id):
        return True


def _make_synth():
    synth = ElevenlabsSynthesizer(
        voice="v", voice_id="vid", synthesizer_key="test", caching=False, task_manager_instance=StubTaskManager()
    )
    synth.current_turn_context_id = CTX
    return synth


def _turn_in_flight(synth, ws, fully_sent=True):
    synth.current_turn_start_time = 1.0
    synth.current_turn_socket = ws
    synth.last_text_sent = fully_sent
    synth.ws_send_time = 1.0


class Reader:
    """Runs a generator in the background: an idle wait must not cancel (and so end) it."""

    def __init__(self, gen, pace=0.0):
        self.gen = gen
        self.pace = pace  # task_manager sleeps get_sleep_time() between synthesizer messages
        self.items = []
        self.task = asyncio.create_task(self._run())

    async def _run(self):
        async for item in self.gen:
            self.items.append(item)
            await asyncio.sleep(self.pace)

    async def after(self, seconds=0.3):
        await asyncio.sleep(seconds)
        return self.items

    async def stop(self):
        self.task.cancel()
        try:
            await self.task
        except asyncio.CancelledError:
            pass


def _kinds(items):
    return ["eos" if audio == b"\x00" else "audio" for audio, _ in items]


async def _receive(synth, seconds=0.3, pace=0.0):
    reader = Reader(synth.receiver(), pace=pace)
    items = list(await reader.after(seconds))
    await reader.stop()
    return items


async def _packets(synth, seconds=0.3):
    """Drive the real consumer loop, which settles each turn on end-of-stream."""
    reader = Reader(synth._generate_ws_loop())
    items = list(await reader.after(seconds))
    await reader.stop()
    return items


# --- draining: frames queued before the close are delivered ------------------------------


async def test_queued_tail_and_real_isfinal_are_read_after_close():
    # server sent the whole reply and isFinal, then closed before the paced reader caught up
    synth = _make_synth()
    dead = FakeSocket([_audio() for _ in range(30)] + [_final()])
    synth.websocket = dead
    _turn_in_flight(synth, dead)
    synth.text_queue.append({"sequence_id": 4, "turn_id": 4})

    packets = await _packets(synth)
    audio = [p for p in packets if p["data"] != b"\x00"]
    eos = [p for p in packets if p["data"] == b"\x00"]
    assert len(audio) == 30
    assert len(eos) == 1  # the real isFinal; no synthetic one on top
    assert synth.current_turn_start_time is None


async def test_turn_with_no_isfinal_ends_after_its_tail_is_read():
    synth = _make_synth()
    dead = FakeSocket([_audio() for _ in range(5)])
    synth.websocket = dead
    _turn_in_flight(synth, dead)
    synth.text_queue.append({"sequence_id": 4, "turn_id": 4})

    packets = await _packets(synth)
    assert [p["data"] == b"\x00" for p in packets] == [False] * 5 + [True]
    final = packets[-1]["meta_info"]
    assert final["end_of_synthesizer_stream"] is True
    assert final["end_of_llm_stream"] is True  # what makes the output handler send the final mark


async def test_old_socket_is_drained_before_the_replacement_is_read():
    # the tail lands and the socket closes; monitor_connection swaps before the reader catches up
    synth = _make_synth()
    old = FakeSocket([_audio("old1")], state=OPEN)
    synth.websocket = old
    _turn_in_flight(synth, old)
    reader = Reader(synth.receiver())
    await reader.after(0.1)

    old.frames.extend([_audio("old2"), _final()])
    old.state = CLOSED
    synth.websocket = FakeSocket([_audio("new", ctx="ctx-next")], state=OPEN)

    items = await reader.after(0.3)
    await reader.stop()
    assert [t for _, t in items] == ["old1", "old2", "", "new"]


async def test_real_socket_burst_then_close_loses_nothing():
    # against websockets itself, so the fake above can't drift from the library
    synth = _make_synth()
    n = 30

    async def handler(ws):
        for _ in range(n):
            await ws.send(_audio())
        await ws.send(_final())
        await ws.close(1008, "policy violation")

    async with websockets.serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        ws = await websockets.connect(f"ws://127.0.0.1:{port}")
        synth.websocket = ws
        _turn_in_flight(synth, ws)
        # a paced reader lets the close land while the reply's tail is still queued
        items = await _receive(synth, 1.5, pace=0.02)

    assert _kinds(items).count("audio") == n
    assert _kinds(items).count("eos") == 1
    assert ws.close_code == 1008


# --- settling: one end-of-stream, only when the turn really was lost ---------------------


async def test_receiver_keeps_running_after_a_drop_and_reads_the_replacement():
    synth = _make_synth()
    dead = FakeSocket([_audio()])
    synth.websocket = dead
    _turn_in_flight(synth, dead)

    reader = Reader(synth.receiver())
    assert _kinds(await reader.after()) == ["audio", "eos"]
    synth.current_turn_start_time = None  # consumer settled the turn
    synth.websocket = FakeSocket([_audio("later", ctx="ctx-next")], state=OPEN)
    assert _kinds(await reader.after()) == ["audio", "eos", "audio"]
    await reader.stop()


async def test_lost_turn_ends_exactly_once_even_if_consumer_never_resets():
    synth = _make_synth()
    dead = FakeSocket()
    synth.websocket = dead
    _turn_in_flight(synth, dead)

    assert _kinds(await _receive(synth)) == ["eos"]


async def test_drop_while_llm_still_streaming_leaves_turn_to_the_new_socket():
    synth = _make_synth()
    dead = FakeSocket([_audio()])
    synth.websocket = dead
    _turn_in_flight(synth, dead, fully_sent=False)

    assert _kinds(await _receive(synth)) == ["audio"]


async def test_drop_after_barge_in_emits_nothing():
    synth = _make_synth()
    dead = FakeSocket([_audio()])
    synth.websocket = dead
    _turn_in_flight(synth, dead)
    synth.current_turn_start_time = None  # handle_interruption clears it

    assert _kinds(await _receive(synth)) == ["audio"]


async def test_idle_drop_emits_nothing():
    synth = _make_synth()
    synth.websocket = FakeSocket()
    synth.current_turn_start_time = None
    synth.last_text_sent = True  # left over from the previous, completed turn

    assert _kinds(await _receive(synth)) == []


async def test_turn_settles_when_fully_sent_only_after_its_socket_drained():
    # whichever side notices last: the socket drains first, then the final push marks the turn sent
    synth = _make_synth()
    dead = FakeSocket()
    synth.websocket = dead
    _turn_in_flight(synth, dead, fully_sent=False)

    reader = Reader(synth.receiver())
    assert _kinds(await reader.after()) == []
    synth.last_text_sent = True
    assert _kinds(await reader.after()) == ["eos"]
    await reader.stop()


async def test_close_code_and_reason_logged_once_per_socket(caplog):
    synth = _make_synth()
    dead = FakeSocket([_audio()], close_code=1011, close_reason="internal error")
    synth.websocket = dead
    _turn_in_flight(synth, dead)

    with caplog.at_level(logging.WARNING):
        await _receive(synth)

    closes = [r.getMessage() for r in caplog.records if "ElevenLabs WebSocket closed" in r.getMessage()]
    assert closes == ["ElevenLabs WebSocket closed code=1011 reason='internal error' trace_id=None"]


# --- sender: a send racing the close is not fatal -----------------------------------------


async def test_final_text_send_on_closing_socket_is_not_fatal_and_turn_settles():
    synth = _make_synth()
    closing = FakeSocket(state=CLOSING, send_raises=True)
    synth.websocket = closing
    synth.context_id = CTX
    _turn_in_flight(synth, closing, fully_sent=False)

    await synth.sender("rest of the reply", sequence_id=4, end_of_llm_stream=True)
    assert synth.connection_error is None
    assert synth.last_text_sent is True
    assert synth.context_id is None

    assert _kinds(await _receive(synth)) == ["eos"]


async def test_mid_stream_text_send_on_closing_socket_keeps_context_for_the_rest():
    synth = _make_synth()
    synth.websocket = FakeSocket(state=CLOSING, send_raises=True)
    synth.context_id = CTX

    await synth.sender("first half", sequence_id=4, end_of_llm_stream=False)
    assert synth.connection_error is None
    assert synth.last_text_sent is False
    assert synth.context_id == CTX  # the rest of the turn continues this context on the replacement


async def test_flush_send_on_closing_socket_is_not_fatal():
    synth = _make_synth()
    synth.websocket = FakeSocket(state=CLOSING, send_raises=True)
    synth.context_id = CTX

    await synth.sender("", sequence_id=4, end_of_llm_stream=True)
    assert synth.connection_error is None
    assert synth.last_text_sent is True
    assert synth.context_id is None


async def test_sender_claims_the_socket_for_text_and_flush():
    synth = _make_synth()
    first = FakeSocket(state=OPEN)
    synth.websocket = first
    synth.context_id = CTX
    await synth.sender("hello there", sequence_id=4)
    assert synth.current_turn_socket is first

    second = FakeSocket(state=OPEN)
    synth.websocket = second  # reconnected before the flush
    await synth.sender("", sequence_id=4, end_of_llm_stream=True)
    assert synth.current_turn_socket is second
    assert any(m.get("flush") for m in second.sent)


# --- interruption on a dead socket --------------------------------------------------------


async def test_interruption_on_dead_socket_does_not_leave_next_turn_on_ignored_context():
    synth = _make_synth()
    synth.websocket = FakeSocket(state=CLOSING, send_raises=True)
    synth.context_id = "ctx-interrupted"

    await synth.handle_interruption()
    assert synth.context_id is None
    assert "ctx-interrupted" in synth.context_ids_to_ignore

    synth._on_push({"sequence_id": 5}, "next reply")
    assert synth.context_id is not None
    assert synth.context_id not in synth.context_ids_to_ignore
