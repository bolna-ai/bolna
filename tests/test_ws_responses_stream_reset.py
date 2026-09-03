"""A cancelled WS Responses turn must not leak its events into the next turn.

One socket serves every turn and `stream_response` reads it without a response_id filter, so an
abandoned stream's leftovers would be read as the next turn's. The guarantee: no response
terminal consumed => socket dirty => it is replaced before reuse.
"""

import asyncio
import json
import time

import pytest
from unittest.mock import patch
from websockets.protocol import State as WSState

from bolna.enums import ResponseStreamEvent
from bolna.llms.openai_llm import OpenAIWSConnection, OpenAiLLM


def delta(text):
    return {"type": ResponseStreamEvent.OUTPUT_TEXT_DELTA.value, "delta": text}


def completed(rid="resp_x"):
    return {"type": ResponseStreamEvent.COMPLETED.value, "response": {"id": rid}}


def created(rid="resp_x"):
    return {"type": ResponseStreamEvent.CREATED.value, "response": {"id": rid}}


class FakeWS:
    """Events queue on send and stay queued until read; a reconnect yields a fresh socket."""

    def __init__(self, script, hang=False, gate=None):
        self.state = WSState.OPEN
        self.sent = []
        self.closed = False
        self._script = script  # shared across reconnects; each send pops one response
        self._queue = []
        self._hang = hang  # block instead of ending, so a turn can be cancelled mid-stream
        self._gate = gate  # withhold events until opened

    async def send(self, raw):
        self.sent.append(json.loads(raw))
        if self._script:
            self._queue.extend(json.dumps(e) for e in self._script.pop(0))

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._gate is not None:
            await self._gate.wait()
        if not self._queue:
            if self._hang:
                await asyncio.Event().wait()
            raise StopAsyncIteration
        return self._queue.pop(0)

    async def close(self):
        self.closed = True
        self.state = WSState.CLOSED


def _transport(*responses, hang=False, gate=None):
    """Transport whose connects are counted; `responses` is one event-list per create sent."""
    t = OpenAIWSConnection.__new__(OpenAIWSConnection)
    t._api_key = "k"
    t._ws = None
    t._needs_reset = False
    t._connected_at = 0.0
    t._lock = asyncio.Lock()
    t._connect_task = None
    t.connects = 0
    t.sockets = []
    script = [list(r) for r in responses]

    async def fake_connect():
        t.connects += 1
        t._ws = FakeWS(script, hang=hang, gate=gate)
        t._connected_at = time.monotonic()
        t.sockets.append(t._ws)

    t._connect = fake_connect
    return t


async def _drain(agen):
    return [e async for e in agen]


async def _settle():
    """Let discard_socket's background close and pre-warm run."""
    for _ in range(4):
        await asyncio.sleep(0)


def _discarded(t, ws):
    """The socket was dropped and a replacement warmed, so no handshake lands on the next turn."""
    return t._ws is not ws and ws.closed and t.connects == 2


# ---------------------------------------------------------------- healthy turns unchanged


async def test_a_completed_turn_leaves_the_socket_clean():
    t = _transport([delta("hi"), completed()])
    events = await _drain(t.stream_response({"input": "a"}))
    assert [e["type"] for e in events] == ["response.output_text.delta", "response.completed"]
    assert t._needs_reset is False
    assert t.connects == 1


async def test_two_healthy_turns_reuse_one_socket():
    t = _transport([completed("r1")], [completed("r2")])
    await _drain(t.stream_response({"input": "a"}))
    await _drain(t.stream_response({"input": "b"}))
    assert t.connects == 1  # no reconnect on the happy path
    assert len(t.sockets) == 1


async def test_a_consumer_that_breaks_on_completed_does_not_dirty_the_socket():
    # Clearing the flag after the yield instead would reconnect on every healthy turn.
    t = _transport([delta("hi"), completed()], [completed("r2")])
    async for evt in t.stream_response({"input": "a"}):
        if evt["type"] == ResponseStreamEvent.COMPLETED.value:
            break
    assert t._needs_reset is False
    await _drain(t.stream_response({"input": "b"}))
    assert t.connects == 1


@pytest.mark.parametrize("terminal", ["response.failed", "response.incomplete", "error"])
async def test_every_terminal_event_clears_the_flag(terminal):
    t = _transport([{"type": terminal}])
    await _drain(t.stream_response({"input": "a"}))
    assert t._needs_reset is False


# ---------------------------------------------------------------- abandoned turns reconnect


async def test_an_abandoned_stream_discards_its_socket():
    t = _transport([delta("one"), delta("two"), completed()], [completed("r2")])
    agen = t.stream_response({"input": "a"})
    await agen.__anext__()  # consume one delta, then walk away
    await agen.aclose()
    await _settle()
    assert _discarded(t, t.sockets[0])


async def test_the_next_turn_after_an_abandoned_stream_reconnects():
    t = _transport([delta("leftover"), completed("r1")], [delta("real"), completed("r2")])
    agen = t.stream_response({"input": "a"})
    await agen.__anext__()
    await agen.aclose()

    events = await _drain(t.stream_response({"input": "b"}))
    assert t.connects == 2
    assert t.sockets[0].closed is True
    assert t._needs_reset is False


async def test_leftover_events_can_never_reach_the_next_turn():
    """The regression: turn 1 is cancelled with its answer still queued, turn 2 must not read it."""
    t = _transport(
        [created("r1"), delta("कुछ और पूछना था?"), completed("r1")],
        [created("r2"), delta("loan answer"), completed("r2")],
    )
    agen = t.stream_response({"input": "ठीक है"})
    assert (await agen.__anext__())["type"] == "response.created"
    await agen.aclose()  # cancelled here; the answer is still on the socket

    events = await _drain(t.stream_response({"input": "loan question"}))
    texts = [e["delta"] for e in events if "delta" in e]
    assert texts == ["loan answer"]
    assert "कुछ और पूछना था?" not in texts
    assert [e["response"]["id"] for e in events if "response" in e] == ["r2", "r2"]


async def test_a_generator_that_never_ran_sends_nothing_and_stays_clean():
    # Nothing sent means nothing queued; reconnecting here would be pure cost.
    t = _transport([completed("r1")])
    agen = t.stream_response({"input": "a"})
    await agen.aclose()  # never iterated, so the body never ran
    assert t._needs_reset is False
    assert t.connects == 0


async def test_task_cancellation_while_awaiting_the_first_event_discards_the_socket():
    """Cancelled milliseconds after the create, before any event arrived."""
    t = _transport([], [completed("r2")], hang=True)

    async def consume():
        async for _ in t.stream_response({"input": "ठीक है"}):
            pass

    task = asyncio.create_task(consume())
    while t._ws is None or not t._ws.sent:  # let the create go out
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    await _settle()
    assert _discarded(t, t.sockets[0])
    assert t._lock.locked() is False  # cancellation lands inside the generator, freeing it


async def test_a_turn_cancelled_mid_stream_discards_the_socket():
    # Partial output delivered, terminal event still to come — the leak window.
    t = _transport([delta("one"), delta("two")], [completed("r2")], hang=True)
    started = asyncio.Event()

    async def consume():
        async for _ in t.stream_response({"input": "a"}):
            started.set()

    task = asyncio.create_task(consume())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    await _settle()
    assert _discarded(t, t.sockets[0])


async def test_the_dirty_flag_is_cleared_by_the_reconnect():
    t = _transport([delta("x"), completed()], [completed("r2")], [completed("r3")])
    agen = t.stream_response({"input": "a"})
    await agen.__anext__()
    await agen.aclose()

    await _drain(t.stream_response({"input": "b"}))
    assert t.connects == 2
    await _drain(t.stream_response({"input": "c"}))
    assert t.connects == 2  # clean again, no further reconnect


# ---------------------------------------------------- end to end through the real consumer


def _make_ws_llm(transport):
    """Real OpenAiLLM driving the real WS consumer over a faked socket."""
    with patch.object(OpenAiLLM, "__init__", lambda self, **kw: None):
        llm = OpenAiLLM.__new__(OpenAiLLM)
    llm.model = "gpt-5.6-luna"
    llm.max_tokens = 100
    llm.buffer_size = 40
    llm.temperature = 0.1
    llm.run_id = "run_123"
    llm.language = "en"
    llm.trigger_function_call = False
    llm.api_params = {}
    llm.tools = []
    llm.started_streaming = False
    llm.llm_host = None
    llm.use_responses_api = True
    llm.previous_response_id = None
    llm._pending_call_ids = set()
    llm.compact_threshold = None
    llm._interruption_hint = None
    llm._ws_transport = transport
    llm.model_args = {"model": llm.model, "max_tokens": 100, "temperature": 0.1}
    return llm


MESSAGES = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]


def _text(chunks):
    return "".join(c.data for c in chunks if getattr(c, "data", None))


async def test_real_consumer_completes_a_turn_without_dirtying_the_socket():
    """The ordering guarantee against the actual consumer, which breaks on COMPLETED."""
    t = _transport(
        [created("r1"), delta("नमस्ते"), completed("r1")],
        [created("r2"), delta("दूसरा"), completed("r2")],
    )
    llm = _make_ws_llm(t)

    chunks = [c async for c in llm._generate_stream_ws_responses(MESSAGES, meta_info={"sequence_id": 1})]
    assert "नमस्ते" in _text(chunks)
    assert t._needs_reset is False

    chunks = [c async for c in llm._generate_stream_ws_responses(MESSAGES, meta_info={"sequence_id": 2})]
    assert "दूसरा" in _text(chunks)
    assert t.connects == 1  # one socket served both healthy turns


async def test_real_consumer_does_not_inherit_a_cancelled_turns_answer():
    """End to end, through the real consumer."""
    gate = asyncio.Event()  # the server answers turn 1 only after it has been cancelled
    t = _transport(
        [created("r1"), delta("कुछ और पूछना था?"), completed("r1")],
        [created("r2"), delta("loan answer"), completed("r2")],
        gate=gate,
    )
    llm = _make_ws_llm(t)

    async def turn_one():
        async for _ in llm._generate_stream_ws_responses(MESSAGES, meta_info={"sequence_id": 4}):
            pass

    task = asyncio.create_task(turn_one())
    while t._ws is None or not t._ws.sent:
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    gate.set()  # turn 1's answer lands on the socket now, with nobody reading it

    chunks = [c async for c in llm._generate_stream_ws_responses(MESSAGES, meta_info={"sequence_id": 5})]
    text = _text(chunks)
    assert "loan answer" in text
    assert "कुछ और पूछना था?" not in text
    assert t.connects == 2


# ---------------------------------------------------------------- review: cancel during send


async def test_a_send_cancelled_mid_frame_still_discards_the_socket():
    """The frame may already be on the wire, and a cancelled send leaves the connection
    inconsistent either way. Marking after the send would miss both."""
    t = _transport([created("r1"), delta("leftover"), completed("r1")], [completed("r2")])
    await t.ensure_connected()
    ws = t._ws
    delivered = asyncio.Event()
    real_send = ws.send

    async def slow_send(raw):
        await real_send(raw)  # frame delivered...
        delivered.set()
        await asyncio.Event().wait()  # ...then the send suspends and never returns

    ws.send = slow_send

    async def consume():
        async for _ in t.stream_response({"input": "a"}):
            pass

    task = asyncio.create_task(consume())
    await delivered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    await _settle()
    assert _discarded(t, ws)


# ---------------------------------------------------------------- review: error is not a settle


async def test_an_error_event_does_not_mark_the_socket_clean():
    """`error` can be raised against the session and still be followed by the response's own
    terminal event. Treating it as a settle would leave that event on a socket called clean."""
    t = _transport([{"type": "error", "error": {"code": "x"}}, completed("r1")], [completed("r2")])
    events = await _drain(t.stream_response({"input": "a"}))
    assert [e["type"] for e in events] == ["error"]
    await _settle()
    assert _discarded(t, t.sockets[0])


async def test_a_terminal_trailing_an_error_cannot_reach_the_next_turn():
    t = _transport(
        [{"type": "error", "error": {"code": "x"}}, delta("stale"), completed("r1")],
        [created("r2"), delta("fresh"), completed("r2")],
    )
    await _drain(t.stream_response({"input": "a"}))
    await _settle()

    events = await _drain(t.stream_response({"input": "b"}))
    assert [e["delta"] for e in events if "delta" in e] == ["fresh"]


# ---------------------------------------------------------------- review: no handshake on the turn


async def test_the_replacement_socket_is_warm_before_the_next_turn():
    """discard_socket pre-warms, so the next turn finds a connected socket instead of paying
    a close handshake plus TLS inside the lock."""
    t = _transport([delta("one")], [completed("r2")], hang=True)

    async def consume():
        async for _ in t.stream_response({"input": "a"}):
            pass

    task = asyncio.create_task(consume())
    while t._ws is None or not t._ws.sent:
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await _settle()

    assert t.connects == 2  # already reconnected, before the next turn asked
    connects_before = t.connects
    await _drain(t.stream_response({"input": "b"}))
    assert t.connects == connects_before  # the turn itself opened nothing


async def test_the_old_socket_close_is_never_awaited_on_the_critical_path():
    t = _transport([delta("one"), completed("r1")], [completed("r2")])
    agen = t.stream_response({"input": "a"})
    await agen.__anext__()
    await agen.aclose()
    # discard_socket hands the close to the loop; it has not run yet.
    assert t.sockets[0].closed is False
    await _settle()
    assert t.sockets[0].closed is True
