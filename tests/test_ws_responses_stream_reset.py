"""A cancelled WS Responses turn must not leak its events into the next turn.

One WebSocket serves every turn of a call, and `stream_response` yields whatever it reads
with no response_id filter. When `llm_task.cancel()` abandons the generator mid-stream, the
cancelled response's remaining events stay queued on the socket — the next turn then reads
them and answers the PREVIOUS utterance, in ~0ms (run 23323c9e: seq4 cancelled, seq5 7.81ms,
seq6 0.52ms, agent replied to 'ठीक है' instead of the question that followed).

The guarantee: a stream that does not consume a terminal event marks the socket dirty, and
the next use reconnects. Consumers `break` on COMPLETED/INCOMPLETE and never let the
generator resume, so the flag must be cleared BEFORE the terminal event is yielded —
otherwise every healthy turn would reconnect too.
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
    """Models the real socket: a response's events are queued when its create is sent, and
    whatever nobody consumes stays queued on THIS socket. A reconnect yields a fresh one."""

    def __init__(self, script, hang=False, gate=None):
        self.state = WSState.OPEN
        self.sent = []
        self.closed = False
        self._script = script  # shared across reconnects; each send pops one response
        self._queue = []
        self._hang = hang  # block instead of ending, to model "server has not answered yet"
        self._gate = gate  # when set, events are withheld until the gate opens

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
    # The real consumer breaks on COMPLETED and never lets the generator resume, so the
    # flag has to be cleared before the yield. Clearing it after would reconnect every turn.
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


async def test_a_cancelled_stream_marks_the_socket_dirty():
    t = _transport([delta("कुछ"), delta(" और"), completed()])
    agen = t.stream_response({"input": "ठीक है"})
    await agen.__anext__()  # consume one delta, then walk away
    await agen.aclose()
    assert t._needs_reset is True


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
    """The regression itself. Turn 1 ('ठीक है') is cancelled with its answer still queued;
    turn 2 (the loan question) must see only its own events. Unfixed, turn 2 reads turn 1's
    answer and its terminal event, i.e. it replies to the previous utterance in ~0ms."""
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
    # Nothing was sent, so nothing is queued — reconnecting here would be pure cost.
    t = _transport([completed("r1")])
    agen = t.stream_response({"input": "a"})
    await agen.aclose()  # never iterated, so the body never ran
    assert t._needs_reset is False
    assert t.connects == 0


async def test_task_cancellation_while_awaiting_the_first_event_dirties_the_socket():
    """The reported call exactly: llm_task.cancel() 6ms after the create was sent, while the
    turn is still awaiting its first event. The response is already committed server-side."""
    t = _transport([], hang=True)

    async def consume():
        async for _ in t.stream_response({"input": "ठीक है"}):
            pass

    task = asyncio.create_task(consume())
    while t._ws is None or not t._ws.sent:  # let the create go out
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert t._needs_reset is True
    assert t._lock.locked() is False  # cancellation lands inside the generator, freeing it


async def test_a_turn_cancelled_mid_stream_dirties_the_socket():
    # Partial output delivered, terminal event still to come — the leak window.
    t = _transport([delta("one"), delta("two")], hang=True)
    started = asyncio.Event()

    async def consume():
        async for _ in t.stream_response({"input": "a"}):
            started.set()

    task = asyncio.create_task(consume())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert t._needs_reset is True


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
    """The ordering guarantee against the actual consumer, which breaks on COMPLETED and
    never lets the generator resume. If the flag were cleared after the yield instead of
    before, every healthy turn would reconnect."""
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
    """End to end reproduction of run 23323c9e: turn 1 is cancelled mid-flight, turn 2 must
    still generate its own answer rather than replay turn 1's."""
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
