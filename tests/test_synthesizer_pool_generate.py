"""SynthesizerPool keeps a multilingual agent's audio path alive.

The pool forwards the active synth's generate() through one task. Most receivers return when their
websocket closes; if the pool doesn't re-enter generate() then, nothing reads the redialled socket
and the agent is mute for the rest of the call.
"""

import asyncio
import base64
import json
from unittest.mock import MagicMock

import pytest
import websockets
import websockets.protocol
from websockets.frames import Close

from bolna.synthesizer.cartesia_synthesizer import CartesiaSynthesizer
from bolna.synthesizer import synthesizer_pool
from bolna.synthesizer.synthesizer_pool import SynthesizerPool


@pytest.fixture(autouse=True)
def _no_reentry_delay(monkeypatch):
    monkeypatch.setattr(synthesizer_pool, "SYNTHESIZER_POOL_REENTRY_DELAY_S", 0)


class _ScriptedSynth:
    """Each generate() call plays the next script: a list of packets, then returns."""

    def __init__(self, *scripts):
        self.scripts = list(scripts)
        self.generate_calls = 0
        self.conversation_ended = False
        self.connection_time = 0

    async def generate(self):
        self.generate_calls += 1
        if not self.scripts:
            await asyncio.Event().wait()  # idle like a live receiver waiting on its socket
        for data in self.scripts.pop(0):
            yield {"data": data, "meta_info": {}}


async def _collect(pool, count, timeout=1.0):
    out = []

    async def read():
        async for packet in pool.generate():
            out.append(packet["data"])
            if len(out) == count:
                return

    await asyncio.wait_for(read(), timeout)
    return out


async def test_a_generate_that_returns_is_re_entered():
    synth = _ScriptedSynth([b"turn-1"], [b"turn-2"])
    pool = SynthesizerPool({"en": synth, "hi": _ScriptedSynth()}, "en", {})
    pool._gen_task = asyncio.create_task(pool._run_generate("en"))

    assert await _collect(pool, 2) == [b"turn-1", b"turn-2"]
    assert not pool._gen_task.done()

    pool._gen_task.cancel()


async def test_an_ended_synth_is_not_re_entered():
    synth = _ScriptedSynth([b"last"])
    pool = SynthesizerPool({"en": synth}, "en", {})

    async def end_after_first_turn():
        async for _ in pool.generate():
            synth.conversation_ended = True
            return

    pool._gen_task = asyncio.create_task(pool._run_generate("en"))
    await asyncio.wait_for(end_after_first_turn(), 1.0)
    await asyncio.wait_for(pool._gen_task, 1.0)

    assert synth.generate_calls == 1


async def test_a_switch_stops_forwarding_from_the_old_synth():
    en = _ScriptedSynth([b"en-1"], [b"en-2"], [b"en-3"])
    hi = _ScriptedSynth([b"hi-1"])
    pool = SynthesizerPool({"en": en, "hi": hi}, "en", {})
    old_task = pool._gen_task = asyncio.create_task(pool._run_generate("en"))
    await asyncio.sleep(0)

    await pool.switch("hi")

    assert old_task.done()
    calls_at_switch = en.generate_calls
    received = []
    async for packet in pool.generate():  # drains up to the switch sentinel
        received.append(packet["data"])
    received += await _collect(pool, 1)
    assert received[-1] == b"hi-1"
    assert en.generate_calls == calls_at_switch

    pool._gen_task.cancel()


# ── the real receiver that triggered this: Cartesia breaks out on a socket close ──


class _FakeWS:
    def __init__(self):
        self.sent = []
        self.inbox = asyncio.Queue()
        self.closed = False

    async def send(self, message):
        self.sent.append(json.loads(message))

    async def recv(self):
        item = await self.inbox.get()
        if item is None:
            self.closed = True
            raise websockets.exceptions.ConnectionClosedError(Close(1006, "abnormal"), None)
        return item

    async def close(self):
        self.closed = True

    @property
    def state(self):
        return websockets.protocol.State.CLOSED if self.closed else websockets.protocol.State.OPEN


def _cartesia():
    synth = CartesiaSynthesizer(
        voice_id="voice", voice="voice", task_manager_instance=MagicMock(), synthesizer_key="key", caching=False
    )
    synth.task_manager_instance.is_sequence_id_in_current_ids.return_value = True
    return synth


async def _speak(synth, text, sequence_id, audio):
    await synth.push(
        {"data": text, "meta_info": {"sequence_id": sequence_id, "turn_id": sequence_id, "end_of_llm_stream": True}}
    )
    await synth.sender_task
    await synth.websocket.inbox.put(
        json.dumps({"context_id": synth.context_id, "data": base64.b64encode(audio).decode()})
    )
    await synth.websocket.inbox.put(json.dumps({"context_id": synth.context_id, "done": True}))


async def test_audio_flows_after_the_active_synth_socket_is_redialled():
    en = _cartesia()
    pool = SynthesizerPool({"en": en, "hi": _cartesia()}, "en", {})
    en.websocket = _FakeWS()
    pool._gen_task = asyncio.create_task(pool._run_generate("en"))

    await _speak(en, "Hello!", 1, b"T1")
    assert await _collect(pool, 2) == [b"T1", b"\x00"]

    await en.websocket.inbox.put(None)  # the socket closes between turns
    await asyncio.sleep(0.05)
    en.websocket = _FakeWS()  # monitor_connection's redial

    await _speak(en, "How can I help?", 2, b"T2")
    assert await _collect(pool, 2) == [b"T2", b"\x00"]

    pool._gen_task.cancel()
