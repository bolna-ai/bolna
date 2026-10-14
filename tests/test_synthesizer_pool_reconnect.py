"""Regression: SynthesizerPool must keep reading the active synthesizer after its
websocket closes and monitor_connection() reconnects.

_run_generate used to iterate generate() once, so the forwarder ended with the socket.
Text for the next turn was still sent (and billed) but nothing read the new socket, so
the active leg was silent for the rest of the call.
"""

import asyncio

from bolna.synthesizer import synthesizer_pool as pool_mod
from bolna.synthesizer.synthesizer_pool import SynthesizerPool


class _ReconnectingSynth:
    """generate() returns when the socket drops, the way most provider receivers do.

    rounds[i] is how many audio chunks the i-th generate() call yields before the
    socket closes under it.
    """

    def __init__(self, rounds):
        self._rounds = list(rounds)
        self.generate_calls = 0
        self.connection_time = 0

    async def generate(self):
        self.generate_calls += 1
        chunks = self._rounds.pop(0) if self._rounds else 0
        for _ in range(chunks):
            await asyncio.sleep(0)
            yield {"meta_info": {}, "data": b"x"}
        # socket closed: the receiver returns rather than raising


class _RaisingThenRecoveringSynth:
    def __init__(self):
        self.generate_calls = 0
        self.connection_time = 0

    async def generate(self):
        self.generate_calls += 1
        if self.generate_calls == 1:
            raise ConnectionResetError("websocket died mid-reply")
        yield {"meta_info": {}, "data": b"recovered"}


async def test_forwarder_reenters_generate_after_socket_close(monkeypatch):
    monkeypatch.setattr(pool_mod, "_REENTER_DELAY", 0)
    synth = _ReconnectingSynth([1, 1])
    pool = SynthesizerPool({"en": synth}, "en", {})

    task = asyncio.create_task(pool._run_generate("en"))
    try:
        first = await asyncio.wait_for(pool._output_queue.get(), 1)
        # Audio from the socket that replaced the one that closed.
        second = await asyncio.wait_for(pool._output_queue.get(), 1)
    finally:
        task.cancel()

    assert first["data"] == b"x"
    assert second["data"] == b"x"
    assert synth.generate_calls >= 2


async def test_forwarder_survives_an_exception_from_generate(monkeypatch):
    monkeypatch.setattr(pool_mod, "_REENTER_DELAY", 0)
    synth = _RaisingThenRecoveringSynth()
    pool = SynthesizerPool({"en": synth}, "en", {})

    task = asyncio.create_task(pool._run_generate("en"))
    try:
        message = await asyncio.wait_for(pool._output_queue.get(), 1)
    finally:
        task.cancel()

    assert message["data"] == b"recovered"


async def test_forwarder_gives_up_instead_of_looping_forever(monkeypatch):
    """A provider refusing every connection must not keep the forwarder spinning."""
    monkeypatch.setattr(pool_mod, "_REENTER_DELAY", 0)
    monkeypatch.setattr(pool_mod, "_MAX_EMPTY_ROUNDS", 3)
    synth = _ReconnectingSynth([])
    pool = SynthesizerPool({"en": synth}, "en", {})

    await asyncio.wait_for(pool._run_generate("en"), 1)

    assert synth.generate_calls == 3


async def test_forwarder_stops_when_the_pool_switches_away(monkeypatch):
    monkeypatch.setattr(pool_mod, "_REENTER_DELAY", 0)
    synth = _ReconnectingSynth([1])
    pool = SynthesizerPool({"en": synth, "hi": _ReconnectingSynth([1])}, "en", {})

    pool.active_label = "hi"
    await asyncio.wait_for(pool._run_generate("en"), 1)

    assert synth.generate_calls == 0
