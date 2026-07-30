"""Reconnect budget for a failed provider WebSocket: the caller hears silence while we
retry, so an instant refusal gets more attempts and a hung connect stops sooner."""

import asyncio
import time

from websockets.protocol import State

from bolna.synthesizer import stream_synthesizer as ss


class FakeWS:
    def __init__(self):
        self.state = State.OPEN

    def drop(self):
        self.state = State.CLOSED


class ScriptedSynthesizer(ss.StreamSynthesizer):
    """StreamSynthesizer whose establish_connection replays a script of
    (seconds_to_take, succeeds) so each prod failure mode can be reproduced."""

    def __init__(self, script):
        self.provider_name = "fake"
        self.websocket = None
        self.conversation_ended = False
        self.connection_error = None
        self.script = list(script)
        self.attempts = []

    async def establish_connection(self):
        delay, succeeds = self.script.pop(0) if self.script else (0.0, False)
        self.attempts.append(delay)
        await asyncio.sleep(delay)
        return FakeWS() if succeeds else None


def _run_until_gives_up(synth, timeout=60):
    async def main():
        start = time.perf_counter()
        await asyncio.wait_for(synth.monitor_connection(), timeout=timeout)
        return time.perf_counter() - start

    return asyncio.run(main())


def test_instant_refusal_gets_the_full_attempt_budget():
    synth = ScriptedSynthesizer([(0.3, False)] * 20)
    elapsed = _run_until_gives_up(synth)

    assert len(synth.attempts) == ss.MAX_CONNECTION_FAILURES
    assert elapsed < ss.CONNECT_RETRY_BUDGET_SECONDS
    assert synth.connection_error == "Max connection failures reached"


def test_hung_connect_does_not_overshoot_the_budget():
    """A provider that hangs each connect must not spend one whole timeout past the
    budget before noticing."""
    synth = ScriptedSynthesizer([(10.0, False)] * 20)
    elapsed = _run_until_gives_up(synth)

    assert len(synth.attempts) < ss.MAX_CONNECTION_FAILURES
    assert elapsed < ss.CONNECT_RETRY_BUDGET_SECONDS + 1
    assert synth.connection_error == "Max connection failures reached"


def test_a_blip_that_clears_late_no_longer_kills_the_call():
    synth = ScriptedSynthesizer([(0.2, False)] * 3 + [(0.2, True)])

    async def main():
        task = asyncio.create_task(synth.monitor_connection())
        while synth.websocket is None and not synth.connection_error:
            await asyncio.sleep(0.05)
        synth.conversation_ended = True
        await asyncio.wait_for(task, timeout=5)

    asyncio.run(main())

    assert synth.connection_error is None
    assert isinstance(synth.websocket, FakeWS)
    assert len(synth.attempts) == 4


def test_mid_call_reconnect_starts_from_a_clean_budget():
    synth = ScriptedSynthesizer([(0.1, False), (0.1, False), (0.1, True)])

    async def main():
        task = asyncio.create_task(synth.monitor_connection())
        while synth.websocket is None:
            await asyncio.sleep(0.05)
        synth.websocket.drop()
        synth.script = [(0.1, False)] * 20
        await asyncio.wait_for(task, timeout=30)

    asyncio.run(main())

    assert len(synth.attempts) == 3 + ss.MAX_CONNECTION_FAILURES


def test_loop_stops_once_the_conversation_ends():
    synth = ScriptedSynthesizer([(0.1, False)] * 100)
    synth.conversation_ended = True

    assert _run_until_gives_up(synth, timeout=5) < 1
    assert synth.attempts == []
    assert synth.connection_error is None


def test_backoff_is_jittered_and_capped():
    first = [ss._connect_backoff(1) for _ in range(200)]
    assert len(set(first)) > 100, "backoff must be jittered so calls don't retry in lockstep"

    for attempt in range(1, 12):
        assert 0 < ss._connect_backoff(attempt) <= ss.CONNECT_BACKOFF_MAX_SECONDS

    assert min(ss._connect_backoff(6) for _ in range(50)) > max(first)
