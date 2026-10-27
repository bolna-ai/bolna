"""A sender waiting for its TTS socket wakes on the connect itself, not on the next one-second poll, and a
multilingual leg can close its socket and connect again later.

Lazy language legs connect a voice only when the call needs it, so the first sentence after a switch can reach
_wait_for_ws before the socket is up; the old poll could add up to a second there."""

import asyncio
import time

from websockets.protocol import State

# Importing the registry loads every provider module, so the subclass walk below sees them all.
from bolna.providers import SUPPORTED_SYNTHESIZER_MODELS
from bolna.synthesizer.elevenlabs_synthesizer import ElevenlabsV3Synthesizer
from bolna.synthesizer.stream_synthesizer import StreamSynthesizer


class FakeSocket:
    def __init__(self):
        self.state = State.OPEN
        self.closed = False

    async def close(self):
        self.closed = True
        self.state = State.CLOSED


class SlowConnectSynth(StreamSynthesizer):
    def __init__(self, connect_delay_s):
        super().__init__(provider_name="fake")
        self.connect_delay_s = connect_delay_s
        self.connects = 0

    async def establish_connection(self):
        await asyncio.sleep(self.connect_delay_s)
        self.connects += 1
        return FakeSocket()


async def stop(task):
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


async def test_waiting_sender_wakes_on_the_connect_not_the_poll():
    synth = SlowConnectSynth(connect_delay_s=0.05)
    monitor = asyncio.create_task(synth.monitor_connection())
    try:
        started = time.perf_counter()
        await synth._wait_for_ws()  # default one-second interval
        waited = time.perf_counter() - started
        assert synth._is_ws_connected()
        assert waited < 0.5
    finally:
        await stop(monitor)


async def test_wait_still_gives_up_on_a_connection_error():
    synth = SlowConnectSynth(connect_delay_s=10)

    async def fail_soon():
        await asyncio.sleep(0.05)
        synth.connection_error = "Max connection failures reached"

    failing = asyncio.create_task(fail_soon())
    await asyncio.wait_for(synth._wait_for_ws(poll_interval=0.1), timeout=1)
    await failing
    assert not synth._is_ws_connected()


async def test_wait_still_gives_up_when_the_call_ends():
    synth = SlowConnectSynth(connect_delay_s=10)

    async def end_soon():
        await asyncio.sleep(0.05)
        synth.conversation_ended = True

    ending = asyncio.create_task(end_soon())
    await asyncio.wait_for(synth._wait_for_ws(poll_interval=0.1), timeout=1)
    await ending


async def test_close_connection_keeps_the_synth_reusable():
    synth = SlowConnectSynth(connect_delay_s=0)
    monitor = asyncio.create_task(synth.monitor_connection())
    await asyncio.wait_for(synth._wait_for_ws(poll_interval=0.1), timeout=1)
    first = synth.websocket
    await stop(monitor)

    await synth.close_connection()
    assert first.closed
    assert synth.websocket is None
    assert synth.conversation_ended is False  # cleanup() would set this and block every later sender

    monitor = asyncio.create_task(synth.monitor_connection())
    try:
        await asyncio.wait_for(synth._wait_for_ws(poll_interval=0.1), timeout=1)
        assert synth.websocket is not first
        assert synth.connects == 2
    finally:
        await stop(monitor)


async def test_close_connection_without_a_socket_is_a_noop():
    synth = SlowConnectSynth(connect_delay_s=0)
    await synth.close_connection()
    assert synth.websocket is None


def all_stream_synthesizers():
    found, pending = set(), [StreamSynthesizer]
    while pending:
        for subclass in pending.pop().__subclasses__():
            if subclass not in found:
                found.add(subclass)
                pending.append(subclass)
    return found


def test_every_streaming_provider_signals_its_connect():
    """A provider with its own connect loop must set websocket_ready itself, or its senders fall back to polling."""
    providers = {cls for cls in all_stream_synthesizers() if cls.__module__.startswith("bolna.synthesizer.")}
    assert providers, "no streaming providers found"
    own_loops = {cls for cls in providers if cls.monitor_connection is not StreamSynthesizer.monitor_connection}
    assert own_loops == {ElevenlabsV3Synthesizer}


async def test_elevenlabs_v3_connect_sets_websocket_ready():
    v3 = SUPPORTED_SYNTHESIZER_MODELS["elevenlabs"](
        model="eleven_v3_conversational",
        voice="George",
        voice_id="JBFqnCBsd6RMkjVDRZzb",
        sampling_rate="8000",
        use_mulaw=True,
        caching=False,
        synthesizer_key="test-key",
    )
    assert isinstance(v3, ElevenlabsV3Synthesizer)

    async def connect():
        return FakeSocket()

    v3.establish_connection = connect
    assert not v3.websocket_ready.is_set()
    assert await v3._ensure_connection() is True
    assert v3.websocket_ready.is_set()


async def test_elevenlabs_v3_close_stops_the_redial_and_keep_alive():
    # A barge-in just before the switch starts a redial; left running it would reopen the closed voice.
    v3 = SUPPORTED_SYNTHESIZER_MODELS["elevenlabs"](
        model="eleven_v3_conversational",
        voice="George",
        voice_id="JBFqnCBsd6RMkjVDRZzb",
        sampling_rate="8000",
        use_mulaw=True,
        caching=False,
        synthesizer_key="test-key",
    )
    redial = asyncio.create_task(asyncio.sleep(3600))
    keep_alive = asyncio.create_task(asyncio.sleep(3600))
    v3._reconnect_task = redial
    v3._keep_alive_task = keep_alive
    socket = FakeSocket()
    v3.websocket = socket

    await v3.close_connection()
    await asyncio.gather(redial, keep_alive, return_exceptions=True)
    assert redial.cancelled() and keep_alive.cancelled()
    assert socket.closed and v3.websocket is None
    assert v3.conversation_ended is False
