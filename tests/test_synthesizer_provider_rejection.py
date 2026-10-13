"""A TTS provider rejecting the call ends it with the provider's reason; a passing failure keeps it going."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import websockets.protocol
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosedOK, InvalidStatus
from websockets.frames import Close
from websockets.http11 import Response

from bolna.exceptions import SynthesizerError
from bolna.synthesizer import deepgram_synthesizer, stream_synthesizer
from bolna.synthesizer.base_synthesizer import raise_if_rejected
from bolna.synthesizer.cartesia_synthesizer import CartesiaSynthesizer
from bolna.synthesizer.deepgram_synthesizer import DeepgramSynthesizer
from bolna.synthesizer.sarvam_synthesizer import SarvamSynthesizer
from bolna.synthesizer.soniox_synthesizer import SonioxSynthesizer
from bolna.synthesizer.stream_synthesizer import StreamSynthesizer
from bolna.synthesizer.synthesizer_pool import SynthesizerPool

INVALID_KEY = Close(1008, "API key is invalid.")
OPEN = "open"


@pytest.fixture
def instant_sleep(monkeypatch):
    real_sleep = asyncio.sleep

    async def no_wait(_):
        await real_sleep(0)

    monkeypatch.setattr(stream_synthesizer.asyncio, "sleep", no_wait)


class Socket:
    """A socket already closed with `close` (None: no close frame), or still open."""

    def __init__(self, close):
        self.protocol = SimpleNamespace(close_rcvd=None if close is OPEN else close)
        self.state = websockets.protocol.State.OPEN if close is OPEN else websockets.protocol.State.CLOSED


class Redialling(StreamSynthesizer):
    """Each dial returns the next socket in `closes`."""

    def __init__(self, closes):
        super().__init__(provider_name="elevenlabs")
        self.closes = list(closes)

    async def establish_connection(self):
        return Socket(self.closes.pop(0)) if self.closes else None


async def test_a_provider_closing_every_socket_with_an_error_ends_the_call(instant_sleep):
    synth = Redialling([INVALID_KEY] * 5)

    await asyncio.wait_for(synth.monitor_connection(), timeout=1)

    assert synth.connection_error == "elevenlabs closed the connection (1008): API key is invalid."
    assert len(synth.closes) == 2, "the monitor kept redialling after giving up"


async def test_a_passing_error_close_between_healthy_sockets_keeps_the_call(instant_sleep):
    synth = Redialling([Close(1011, "internal error"), None, Close(1011, "internal error"), Close(1000, ""), OPEN])

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(synth.monitor_connection(), timeout=0.05)

    assert synth.connection_error is None


class FailingSynth:
    connection_time = 0

    async def generate(self):
        raise SynthesizerError("Cartesia rejected the request (402): out of credits")
        yield  # pragma: no cover


async def test_a_pool_passes_the_active_synthesizers_failure_to_the_call():
    pool = SynthesizerPool({"en": FailingSynth()}, "en", {})
    pool._gen_task = asyncio.create_task(pool._run_generate("en"))

    with pytest.raises(SynthesizerError, match="out of credits"):
        await asyncio.wait_for(anext(pool.generate()), timeout=1)


@pytest.mark.parametrize("status", [401, 402, 403, 404])
def test_a_rejected_request_ends_the_call(status):
    with pytest.raises(SynthesizerError, match=f"Gemini rejected the request \\({status}\\): denied"):
        raise_if_rejected("Gemini", status, "denied")


@pytest.mark.parametrize("status", [400, 429, 500, 503])
def test_a_request_that_may_pass_next_turn_does_not_end_the_call(status):
    raise_if_rejected("Gemini", status, "busy")


class Frames:
    """Serves `frames` to one synthesizer, then ends the call."""

    def __init__(self, synth, frames):
        self.synth, self.frames = synth, list(frames)
        self.state = websockets.protocol.State.OPEN

    async def recv(self):
        if self.frames:
            return json.dumps(self.frames.pop(0))
        self.synth.conversation_ended = True
        raise ConnectionClosedOK(Close(1000, ""), None)


async def received(synth, frames):
    synth.websocket = Frames(synth, frames)

    async def collect():
        return [chunk async for chunk in synth.receiver()]

    return await asyncio.wait_for(collect(), timeout=1)


def sarvam():
    return SarvamSynthesizer(
        voice_id="shubh",
        model="bulbul:v3",
        language="hi-IN",
        stream=True,
        synthesizer_key="test-key",
        task_manager_instance=MagicMock(),
    )


def sarvam_error(code, message):
    return {"type": "error", "data": {"message": message, "code": code}}


async def test_sarvam_closing_an_idle_socket_keeps_the_call():
    synth = sarvam()

    chunks = await received(synth, [sarvam_error(408, "Websocket was left open without any messages for too long.")])

    assert chunks == []
    assert synth.connection_error is None


@pytest.mark.parametrize("code", [400, 422])
async def test_sarvam_refusing_one_sentence_ends_only_that_turn(code):
    synth = sarvam()

    chunks = await received(
        synth, [sarvam_error(code, "Text must contain at least one character from the allowed languages.")]
    )

    assert chunks == [b"\x00"]
    assert synth.connection_error is None


async def test_sarvam_out_of_credits_ends_the_call():
    synth = sarvam()
    audio_after = {"type": "audio", "data": {"audio": "AAAA"}}

    chunks = await received(synth, [sarvam_error(402, "Credits exhausted."), audio_after])

    assert chunks == []
    assert "Credits exhausted." in synth.connection_error


async def test_soniox_rejecting_the_key_ends_the_call():
    synth = SonioxSynthesizer(
        voice="Adrian", synthesizer_key="test-key", stream=True, task_manager_instance=MagicMock()
    )
    rejection = {
        "stream_id": "s1",
        "error_type": "unauthenticated",
        "error_code": 401,
        "error_message": "Invalid API key",
    }

    chunks = await received(synth, [rejection])

    assert chunks == []
    assert synth.connection_error == "Soniox rejected the request (401): Invalid API key"


async def test_cartesia_out_of_credits_ends_the_call():
    synth = CartesiaSynthesizer(
        voice_id="v", voice="v", synthesizer_key="test-key", stream=True, task_manager_instance=MagicMock()
    )
    rejection = {"type": "error", "status_code": 402, "done": True, "error": "Insufficient credits"}

    chunks = await received(synth, [rejection])

    assert chunks == []
    assert synth.connection_error == "Cartesia rejected the request (402): Insufficient credits"


@pytest.mark.parametrize("status, fatal", [(401, True), (402, True), (503, False)])
async def test_deepgram_rejecting_the_handshake_ends_the_call(monkeypatch, status, fatal):
    synth = DeepgramSynthesizer(voice_id="aura-2-thalia-en", voice="Thalia", synthesizer_key="test-key", stream=True)

    async def reject(*args, **kwargs):
        raise InvalidStatus(Response(status, "rejected", Headers()))

    monkeypatch.setattr(deepgram_synthesizer.websockets, "connect", reject)

    assert await synth.establish_connection() is None
    assert bool(synth.connection_error) is fatal
