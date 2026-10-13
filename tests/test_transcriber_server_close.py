"""A provider rejecting the stream, by close frame or at connect, becomes the transcriber's connection error."""

import asyncio
import json
from unittest.mock import AsyncMock

import pytest
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK
from websockets.frames import Close

from bolna.transcriber.assemblyai_transcriber import AssemblyAITranscriber
from bolna.transcriber.elevenlabs_transcriber import ElevenLabsTranscriber
from bolna.transcriber.gladia_transcriber import GladiaTranscriber
from bolna.transcriber.openai_transcriber import OpenAITranscriber
from bolna.transcriber.pixa_transcriber import PixaTranscriber
from bolna.transcriber.sarvam_transcriber import SarvamTranscriber
from bolna.transcriber.smallest_transcriber import SmallestTranscriber

CREDITS = "Credits exhausted. Add credits in the API Dashboard."


def sarvam():
    return SarvamTranscriber(
        telephony_provider="twilio", model="saaras:v3", language="te-IN", stream=True, transcriber_key="test-key"
    )


def openai():
    return OpenAITranscriber("twilio", transcriber_key="test-key")


def pixa():
    return PixaTranscriber("twilio", transcriber_key="test-key")


class ClosingSocket:
    """Yields the given messages, then ends the way the server closed it."""

    def __init__(self, close, messages=()):
        self.close, self.messages = close, list(messages)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.messages:
            return self.messages.pop(0)
        if self.close is not None and self.close.code in (1000, 1001):
            raise ConnectionClosedOK(self.close, None)
        raise ConnectionClosedError(self.close, None)


async def receive(transcriber, socket):
    return [message async for message in transcriber.receiver(socket)]


@pytest.mark.parametrize("make", [sarvam, openai, pixa])
async def test_a_rejecting_close_becomes_the_connection_error(make):
    transcriber = make()

    await receive(transcriber, ClosingSocket(Close(1003, CREDITS)))

    assert CREDITS in transcriber.connection_error
    assert CREDITS in transcriber.closing_meta()["connection_error"]


@pytest.mark.parametrize("make", [sarvam, openai, pixa])
@pytest.mark.parametrize("close", [Close(1000, ""), Close(1001, "going away"), None], ids=["1000", "1001", "no frame"])
async def test_a_normal_close_or_a_socket_that_just_died_is_not_an_error(make, close):
    transcriber = make()

    await receive(transcriber, ClosingSocket(close))

    assert transcriber.connection_error is None
    assert "connection_error" not in transcriber.closing_meta()


async def test_an_error_message_sent_before_the_close_is_kept():
    transcriber = sarvam()
    rejection = json.dumps({"type": "error", "data": {"message": "sample_rate not supported"}})

    await receive(transcriber, ClosingSocket(Close(1003, CREDITS), [rejection]))

    assert transcriber.connection_error == "sample_rate not supported"


@pytest.mark.parametrize(
    "cls, connect",
    [
        (AssemblyAITranscriber, "assemblyai_connect"),
        (ElevenLabsTranscriber, "elevenlabs_connect"),
        (GladiaTranscriber, "gladia_connect"),
        (PixaTranscriber, "pixa_connect"),
        (SmallestTranscriber, "smallest_connect"),
    ],
)
async def test_a_failed_connect_is_reported_on_the_closing_packet(cls, connect):
    output = asyncio.Queue()
    transcriber = cls("twilio", output_queue=output, transcriber_key="test-key")
    setattr(transcriber, connect, AsyncMock(side_effect=ConnectionError("Invalid API key")))

    await transcriber.transcribe()

    packets = [output.get_nowait() for _ in range(output.qsize())]
    closing = next(p for p in packets if p["data"] == "transcriber_connection_closed")
    assert "Invalid API key" in closing["meta_info"]["connection_error"]
