"""Every ASR connection bills its own audio exactly once.

Most providers never reported a duration (billed 0), Sarvam kept only the last segment, a pool reconnect
re-sent the previous connection's number, and some providers sent a second closing packet that was billed
again. task_manager also summed only the closing packets its listener read, and after a hangup that is the
first one, so a pool call could bill a standby's keepalives alone. The call total now comes from the
transcribers themselves at teardown.
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from websockets.exceptions import ConnectionClosedError

from bolna.agent_manager.task_manager import TaskManager
from bolna.helpers.utils import create_ws_data_packet
from bolna.transcriber.assemblyai_transcriber import AssemblyAITranscriber
from bolna.transcriber.azure_transcriber import AzureTranscriber
from bolna.transcriber.deepgram_transcriber import DeepgramTranscriber
from bolna.transcriber.elevenlabs_transcriber import ElevenLabsTranscriber
from bolna.transcriber.gemini_transcriber import GeminiTranscriber
from bolna.transcriber.gladia_transcriber import GladiaTranscriber
from bolna.transcriber.google_transcriber import GoogleTranscriber
from bolna.transcriber.openai_transcriber import OpenAITranscriber
from bolna.transcriber.pixa_transcriber import PixaTranscriber
from bolna.transcriber.sarvam_transcriber import SarvamTranscriber
from bolna.transcriber.smallest_transcriber import SmallestTranscriber
from bolna.transcriber.soniox_transcriber import SonioxTranscriber
from bolna.transcriber.transcriber_pool import TranscriberPool

# plivo streams 8 kHz linear16: 0.2 s per 3200-byte frame.
FRAME_BYTES = 3200
FRAME_S = 0.2
LOUD_FRAME = (3000).to_bytes(2, "little", signed=True) * (FRAME_BYTES // 2)

WEBSOCKET_PROVIDERS = {
    "deepgram": (DeepgramTranscriber, "deepgram_connect"),
    "soniox": (SonioxTranscriber, "soniox_connect"),
    "sarvam": (SarvamTranscriber, "sarvam_connect"),
    "elevenlabs": (ElevenLabsTranscriber, "elevenlabs_connect"),
    "openai": (OpenAITranscriber, "openai_connect"),
    "pixa": (PixaTranscriber, "pixa_connect"),
    "gladia": (GladiaTranscriber, "gladia_connect"),
    "smallest": (SmallestTranscriber, "smallest_connect"),
    "assemblyai": (AssemblyAITranscriber, "assemblyai_connect"),
    "gemini": (GeminiTranscriber, "gemini_connect"),
}


class _AlwaysSet(asyncio.Event):
    """OpenAI's sender waits up to 5 s for a final transcript after eos; no server here sends one."""

    def clear(self):
        pass


class FakeSocket:
    """Holds its replies until the test releases them, so the sender drains the input first."""

    def __init__(self, replies=(), then=None):
        self.sent = []
        self.replies = list(replies)
        self.then = then
        self.release = asyncio.Event()

    async def send(self, data):
        self.sent.append(data)

    async def close(self):
        pass

    async def ping(self):
        pong = asyncio.get_running_loop().create_future()
        pong.set_result(None)
        return pong

    def __aiter__(self):
        return self._replies()

    async def _replies(self):
        await self.release.wait()
        for reply in self.replies:
            yield reply
        if self.then is not None:
            raise self.then


def _transcriber(name, provider="plivo"):
    cls, _ = WEBSOCKET_PROVIDERS[name]
    t = cls(provider, input_queue=asyncio.Queue(), output_queue=asyncio.Queue(), transcriber_key="test-key")
    # These pick their wire format while building the URL, which connect() normally does.
    for build_url in ("get_deepgram_ws_url", "get_elevenlabs_ws_url", "get_assemblyai_ws_url"):
        if hasattr(t, build_url):
            getattr(t, build_url)()
    if name == "openai":
        t._final_transcript_event = _AlwaysSet()
        t._final_transcript_event.set()
    return t


def _feed(t, frames, frame=LOUD_FRAME, eos=True):
    for _ in range(frames):
        t.input_queue.put_nowait({"data": frame, "meta_info": {"io": "plivo"}})
    if eos:
        t.input_queue.put_nowait({"data": None, "meta_info": {"io": "plivo", "eos": True}})


def _closing_packets(t):
    packets = []
    while not t.transcriber_output_queue.empty():
        packet = t.transcriber_output_queue.get_nowait()
        if packet["data"] == "transcriber_connection_closed":
            packets.append(packet)
    return packets


def test_closing_packet_bills_the_providers_own_number_when_it_reports_one():
    t = _transcriber("deepgram")
    t.meta_info = {"request_id": "r1"}
    t.count_audio_sent(16000, 8000)
    t.provider_audio_duration_s = 1.7

    meta = t.closing_meta()

    assert meta["transcriber_duration"] == 1.7
    assert meta["request_id"] == "r1"
    assert "transcriber_duration" not in t.meta_info


def test_closing_packet_falls_back_to_the_audio_actually_sent():
    t = _transcriber("deepgram")
    t.count_audio_sent(16000, 8000)
    t.count_audio_sent(8000, 8000, mulaw=True)

    assert t.closing_meta()["transcriber_duration"] == 2.0


def test_a_second_closing_packet_for_the_same_connection_bills_nothing():
    t = _transcriber("deepgram")
    t.count_audio_sent(16000, 8000)

    first, second = t.closing_meta(), t.closing_meta()

    assert first["transcriber_duration"] == 1.0
    assert second["transcriber_duration"] == 0
    assert t.billed_audio_s() == 1.0


def test_call_total_counts_closed_unclosed_and_open_connections_once():
    t = _transcriber("deepgram")
    t.provider_audio_duration_s = 20.0
    t.closing_meta()  # conn 1 closed on the provider's number
    t.count_audio_sent(5 * 16000, 8000)
    t.reset_billed_audio()  # conn 2 died without a closing packet; the reconnect opens conn 3
    t.count_audio_sent(2 * 16000, 8000)  # conn 3 is still open at teardown

    assert t.billed_audio_s() == pytest.approx(27.0)


@pytest.mark.parametrize("name", WEBSOCKET_PROVIDERS)
async def test_failed_reconnect_sends_one_closing_packet_billed_zero(name):
    """A pool reconnect reuses the instance; the dead attempt must not re-bill the previous connection."""
    t = _transcriber(name)
    t.meta_info = {"request_id": "previous-connection"}
    t.count_audio_sent(20 * 16000, 8000)
    t.provider_audio_duration_s = 20.0

    _, connect = WEBSOCKET_PROVIDERS[name]
    with patch.object(t, connect, AsyncMock(side_effect=ConnectionError("refused"))):
        await asyncio.wait_for(t.transcribe(), timeout=5)

    closing = _closing_packets(t)
    assert len(closing) == 1
    assert closing[0]["meta_info"]["transcriber_duration"] == 0


@pytest.mark.parametrize("name", WEBSOCKET_PROVIDERS)
async def test_sender_counts_the_audio_it_sends(name):
    t = _transcriber(name)
    _feed(t, frames=5)

    await asyncio.wait_for(t.sender_stream(FakeSocket()), timeout=5)

    # OpenAI resamples to 24 kHz before sending, which can shift the length by a sample.
    assert t.audio_sent_s == pytest.approx(5 * FRAME_S, abs=0.005)


async def test_mulaw_audio_counts_one_byte_per_sample():
    t = _transcriber("smallest", provider="twilio")
    _feed(t, frames=5, frame=b"\xff" * 1600)

    await asyncio.wait_for(t.sender_stream(FakeSocket()), timeout=5)

    assert t.audio_sent_s == pytest.approx(1.0)


@pytest.mark.parametrize(
    "name, end_message, provider_duration",
    [
        ("gladia", {"type": "done", "data": {"duration": 7.5}}, 7.5),
        ("assemblyai", {"type": "Termination", "audio_duration_seconds": 7.5}, 7.5),
        ("smallest", {"transcript": "", "is_final": True, "is_last": True}, None),
    ],
)
async def test_session_end_message_records_duration_without_a_second_closing_packet(
    name, end_message, provider_duration
):
    t = _transcriber(name)
    t.meta_info = {}
    ws = FakeSocket(replies=[json.dumps(end_message)])
    ws.release.set()

    packets = [packet async for packet in t.receiver(ws)]

    assert all(packet["data"] != "transcriber_connection_closed" for packet in packets)
    assert t.provider_audio_duration_s == provider_duration


async def _run_connection(t, ws, frames):
    """One transcribe() on the instance, as the pool runs it, against a scripted socket."""
    _feed(t, frames=frames, eos=False)
    with patch.object(t, "deepgram_connect", AsyncMock(return_value=ws)):
        connection = asyncio.create_task(t.transcribe())
        while not t.input_queue.empty():
            await asyncio.sleep(0)
        await asyncio.sleep(0.01)
        ws.release.set()
        await asyncio.wait_for(connection, timeout=5)
    return [p["meta_info"]["transcriber_duration"] for p in _closing_packets(t)]


async def test_deepgram_pool_reconnects_bill_each_connection_once():
    """conn 1 ends with Metadata, conn 2 drops before Metadata, the next reconnect is refused."""
    t = _transcriber("deepgram")

    metadata = json.dumps({"type": "Metadata", "duration": 20.0})
    assert await _run_connection(t, FakeSocket(replies=[metadata]), frames=10) == [20.0]

    dropped = FakeSocket(then=ConnectionClosedError(None, None))
    assert await _run_connection(t, dropped, frames=25) == [pytest.approx(5.0)]

    with patch.object(t, "deepgram_connect", AsyncMock(side_effect=ConnectionError("refused"))):
        await t.transcribe()
    assert [p["meta_info"]["transcriber_duration"] for p in _closing_packets(t)] == [0]
    assert t.billed_audio_s() == pytest.approx(25.0)


def _pool(**legs):
    pool = TranscriberPool.__new__(TranscriberPool)
    pool.transcribers = legs
    return pool


def _hung_up_task_manager(transcriber):
    tm = MagicMock()
    tm.tools = {"transcriber": transcriber}
    tm.http_transcriber_duration = 0
    tm.transcriber_output_queue = asyncio.Queue()
    tm._should_ignore_transcriber_input = lambda: True
    tm._log_transcriber_connection_error = AsyncMock()
    return tm


@pytest.mark.parametrize("active_closed", [True, False], ids=["active-packet-unread", "active-still-closing"])
async def test_hangup_bills_every_pool_leg_when_a_standby_closes_first(active_closed):
    """After a hangup the listener stops at the first closing packet, here a standby that only got keepalives."""
    active, standby = _transcriber("deepgram"), _transcriber("sarvam")
    active.count_audio_sent(180 * 16000, 8000)
    standby.count_audio_sent(45 * 1600, 8000)  # one 100 ms keepalive every 4 s
    tm = _hung_up_task_manager(_pool(hi=standby, en=active))
    closing = [standby, active] if active_closed else [standby]
    for leg in closing:
        tm.transcriber_output_queue.put_nowait(
            create_ws_data_packet("transcriber_connection_closed", leg.closing_meta())
        )

    await asyncio.wait_for(TaskManager._listen_transcriber(tm), timeout=2)

    assert tm.transcriber_output_queue.qsize() == len(closing) - 1
    assert TaskManager._transcriber_billed_duration(tm) == pytest.approx(184.5)


def test_call_without_a_transcriber_bills_only_http_results():
    tm = MagicMock()
    tm.tools = {"transcriber": None}
    tm.http_transcriber_duration = 3.5

    assert TaskManager._transcriber_billed_duration(tm) == 3.5


def test_google_counts_the_audio_handed_to_the_stream():
    with patch("bolna.transcriber.google_transcriber.speech.SpeechClient"):
        t = GoogleTranscriber("twilio", input_queue=asyncio.Queue(), output_queue=asyncio.Queue())
    for _ in range(5):
        t._audio_q.put(b"\xff" * 1600)
    t._audio_q.put(None)

    requests = list(t._audio_generator())

    assert len(requests) == 5
    assert t.audio_sent_s == pytest.approx(1.0)


async def test_azure_bills_audio_written_not_session_wall_clock():
    t = AzureTranscriber(telephony_provider="twilio", input_queue=asyncio.Queue(), output_queue=asyncio.Queue())
    t.push_stream = MagicMock()
    t.start_time = 0.0  # a session "open" for decades by wall clock
    _feed(t, frames=5, frame=b"\xff" * 1600)

    with patch.object(t, "_sync_cleanup"):
        await asyncio.wait_for(t.send_audio_to_transcriber(), timeout=5)
    await t.session_stopped_handler(SimpleNamespace())

    closing = _closing_packets(t)
    assert len(closing) == 1
    assert closing[0]["meta_info"]["transcriber_duration"] == pytest.approx(1.0)
