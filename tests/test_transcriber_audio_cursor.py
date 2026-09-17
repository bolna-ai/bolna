"""Where a chunk of audio sat in the stream, and when it was actually handed over.

Deepgram reports transcript timings as positions inside the audio stream. Turning one into a
latency means looking up the wall-clock moment the audio at that position was sent, so the
position -> send-time map is only as good as the duration each websocket send is booked for.
Booking a fixed 200 ms per send while the sip-trunk input handler batches 80 ms stretches the
map 2.5x: every lookup resolves to a packet from far earlier in the call, and the reported
latency ramps with call elapsed time instead of measuring anything.

The clock here is driven by the audio itself — a send advances it by exactly the duration of
the bytes handed over — so packets arrive at real-time pace without the test sleeping.
"""

import asyncio
import json
import time as real_time

import pytest

from bolna.helpers.utils import create_ws_data_packet
from bolna.transcriber import deepgram_transcriber
from bolna.transcriber.deepgram_transcriber import DeepgramTranscriber

MULAW_BYTES_PER_SECOND = 8000
# sip_trunk._listen merges 4 x 20 ms ulaw frames before ingest_audio -> 80 ms per send.
SIP_TRUNK_PACKET_BYTES = 640
# TranscriberPool._silence_frame keepalive -> 40 ms of ulaw at 8 kHz.
POOL_KEEPALIVE_PACKET_BYTES = 320
# freeswitch.INGEST_CHUNK_BYTES -> 200 ms of linear16 at 16 kHz.
WEBCALL_CHUNK_BYTES = 6400
STREAM_START_MS = 1_700_000_000_000.0


class _Clock:
    """Wall clock advanced by exactly the audio handed over, i.e. real-time pace."""

    def __init__(self, start_ms=STREAM_START_MS):
        self.now_ms = float(start_ms)

    def ms(self):
        return self.now_ms

    def seconds(self):
        return self.now_ms / 1000.0

    def advance_seconds(self, seconds):
        self.now_ms += seconds * 1000.0


class _FakeTimeModule:
    """Stands in for `time` inside the transcriber module; only time() is redirected."""

    def __init__(self, clock):
        self._clock = clock

    def time(self):
        return self._clock.seconds()

    def __getattr__(self, name):
        return getattr(real_time, name)


class _FakeWS:
    """Accepts sends and yields canned frames, with no network.

    Binary sends advance the clock by the real duration of the audio they carry; text sends
    (CloseStream, KeepAlive) do not. When `drain_queue` is given, iteration first lets the
    sender task empty that queue, so a full transcribe() run books its audio before the first
    Deepgram message lands.
    """

    _MAX_DRAIN_TICKS = 50

    def __init__(self, clock, messages=(), bytes_per_second=MULAW_BYTES_PER_SECOND, drain_queue=None):
        self._clock = clock
        self._messages = list(messages)
        self._bytes_per_second = bytes_per_second
        self._drain_queue = drain_queue

    async def send(self, data):
        if isinstance(data, (bytes, bytearray)):
            self._clock.advance_seconds(len(data) / self._bytes_per_second)

    async def close(self):
        pass

    async def __aiter__(self):
        for _ in range(self._MAX_DRAIN_TICKS):
            if self._drain_queue is None or self._drain_queue.empty():
                break
            await asyncio.sleep(0)
        for message in self._messages:
            yield message


def _make_transcriber(monkeypatch, clock, provider="sip-trunk"):
    monkeypatch.setattr(deepgram_transcriber, "timestamp_ms", clock.ms)
    monkeypatch.setattr(deepgram_transcriber, "time", _FakeTimeModule(clock))
    transcriber = DeepgramTranscriber(
        telephony_provider=provider,
        model="nova-3",
        language="en",
        stream=True,
        input_queue=asyncio.Queue(),
        output_queue=asyncio.Queue(),
    )
    transcriber.meta_info = {"request_id": "test-request"}
    # Encoding, sample rate and audio_frame_duration are provider-resolved on the connect path.
    transcriber.get_deepgram_ws_url()
    return transcriber


def _queue_packets(transcriber, packets):
    for payload in packets:
        transcriber.input_queue.put_nowait(create_ws_data_packet(payload, dict(transcriber.meta_info)))
    transcriber.input_queue.put_nowait(create_ws_data_packet(None, {"eos": True}))


async def _stream(transcriber, ws, packets):
    _queue_packets(transcriber, packets)
    await transcriber.sender_stream(ws)


@pytest.mark.parametrize(
    "payload_bytes, packet_seconds",
    [
        pytest.param(SIP_TRUNK_PACKET_BYTES, 0.080, id="sip-trunk-80ms"),
        pytest.param(POOL_KEEPALIVE_PACKET_BYTES, 0.040, id="pool-keepalive-40ms"),
    ],
)
async def test_lookup_resolves_to_the_packet_that_carried_the_audio(monkeypatch, payload_bytes, packet_seconds):
    """INVARIANT: a transcript position resolves to the send time of the packet that really
    carried that audio, whatever payload size the input handler batches — so the reported
    latency stays flat instead of ramping with how far into the call you are."""
    clock = _Clock()
    transcriber = _make_transcriber(monkeypatch, clock)

    position_s = 40.01
    packet_count = int((position_s + 5.0) / packet_seconds)
    await _stream(transcriber, _FakeWS(clock), [b"\xff" * payload_bytes] * packet_count)

    sent_at_ms = transcriber._find_audio_send_timestamp(position_s)

    assert sent_at_ms is not None
    error_ms = abs(sent_at_ms - (STREAM_START_MS + position_s * 1000))
    assert error_ms <= packet_seconds * 1000


async def test_connection_start_time_lands_at_the_real_stream_start(monkeypatch):
    """INVARIANT: connection_start_time is the wall clock at which the stream began. It is
    derived by subtracting the audio submitted so far, so it is only right when that total is
    the audio really submitted. user_stop_ts_wall, and every user-bot latency built on it,
    inherits whatever bias this carries."""
    clock = _Clock()
    transcriber = _make_transcriber(monkeypatch, clock)
    stream_start_s = clock.seconds()

    # 250 x 80 ms = 20 s of sip-trunk audio.
    await _stream(transcriber, _FakeWS(clock), [b"\xff" * SIP_TRUNK_PACKET_BYTES] * 250)
    ws = _FakeWS(clock, messages=[json.dumps({"type": "SpeechStarted"})])
    _ = [packet async for packet in transcriber.receiver(ws)]

    assert transcriber.connection_start_time is not None
    assert abs(transcriber.connection_start_time - stream_start_s) <= 0.080


@pytest.mark.parametrize(
    "provider",
    [pytest.param("twilio", id="twilio"), pytest.param("plivo", id="plivo"), pytest.param("exotel", id="exotel")],
)
async def test_telephony_providers_keep_booking_their_constant(monkeypatch, provider):
    """INVARIANT: the telephony providers whose handler really does batch 200 ms are untouched.
    They keep booking exactly the constant they booked before, whatever payload size arrives, so
    neither this change nor a later input-handler change can move their map unnoticed."""
    clock = _Clock()
    transcriber = _make_transcriber(monkeypatch, clock, provider=provider)
    legacy = transcriber.audio_frame_duration

    # A payload of the wrong size must not move them: the constant is what they book.
    transcriber.record_audio_frame(transcriber._audio_frame_seconds(SIP_TRUNK_PACKET_BYTES), clock.ms())
    transcriber.record_audio_frame(transcriber._audio_frame_seconds(1600), clock.ms())

    assert legacy == 0.200
    assert transcriber.audio_frame_timestamps[0][:2] == (0.0, legacy)
    assert transcriber.audio_frame_timestamps[1][:2] == (legacy, 2 * legacy)


async def test_sip_trunk_batches_are_booked_at_eighty_milliseconds(monkeypatch):
    """INVARIANT: the shape this fix exists for is exact, not merely close. sip_trunk._listen
    merges 4 x 20 ms ulaw frames, so 640 B at 8 kHz is 80 ms — a quarter of the 200 ms it was
    booked for — and consecutive sends must tile the stream with no gap and no overlap."""
    clock = _Clock()
    sip_trunk = _make_transcriber(monkeypatch, clock)

    sip_trunk.record_audio_frame(sip_trunk._audio_frame_seconds(SIP_TRUNK_PACKET_BYTES), clock.ms())
    sip_trunk.record_audio_frame(sip_trunk._audio_frame_seconds(SIP_TRUNK_PACKET_BYTES), clock.ms())

    assert sip_trunk.audio_frame_timestamps[0][:2] == (0.0, 0.080)
    assert sip_trunk.audio_frame_timestamps[1][:2] == (0.080, 0.160)


async def test_playground_stream_position_stays_pinned(monkeypatch):
    """INVARIANT: audio_frame_duration 0.0 marks a path that does not stream in real time, so
    nothing may be booked against it. The playground can hand over audio faster than real time;
    a running cursor would skew connection_start_time on a path nobody has measured."""
    clock = _Clock()
    transcriber = _make_transcriber(monkeypatch, clock, provider="playground")
    assert transcriber.audio_frame_duration == 0.0

    transcriber.record_audio_frame(transcriber._audio_frame_seconds(1600), clock.ms())
    transcriber.record_audio_frame(transcriber._audio_frame_seconds(1600), clock.ms())

    assert transcriber.audio_frame_timestamps == []
    assert transcriber._find_audio_send_timestamp(0.05) is None


async def test_reconnect_restarts_the_stream_position(monkeypatch):
    """INVARIANT: Deepgram audio positions restart at 0 on a new socket, so the local
    position -> send-time map and the stream start must restart with them. A pool reconnect
    re-runs run() -> transcribe() on the same instance, so stale state would map the new
    connection's positions onto the previous connection's wall-clock times."""
    clock = _Clock()
    transcriber = _make_transcriber(monkeypatch, clock)

    async def connect():
        return _FakeWS(
            clock,
            messages=[json.dumps({"type": "SpeechStarted"})],
            drain_queue=transcriber.input_queue,
        )

    monkeypatch.setattr(transcriber, "deepgram_connect", connect)

    # First connection: 20 s of audio.
    _queue_packets(transcriber, [b"\xff" * SIP_TRUNK_PACKET_BYTES] * 250)
    await transcriber.transcribe()
    assert transcriber.audio_frame_timestamps, "first connection booked no audio"

    # Second connection on the same instance.
    second_stream_start_s = clock.seconds()
    _queue_packets(transcriber, [b"\xff" * SIP_TRUNK_PACKET_BYTES] * 5)
    await transcriber.transcribe()

    assert transcriber.audio_frame_timestamps[0][0] == 0.0
    assert transcriber._find_audio_send_timestamp(0.04) is not None
    assert abs(transcriber.connection_start_time - second_stream_start_s) <= 0.080


@pytest.mark.parametrize(
    "provider, legacy_constant",
    [
        pytest.param("web_based_call", 0.256, id="web_based_call"),
        pytest.param("freeswitch", 0.500, id="freeswitch"),
    ],
)
async def test_web_calls_are_booked_for_the_two_hundred_milliseconds_they_carry(monkeypatch, provider, legacy_constant):
    """INVARIANT: the webcall paths carry 200 ms per send — freeswitch coalesces to
    INGEST_CHUNK_BYTES, 6400 B of linear16 at 16 kHz — not the 256 ms / 500 ms they were booked
    at. Same defect as sip-trunk, measured at a 0.64 drift slope in prod, so the cursor has to
    measure here too and consecutive sends must tile without gap or overlap."""
    clock = _Clock()
    transcriber = _make_transcriber(monkeypatch, clock, provider=provider)
    assert transcriber.audio_frame_duration == legacy_constant

    transcriber.record_audio_frame(transcriber._audio_frame_seconds(WEBCALL_CHUNK_BYTES), clock.ms())
    transcriber.record_audio_frame(transcriber._audio_frame_seconds(WEBCALL_CHUNK_BYTES), clock.ms())

    assert transcriber.audio_frame_timestamps[0][:2] == (0.0, 0.200)
    assert transcriber.audio_frame_timestamps[1][:2] == (0.200, 0.400)
