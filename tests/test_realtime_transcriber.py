"""RealtimeTranscriber: the OpenAI Realtime transcription protocol onto the transcriber queue."""

import asyncio
import json

import websockets

from bolna.transcriber.realtime_transcriber import RealtimeTranscriber

AUDIO = {"data": b"\x00" * 1600, "meta_info": {"io": "plivo", "sequence": 1, "call_sid": "c", "stream_sid": "s"}}
EOS = {"data": None, "meta_info": {"eos": True}}


def _transcriber(provider="plivo", **kwargs):
    return RealtimeTranscriber(provider, output_queue=asyncio.Queue(), **kwargs)


def _feed(t, *events):
    packets = []
    for event in events:
        packets += t.handle_event(event)
    return [p["data"] if isinstance(p["data"], str) else p["data"]["type"] for p in packets], packets


def _input(t):
    return t.session_update()["session"]["audio"]["input"]


def test_session_uses_the_telephony_audio_as_it_arrives():
    assert _input(_transcriber("plivo"))["format"] == {"type": "audio/pcm", "rate": 8000}
    assert _input(_transcriber("twilio"))["format"] == {"type": "audio/pcmu", "rate": 8000}
    web = _transcriber("web_based_call", encoding="linear16", sampling_rate=16000)
    assert _input(web)["format"] == {"type": "audio/pcm", "rate": 16000}


def test_endpointing_is_the_silence_wait_within_the_server_range():
    assert _input(_transcriber(endpointing=500))["turn_detection"]["silence_duration_ms"] == 500
    assert _input(_transcriber(endpointing=50))["turn_detection"]["silence_duration_ms"] == 200
    assert "silence_duration_ms" not in _input(_transcriber())["turn_detection"]
    assert _input(_transcriber(eager_end_of_turn=False))["turn_detection"]["eager_end_of_turn"] is False


def test_a_turn_maps_onto_the_queue_with_cumulative_interims():
    t = _transcriber()
    kinds, packets = _feed(
        t,
        {"type": "input_audio_buffer.speech_started", "item_id": "i1"},
        {
            "type": "conversation.item.input_audio_transcription.delta",
            "item_id": "i1",
            "delta": "haan",
            "transcript": "haan",
        },
        {
            "type": "conversation.item.input_audio_transcription.delta",
            "item_id": "i1",
            "delta": " ji",
            "transcript": "haan ji",
        },
        {"type": "input_audio_buffer.eager_end_of_turn", "item_id": "i1", "transcript": "haan ji"},
        {"type": "input_audio_buffer.speech_stopped", "item_id": "i1", "audio_end_ms": 900},
        {"type": "conversation.item.input_audio_transcription.completed", "item_id": "i1", "transcript": "haan ji"},
    )
    assert kinds == [
        "speech_started",
        "interim_transcript_received",
        "interim_transcript_received",
        "eager_end_of_turn",
        "transcript",
    ]
    assert packets[2]["data"]["content"] == "haan ji"
    assert packets[-1]["data"] == {"type": "transcript", "content": "haan ji", "was_eager": True}
    assert packets[-1]["meta_info"]["asr_turn_id"] == 1
    assert t.turn_latencies[-1]["final_transcript"] == "haan ji"


def test_a_server_sending_only_deltas_keeps_the_spaces_between_them():
    t = _transcriber()
    delta = {"type": "conversation.item.input_audio_transcription.delta", "item_id": "i1"}
    _, packets = _feed(t, {**delta, "delta": "haan "}, {**delta, "delta": "ji"})
    assert packets[-1]["data"]["content"] == "haan ji"


def test_speech_stopped_alone_never_ends_the_turn():
    t = _transcriber()
    kinds, _ = _feed(
        t,
        {"type": "input_audio_buffer.speech_started", "item_id": "i1"},
        {"type": "input_audio_buffer.speech_stopped", "item_id": "i1", "audio_end_ms": 900},
    )
    assert kinds == ["speech_started"]


def test_a_resumed_turn_voids_the_eager_transcript():
    t = _transcriber()
    kinds, packets = _feed(
        t,
        {"type": "input_audio_buffer.eager_end_of_turn", "item_id": "i1", "transcript": "haan"},
        {"type": "input_audio_buffer.turn_resumed", "item_id": "i1"},
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "i1",
            "transcript": "haan ji bolo",
        },
    )
    assert kinds == ["eager_end_of_turn", "turn_resumed", "transcript"]
    assert packets[-1]["data"]["was_eager"] is False


def test_a_final_that_differs_from_the_eager_transcript_voids_it():
    t = _transcriber()
    kinds, packets = _feed(
        t,
        {"type": "input_audio_buffer.eager_end_of_turn", "item_id": "i1", "transcript": "haan"},
        {"type": "conversation.item.input_audio_transcription.completed", "item_id": "i1", "transcript": "haan ji"},
    )
    assert kinds == ["eager_end_of_turn", "turn_resumed", "transcript"]
    assert packets[-1]["data"]["was_eager"] is False


def test_an_empty_final_releases_the_turn():
    t = _transcriber()
    kinds, _ = _feed(
        t,
        {"type": "input_audio_buffer.eager_end_of_turn", "item_id": "i1", "transcript": "haan"},
        {"type": "conversation.item.input_audio_transcription.completed", "item_id": "i1", "transcript": ""},
    )
    assert kinds == ["eager_end_of_turn", "turn_resumed", "speech_ended"]


async def test_a_turn_the_server_never_settles_is_released():
    t = _transcriber()
    t.STUCK_TURN_S = 0.0
    t.handle_event({"type": "input_audio_buffer.eager_end_of_turn", "item_id": "i1", "transcript": "haan"})
    watchdog = asyncio.create_task(t._watch_turns())
    resumed = await asyncio.wait_for(t.transcriber_output_queue.get(), 2)
    final = await asyncio.wait_for(t.transcriber_output_queue.get(), 2)
    watchdog.cancel()
    assert resumed["data"] == {"type": "turn_resumed"}
    assert final["data"] == {"type": "transcript", "content": "haan", "was_eager": False, "force_finalized": True}


class FakeServer:
    """Answers the session (or refuses it), records what the client sends, and plays `script` after the first
    audio append. `hang_up` closes the socket after the script, as a crashed server would."""

    def __init__(self, script=(), refuse=False, hang_up=False):
        self.script, self.refuse, self.hang_up = script, refuse, hang_up
        self.session, self.received, self.auth, self.path = None, [], None, None

    async def handler(self, ws):
        self.auth, self.path = ws.request.headers.get("Authorization"), ws.request.path
        await ws.send(json.dumps({"type": "transcription_session.created"}))
        if self.refuse:
            error = {"code": "server_at_capacity", "message": "at the session cap"}
            await ws.send(json.dumps({"type": "error", "error": error}))
            await ws.close(1013)
            return
        self.session = json.loads(await ws.recv())
        await ws.send(json.dumps({"type": "transcription_session.updated"}))
        played = False
        async for raw in ws:
            msg = json.loads(raw)
            self.received.append(msg["type"])
            if msg["type"] == "input_audio_buffer.append" and not played:
                played = True
                for event in self.script:
                    await ws.send(json.dumps(event))
                if self.hang_up:
                    await ws.close(1011)
                    return


async def _call(server, until=lambda packets: True, monkeypatch=None, prepare=None, **kwargs):
    """Stream one audio packet, wait for `until`, end the stream, and return every queue packet up to the close.
    `prepare` sets up the transcriber before it connects."""
    out, inq = asyncio.Queue(), asyncio.Queue()
    packets = []
    async with websockets.serve(server.handler, "127.0.0.1", 0) as srv:
        monkeypatch.setenv("REALTIME_TRANSCRIBER_URL", f"ws://127.0.0.1:{srv.sockets[0].getsockname()[1]}/{{model}}")
        t = RealtimeTranscriber("plivo", input_queue=inq, output_queue=out, **kwargs)
        if prepare:
            prepare(t)
        await t.run()
        await inq.put(AUDIO)
        while not until(packets) and not any(p["data"] == "transcriber_connection_closed" for p in packets):
            packets.append(await asyncio.wait_for(out.get(), 5))
        await inq.put(EOS)
        while not any(p["data"] == "transcriber_connection_closed" for p in packets):
            packets.append(await asyncio.wait_for(out.get(), 5))
    return t, packets


async def test_a_call_streams_audio_and_closes_cleanly_at_end_of_stream(monkeypatch):
    script = [
        {"type": "input_audio_buffer.speech_started", "item_id": "i1"},
        {"type": "conversation.item.input_audio_transcription.completed", "item_id": "i1", "transcript": "haan"},
    ]
    server = FakeServer(script)
    t, packets = await _call(
        server,
        until=lambda ps: any(isinstance(p["data"], dict) for p in ps),
        monkeypatch=monkeypatch,
        transcriber_key="k",
    )
    assert server.auth == "Bearer k"
    assert server.path == "/nemotron-asr-hi"
    assert server.received == ["input_audio_buffer.append", "input_audio_buffer.commit"]
    closed = packets[-1]["meta_info"]
    assert "connection_error" not in closed
    assert closed["transcriber_duration"] == 0.1
    assert t.connection_time is not None


async def test_a_refused_session_is_a_connection_error(monkeypatch):
    _, packets = await _call(FakeServer(refuse=True), monkeypatch=monkeypatch)
    assert packets[-1]["data"] == "transcriber_connection_closed"
    assert "server_at_capacity" in packets[-1]["meta_info"]["connection_error"]


async def test_a_server_hanging_up_mid_call_is_a_connection_error(monkeypatch):
    script = [{"type": "input_audio_buffer.speech_started", "item_id": "i1"}]
    _, packets = await _call(FakeServer(script, hang_up=True), until=lambda ps: False, monkeypatch=monkeypatch)
    assert packets[-1]["meta_info"]["connection_error"]


async def test_a_reconnected_session_never_finalizes_a_turn_of_the_one_that_died(monkeypatch):
    def died_mid_turn(t):
        t.STUCK_TURN_S = 0.0
        t.handle_event({"type": "input_audio_buffer.eager_end_of_turn", "item_id": "old", "transcript": "haan"})

    _, packets = await _call(FakeServer(), monkeypatch=monkeypatch, prepare=died_mid_turn)
    assert not [p for p in packets if isinstance(p["data"], dict)]


async def test_no_endpoint_is_a_connection_error(monkeypatch):
    monkeypatch.delenv("REALTIME_TRANSCRIBER_URL", raising=False)
    out = asyncio.Queue()
    t = RealtimeTranscriber("plivo", input_queue=asyncio.Queue(), output_queue=out)
    await t.run()
    packet = await asyncio.wait_for(out.get(), 2)
    assert "REALTIME_TRANSCRIBER_URL" in packet["meta_info"]["connection_error"]


def test_an_agent_config_cannot_choose_the_endpoint(monkeypatch):
    monkeypatch.setenv("REALTIME_TRANSCRIBER_URL", "wss://asr.example/{model}")
    t = _transcriber(transcriber_url="wss://attacker.example", model="nemotron-asr-hi")
    assert t.url == "wss://asr.example/nemotron-asr-hi"


def test_the_caller_stopped_when_the_speech_ended_not_when_its_frame_was_sent():
    t = _transcriber()
    t.record_audio_frame(0.2, 1_000_000.0)  # audio 0.0-0.2 s, sent at t=1000 s
    t.record_audio_frame(0.2, 1_000_200.0)  # audio 0.2-0.4 s, sent at t=1000.2 s
    t.handle_event({"type": "input_audio_buffer.speech_started", "item_id": "i1"})
    t.handle_event(
        {"type": "input_audio_buffer.speech_stopped", "item_id": "i1", "audio_end_ms": 350, "speech_end_ms": 250}
    )
    assert abs(t.meta_info["user_stop_ts_wall"] - (1000.2 - 0.15)) < 1e-6


def test_a_server_without_speech_end_ms_falls_back_to_audio_end_ms():
    t = _transcriber()
    t.record_audio_frame(0.2, 1_000_000.0)
    t.handle_event({"type": "input_audio_buffer.speech_stopped", "item_id": "i1", "audio_end_ms": 100})
    assert abs(t.meta_info["user_stop_ts_wall"] - (1000.0 - 0.1)) < 1e-6
