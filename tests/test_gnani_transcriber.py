"""Gnani Prisma uses Bolna's per-call key, native recording and streaming turn contracts."""

import asyncio
import audioop
import json
import wave
from contextlib import asynccontextmanager

import pytest
import websockets
from aiohttp import web

from bolna.assistant import Assistant
from bolna.models import Transcriber
from bolna.providers import SUPPORTED_TRANSCRIBER_PROVIDERS
from bolna.transcriber import GnaniTranscriber
from bolna.transcriber import gnani_transcriber as gnani


def make_asr(carrier="default", **kwargs):
    return SUPPORTED_TRANSCRIBER_PROVIDERS["gnani"](
        carrier, input_queue=asyncio.Queue(), output_queue=asyncio.Queue(), **kwargs
    )


def test_assistant_serializes_selected_transcriber():
    agent = Assistant()
    agent.add_task("conversation", llm_agent=None, transcriber=Transcriber(provider="gnani"))
    config = agent.tasks[0]["tools_config"]["transcriber"]
    assert (config["provider"], config["model"], config["language"]) == ("gnani", "prisma-v2.5", "en-IN")


def test_prisma_models_are_discoverable_in_the_agent_schema():
    schema = Transcriber.model_json_schema()
    assert "gnani" in schema["properties"]["provider"]["enum"]
    options = schema["x-provider-options"]["gnani"]
    assert options["models"] == ["prisma-v2.5", "prisma-v3.0"]
    assert options["default_model"] == Transcriber(provider="gnani").model
    for model in options["models"]:
        assert Transcriber(provider="gnani", model=model).model == model


def test_agent_configuration_and_keys(monkeypatch):
    config = Transcriber(provider="gnani")
    assert (config.model, config.language) == ("prisma-v2.5", "en-IN")
    assert Transcriber().model == "nova-2"
    monkeypatch.setenv("GNANI_API_KEY", "environment")
    assert make_asr()._auth_headers() == {"x-api-key-id": "environment"}
    assert make_asr(transcriber_key="tenant")._auth_headers() == {"x-api-key-id": "tenant"}
    monkeypatch.delenv("GNANI_API_KEY")
    with pytest.raises(ValueError, match="GNANI_API_KEY"):
        make_asr()._auth_headers()


@pytest.mark.parametrize(
    "model,language",
    [
        ("prisma-v3", "en-IN"),
        ("prisma-v3.0", "as-IN"),
        ("prisma-v2.5", "hi-IN,en-IN"),
        ("prisma-v3.0", "hi-IN,en-IN,bn-IN,ta-IN,te-IN"),
    ],
)
def test_reject_invalid_settings(model, language):
    with pytest.raises(ValueError, match="Gnani"):
        Transcriber(provider="gnani", model=model, language=language)


def test_language_settings_follow_selected_transport():
    assert make_asr(language="indic-auto").language == "indic-auto"
    with pytest.raises(ValueError, match="language"):
        make_asr(language="indic-auto", stream=False)
    for language in ("indic-auto", "hi-IN,en-IN,ta-IN,bn-IN"):
        assert make_asr(model="prisma-v3.0", language=language, stream=False).language == language


@pytest.mark.parametrize(
    "model,carrier,language", [("prisma-v2.5", "twilio", "en-IN"), ("prisma-v3.0", "default", "hi-IN,en-IN")]
)
async def test_stream_protocol_and_final_drain(monkeypatch, model, carrier, language):
    asr = make_asr(carrier, model=model, language=language, transcriber_key="test")
    pcm = b"\x20\x04" * 701
    if carrier == "twilio":
        source = audioop.lin2ulaw(pcm, 2)
        pcm, _ = audioop.ratecv(audioop.ulaw2lin(source, 2), 2, 1, 8000, 16000, None)
    else:
        source = pcm
    expected = pcm.ljust((len(pcm) + 1023) // 1024 * 1024, b"\0")
    frames = []
    headers = {}
    send_times = []

    async def serve(ws):
        headers.update(ws.request.headers)
        await ws.send(json.dumps({"type": "connected", "config": {"sample_rate": 16000, "chunk_size": 512}}))
        async for frame in ws:
            frames.append(frame)
            send_times.append(asyncio.get_running_loop().time())
            if len(frames) == 1:
                await ws.send('{"type":"speech_start","utterance_id":"u1"}')
            if len(frames) == len(expected) // 1024 + 2:
                await ws.send('{"type":"processing"}')
                final = json.dumps(
                    {
                        "type": "transcript",
                        "utterance_id": "u1",
                        "segment_id": "s1",
                        "text": "hello",
                        "detected_language_code": "en-IN",
                    }
                )
                await ws.send(final)
                await ws.send(final)

    async with websockets.serve(serve, "127.0.0.1", 0) as server:
        monkeypatch.setattr(gnani, "STREAM_URL", f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}")
        for part in (source[:75], source[75:]):
            asr.input_queue.put_nowait({"data": part, "meta_info": {"sequence": 1}})
        asr.input_queue.put_nowait({"meta_info": {"eos": True}})
        await asyncio.wait_for(asr.transcribe(), 5)
    assert (headers["model"], headers["language_code"], headers["x-sample-rate"], headers["x-api-key-id"]) == (
        model,
        "en-IN" if model == "prisma-v2.5" else "indic-auto",
        "16000",
        "test",
    )
    assert headers.get("x-language-candidates") == (None if model == "prisma-v2.5" else "hi-IN,en-IN")
    assert all(len(frame) == 1024 for frame in frames)
    assert b"".join(frames) == expected + bytes(32 * 1024)
    assert send_times[-1] - send_times[0] >= (len(frames) - 1) * 0.031
    packets = [asr.transcriber_output_queue.get_nowait() for _ in range(asr.transcriber_output_queue.qsize())]
    assert [p["data"] for p in packets] == [
        "speech_started",
        {"type": "transcript", "content": "hello"},
        "transcriber_connection_closed",
    ]
    assert packets[1]["meta_info"]["detected_language_code"] == "en-IN"
    assert packets[1]["meta_info"]["sequence"] == 1
    assert packets[1]["meta_info"]["request_id"] == packets[0]["meta_info"]["request_id"]
    assert packets[-1]["meta_info"]["transcriber_duration"] == round(len(frames) * 0.032, 4)
    assert not asr.connection_error


@pytest.mark.parametrize("model", ["prisma-v2.5", "prisma-v3.0"])
async def test_recorded_turns(monkeypatch, model):
    asr = make_asr(model=model, stream=False, transcriber_key="byok")
    uploads = []

    async def upload(request):
        fields = await request.post()
        with wave.open(fields["audio_file"].file, "rb") as wav:
            samples = wav.readframes(wav.getnframes())
            uploads.append(
                (fields["model"], fields["language_code"], request.headers["x-api-key-id"], wav.getframerate(), samples)
            )
        result = {"success": True, "request_id": "rid", "transcript": "hello" if any(samples) else ""}
        if model == "prisma-v3.0" and any(samples):
            result["detected_language_code"] = "en-IN"
        return web.json_response(result)

    app = web.Application()
    app.router.add_post("/stt/v3", upload)
    async with start_server(app) as url:
        monkeypatch.setattr(gnani, "REST_URL", url + "/stt/v3")
        pcm = b"\x00\x05" * 1600
        wav, _ = asr._recording(pcm)
        result = await asr.transcribe_recording(wav)
        assert result["data"] == "hello"
        assert result["meta_info"]["transcriber_duration"] == 0.1
        assert ("detected_language_code" in result["meta_info"]) == (model == "prisma-v3.0")
        asr.input_queue.put_nowait({"data": bytes(3200), "meta_info": {"sequence": 2}})
        asr.input_queue.put_nowait({"meta_info": {"eos": True}})
        await asr.transcribe()
    assert uploads == [(model, "en-IN", "byok", 16000, pcm), (model, "en-IN", "byok", 16000, bytes(3200))]
    silence = asr.transcriber_output_queue.get_nowait()
    assert silence["data"] == "" and silence["meta_info"]["sequence"] == 2
    assert not asr.connection_error


@asynccontextmanager
async def start_server(app):
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    try:
        yield f"http://127.0.0.1:{runner.addresses[0][1]}"
    finally:
        await runner.cleanup()


@pytest.mark.parametrize("status", [401, 429, 500])
async def test_http_errors_reach_bolna(monkeypatch, status):
    async def reject(request):
        return web.Response(status=status)

    app = web.Application()
    app.router.add_post("/", reject)
    async with start_server(app) as url:
        monkeypatch.setattr(gnani, "REST_URL", url)
        asr = make_asr(stream=False, transcriber_key="private-test-key")
        asr.input_queue.put_nowait({"data": bytes(1024)})
        await asr.transcribe()
    closed = asr.transcriber_output_queue.get_nowait()
    assert closed["data"] == "transcriber_connection_closed"
    assert str(status) in closed["meta_info"]["connection_error"]
    assert "private-test-key" not in str(closed)


async def test_recording_rejects_oversize_before_upload():
    with pytest.raises(ValueError, match="30 seconds"):
        await GnaniTranscriber("default", stream=False).transcribe_recording(bytes(32_000 * 30 + 2))


async def test_cancel_pending_handshake(monkeypatch):
    entered = asyncio.Event()
    disconnected = asyncio.Event()

    async def stalled(ws):
        entered.set()
        await ws.wait_closed()
        disconnected.set()

    async with websockets.serve(stalled, "127.0.0.1", 0) as server:
        monkeypatch.setattr(gnani, "STREAM_URL", f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}")
        asr = make_asr(transcriber_key="test")
        await asr.run()
        await asyncio.wait_for(entered.wait(), 2)
        await asr.cleanup()
        await asyncio.wait_for(disconnected.wait(), 2)
        assert not asr.is_connected()
        assert asr.transcriber_output_queue.get_nowait()["data"] == "transcriber_connection_closed"


async def test_overlapping_turns_and_recoverable_utterance_error():
    asr = make_asr(model="prisma-v3.0", language="indic-auto")

    async def events():
        for event in (
            {"type": "speech_start", "utterance_id": "first"},
            {"type": "processing"},
            {"type": "speech_start", "utterance_id": "second"},
            {"type": "transcript", "text": "partial", "is_final": False, "utterance_id": "first"},
            {"type": "transcript", "text": "नमस्ते", "utterance_id": "first", "detected_language_code": "hi-IN"},
            {"type": "error", "message": "Audio unclear"},
            {"type": "speech_start", "utterance_id": "third"},
            {"type": "transcript", "text": "hello", "utterance_id": "third", "latency": 360},
        ):
            yield json.dumps(event)

    await asr._receive(events())
    assert asr.drained.is_set()
    packets = [asr.transcriber_output_queue.get_nowait() for _ in range(asr.transcriber_output_queue.qsize())]
    assert [p["data"] for p in packets] == [
        "speech_started",
        "speech_started",
        {"type": "transcript", "content": "नमस्ते"},
        {"type": "speech_ended", "error": "Audio unclear"},
        "speech_started",
        {"type": "transcript", "content": "hello"},
    ]
    assert [p["meta_info"]["asr_turn_id"] for p in packets] == [1, 2, 1, 2, 3, 3]
    assert packets[2]["meta_info"]["detected_language_code"] == "hi-IN"
    assert "detected_language_code" not in packets[-1]["meta_info"]
    assert packets[-1]["meta_info"]["transcriber_latency"] == 0.36


async def test_server_refusal_before_audio(monkeypatch):
    async def refuse(ws):
        await ws.send('{"type":"error","message":"Session limit reached"}')
        await ws.close(code=1008)

    async with websockets.serve(refuse, "127.0.0.1", 0) as server:
        monkeypatch.setattr(gnani, "STREAM_URL", f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}")
        asr = make_asr(transcriber_key="test")
        await asr.transcribe()
    assert asr.transcriber_output_queue.get_nowait()["meta_info"]["connection_error"] == "Session limit reached"
    assert asr.billed_audio_s() == 0
