"""Exercise Timbre's registered provider against local REST, SSE and WebSocket endpoints."""

import asyncio
import base64
import json
from contextlib import aclosing
from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from pydantic import ValidationError

import bolna.synthesizer.gnani_synthesizer as gnani
from bolna.models import Synthesizer, tts_control_range, tts_provider_config_error, tts_render_settings
from bolna.providers import SUPPORTED_SYNTHESIZER_MODELS
from bolna.synthesizer.gnani_config import GnaniConfig


def make_synth(**config):
    return SUPPORTED_SYNTHESIZER_MODELS["gnani"](voice="Kaveri", synthesizer_key="per-call-key", **config)


def message(text, sequence=1, final=True):
    return {"data": text, "meta_info": {"sequence_id": sequence, "end_of_llm_stream": final}}


@pytest.fixture
async def provider(monkeypatch):
    state = SimpleNamespace(requests=[], connections=0, stalled=asyncio.Event())

    async def rest(request):
        payload = await request.json()
        state.requests.append((request.headers["X-API-Key-ID"], payload))
        if payload["text"] == "Failure":
            return web.json_response({"message": "invalid key"}, status=401)
        if payload["text"] == "No audio":
            return web.Response(body=b"", content_type="audio/wav")
        container = payload["audio_config"]["container"]
        return web.Response(body=container.encode(), content_type="application/octet-stream")

    def events(text):
        yield "start", {}
        if text == "Failure":
            yield "error", {"message": "provider error"}
            return
        yield "audio", {"audio": base64.b64encode(text.encode()).decode(), "is_final": False}
        if text != "Truncated":
            yield "complete", {"audio": base64.b64encode(b"last").decode(), "is_final": True}

    async def sse(request):
        payload = await request.json()
        state.requests.append((request.headers["X-API-Key-ID"], payload))
        frames = []
        for kind, data in events(payload["text"]):
            event = "chunk" if kind == "audio" else kind
            frames.append(f"event: {event}\r\ndata: {json.dumps(data)}\r\n\r\n")
        return web.Response(text="".join(frames), content_type="text/event-stream")

    async def websocket(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        state.connections += 1
        async for frame in ws:
            payload = frame.json()
            state.requests.append((request.headers["X-API-Key-ID"], payload))
            if payload["text"] == "Stall":
                state.stalled.set()
                continue
            for kind, data in events(payload["text"]):
                await ws.send_json({"type": kind, "message": data.get("message"), "data": data})
            await ws.send_json({"type": "complete", "message": "Streaming completed", "request_id": "example"})
        return ws

    app = web.Application()
    app.add_routes([web.post("/rest", rest), web.post("/sse", sse), web.get("/ws", websocket)])
    async with TestServer(app) as server:
        monkeypatch.setattr(gnani, "REST_URL", str(server.make_url("/rest")))
        monkeypatch.setattr(gnani, "SSE_URL", str(server.make_url("/sse")))
        monkeypatch.setattr(gnani, "WS_URL", str(server.make_url("/ws").with_scheme("ws")))
        yield state


def test_agent_and_openapi_expose_timbre_models_and_matching_voices():
    schema = Synthesizer.model_json_schema()
    assert "gnani" in schema["properties"]["provider"]["enum"]
    catalog = schema["$defs"]["GnaniConfig"]["x-model-voices"]
    for model, voice in [("timbre-v2.5", "Kaveri"), ("timbre-v3.0", "Shivani")]:
        assert voice in catalog[model]
        config = Synthesizer(provider="gnani", provider_config={"model": model, "voice": voice})
        assert isinstance(config.provider_config, GnaniConfig)
        assert config.provider_config.stream_transport == "websocket"
    assert tts_render_settings("gnani", {"language": "hi-IN", "speed": 1.2}) == "language=hi-IN,speed=1.2"


@pytest.mark.parametrize(
    "config",
    [
        {"voice": "Shivani"},
        {"model": "timbre-v3.0", "voice": "Kaveri"},
        {"voice": "Kaveri", "speed": 1.2},
        {"model": "timbre-v3.0", "voice": "Shivani", "speed": 1.1},
        {"voice": "Kaveri", "language": "xx"},
    ],
)
def test_invalid_settings_are_rejected_at_agent_creation(config):
    with pytest.raises(ValidationError):
        Synthesizer(provider="gnani", provider_config=config)


def test_model_specific_speed_validation():
    assert GnaniConfig(voice="Kaveri", speed="slow").speed == 0.85
    assert GnaniConfig(voice="Shivani", model="timbre-v3.0", speed=1.2).speed == 1.2
    assert tts_provider_config_error("gnani", {"model": "timbre-v3.0", "speed": 1.1})
    assert tts_provider_config_error("gnani", {"model": "timbre-v2.5", "speed": "fast"}) is None
    assert tts_provider_config_error("gnani", {"speed": 1.2})
    assert tts_control_range("gnani", "speed", "timbre-v2.5") == (0.85, 1.15)


async def test_clip_formats_and_per_call_credentials(provider, monkeypatch):
    monkeypatch.setenv("GNANI_API_KEY", "environment-key")
    synth = make_synth(sampling_rate=24000)
    assert await synth.synthesize("Clip") == b"wav"
    assert await synth.synthesize_pcm_clip("Clip", 16000) == b"raw"
    assert await synth.synthesize_telephony_clip("Clip") == b"mulaw"
    assert [p["audio_config"]["sample_rate"] for _, p in provider.requests] == [24000, 16000, 8000]
    assert all(key == "per-call-key" for key, _ in provider.requests)
    assert gnani.GnaniSynthesizer(voice="Kaveri").api_key == "environment-key"
    assert all(p["model"] == "timbre-v2.5" for _, p in provider.requests)


@pytest.mark.parametrize("transport", ["sse", "websocket"])
async def test_turn_boundaries_audio_and_connection_reuse(provider, transport):
    synth = make_synth(stream=True, stream_transport=transport, use_mulaw=True)
    try:
        async with aclosing(synth.generate()) as packets:
            for sequence in (1, 2):
                await synth.push(message("Your appointment", sequence, final=False))
                await synth.push(message("is ready.", sequence))
                first, last = await anext(packets), await anext(packets)
                assert first["data"] == b"Your appointment is ready."
                assert last["data"] == b"last"
                a, b = first["meta_info"], last["meta_info"]
                assert a["format"] == b["format"] == "mulaw"
                assert a["sequence_id"] == b["sequence_id"] == sequence
                assert a["is_first_chunk"] and not a.get("end_of_synthesizer_stream")
                assert not b["is_first_chunk"] and b["end_of_synthesizer_stream"]
                assert a["mark_id"] != b["mark_id"]
                assert a["text_synthesized"] == "" and b["text_synthesized"] == "Your appointment is ready. "
            await synth.push(message("", 3))
            assert (await anext(packets))["meta_info"]["end_of_synthesizer_stream"]
        assert len(provider.requests) == 2
        assert all(p["audio_config"]["container"] == "mulaw" for _, p in provider.requests)
        assert provider.connections == int(transport == "websocket")
    finally:
        await synth.cleanup()


async def test_interruption_retires_stalled_socket_and_buffered_text(provider):
    synth = make_synth(stream=True)
    try:
        async with aclosing(synth.generate()) as packets:
            await synth.push(message("Stall"))
            pending = asyncio.create_task(anext(packets))
            await asyncio.wait_for(provider.stalled.wait(), timeout=3)
            await synth.push(message("Abandoned fragment", final=False))
            await synth.handle_interruption()
            await synth.push(message("Replacement.", 2))
            first = await asyncio.wait_for(pending, timeout=3)
            assert first["data"] == b"Replacement."
            assert first["meta_info"]["sequence_id"] == 2
            assert (await anext(packets))["meta_info"]["end_of_synthesizer_stream"]
        assert provider.connections == 2
        assert all(p["audio_config"]["container"] == "raw" for _, p in provider.requests)
    finally:
        await synth.cleanup()


@pytest.mark.parametrize("transport", ["sse", "websocket"])
async def test_provider_errors_propagate(provider, transport):
    synth = make_synth(stream=True, stream_transport=transport)
    try:
        await synth.push(message("Failure"))
        async with aclosing(synth.generate()) as packets:
            with pytest.raises(RuntimeError, match="provider error"):
                await anext(packets)
    finally:
        await synth.cleanup()


async def test_sse_eof_is_not_a_successful_turn(provider):
    synth = make_synth(stream=True, stream_transport="sse")
    await synth.push(message("Truncated"))
    async with aclosing(synth.generate()) as packets:
        with pytest.raises(ConnectionError, match="before completion"):
            await anext(packets)


async def test_rest_cache_and_failure(provider):
    synth = make_synth(audio_format="mp3")
    async with aclosing(synth.generate()) as packets:
        for cached in [False, True]:
            await synth.push(message("Hello"))
            packet = await anext(packets)
            assert packet["data"] == b"mp3" and packet["meta_info"]["is_cached"] is cached
    assert len(provider.requests) == 1
    with pytest.raises(gnani.aiohttp.ClientResponseError) as error:
        await synth.synthesize("Failure")
    assert error.value.status == 401
    with pytest.raises(ValueError, match="empty audio"):
        await synth.synthesize("No audio")


async def test_v3_expressive_tag_scope_is_preserved_across_chunks(provider):
    synth = gnani.GnaniSynthesizer(voice="Shivani", model="timbre-v3.0", stream=True, synthesizer_key="per-call-key")
    try:
        async with aclosing(synth.generate()) as packets:
            await synth.push(message("<whispers> Please hold.", final=False))
            await synth.push(message("Your order is ready."))
            assert (await anext(packets))["data"] == b"<whispers> Please hold. Your order is ready."
            assert (await anext(packets))["meta_info"]["end_of_synthesizer_stream"]
        assert len(provider.requests) == 1
    finally:
        await synth.cleanup()
