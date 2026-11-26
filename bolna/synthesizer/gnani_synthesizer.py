"""Gnani Timbre one-shot and streaming synthesis."""

import asyncio
import base64
import json
import os
import time
from contextlib import aclosing

import aiohttp
import websockets
from websockets.protocol import State

from bolna.helpers.ssl_context import get_ssl_context
from bolna.helpers.utils import create_ws_data_packet
from bolna.memory.cache.inmemory_scalar_cache import InmemoryScalarCache

from .base_synthesizer import BaseSynthesizer
from .elevenlabs_synthesizer import split_flushable
from .gnani_config import GnaniConfig

REST_URL = "https://api.vachana.ai/api/v1/tts/inference"
SSE_URL = "https://api.vachana.ai/api/v1/tts/sse"
WS_URL = "wss://api.vachana.ai/api/v1/tts"
SAMPLE_RATES = (8000, 16000, 22050, 24000, 44100, 48000)
TIMEOUT = 30


class GnaniSynthesizer(BaseSynthesizer):
    def __init__(
        self,
        voice,
        model="timbre-v2.5",
        language="auto",
        speed=1.0,
        stream_transport="websocket",
        stream=False,
        sampling_rate=24000,
        audio_format="pcm",
        use_mulaw=False,
        buffer_size=40,
        synthesizer_key=None,
        caching=True,
        **kwargs,
    ):
        super().__init__(kwargs.get("task_manager_instance"), stream, buffer_size)
        self.config = GnaniConfig(
            model=model, voice=voice, language=language, speed=speed, stream_transport=stream_transport
        )
        self.model, self.voice = self.config.model, self.config.voice
        self.api_key = synthesizer_key or os.getenv("GNANI_API_KEY")
        if not self.api_key:
            raise ValueError("Gnani requires synthesizer_key or GNANI_API_KEY")
        self.sampling_rate = 8000 if use_mulaw else int(sampling_rate)
        self.audio_format = "mulaw" if use_mulaw else ("pcm" if stream else audio_format)
        self.caching = caching
        if caching:
            self.cache = InmemoryScalarCache()
        self.websocket = None
        self._read_task = None
        self._generation = 0
        self._pending_text = ""
        self._pending_meta = {}

    def get_sleep_time(self):
        return 0

    def supports_websocket(self):
        return self.stream and self.config.stream_transport == "websocket"

    def _payload(self, text, audio_format, sample_rate):
        container = {"pcm": "raw", "linear16": "raw"}.get(audio_format, audio_format)
        if container not in ("raw", "wav", "mp3", "ogg", "mulaw", "alaw"):
            raise ValueError(f"Unsupported Gnani audio format: {audio_format}")
        if sample_rate not in SAMPLE_RATES:
            raise ValueError(f"Unsupported Gnani sample rate: {sample_rate}")
        return {
            "text": text,
            "model": self.model,
            "voice": self.voice,
            "language": self.config.language,
            "speed": self.config.speed,
            "audio_config": {
                "container": container,
                "sample_rate": 8000 if container in ("mulaw", "alaw") else sample_rate,
                "encoding": "linear_pcm",
                "num_channels": 1,
                "sample_width": 2,
            },
        }

    def _headers(self):
        return {"X-API-Key-ID": self.api_key}

    async def _request_audio(self, text, audio_format, rate):
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=TIMEOUT)) as session:
            async with session.post(
                REST_URL,
                headers=self._headers(),
                json=self._payload(text, audio_format, rate),
                ssl=get_ssl_context(),
                allow_redirects=False,
            ) as response:
                response.raise_for_status()
                if response.status != 200 or not (
                    response.content_type.startswith("audio/") or response.content_type == "application/octet-stream"
                ):
                    raise ValueError(f"Unexpected Gnani audio response ({response.status}, {response.content_type})")
                audio = await response.read()
                if not audio:
                    raise ValueError("Gnani returned empty audio")
                return audio

    async def synthesize(self, text):
        return await self._request_audio(text, "wav", self.sampling_rate)

    async def synthesize_pcm_clip(self, text, sample_rate):
        return await self._request_audio(text, "pcm", sample_rate)

    async def synthesize_telephony_clip(self, text):
        return await self._request_audio(text, "mulaw", 8000)

    async def _generate_http(self, text):
        return await self._request_audio(text, self.audio_format, self.sampling_rate)

    async def _ws_chunks(self, payload):
        if self.websocket is None or self.websocket.state is not State.OPEN:
            self.websocket = await websockets.connect(
                WS_URL,
                additional_headers=self._headers(),
                ssl=get_ssl_context(WS_URL),
                open_timeout=10,
                close_timeout=2,
            )
        ws = self.websocket
        complete = False
        try:
            await ws.send(json.dumps(payload))
            while True:
                event = json.loads(await asyncio.wait_for(ws.recv(), TIMEOUT))
                kind = event.get("type")
                if kind == "error":
                    raise RuntimeError(f"Gnani synthesis failed: {event.get('message')}")
                # The service also sends a request-level acknowledgement after the final audio
                # frame. It can still be queued when the next utterance reuses this socket.
                if kind == "complete" and "data" not in event:
                    continue
                if kind in ("audio", "complete"):
                    data = event["data"]
                    if data.get("audio"):
                        yield base64.b64decode(data["audio"], validate=True)
                    if kind == "complete" or data.get("is_final"):
                        complete = True
                        return
        finally:
            # After an interruption that socket may still owe audio to the retired turn.
            if not complete and self.websocket is ws:
                await self.close_connection()

    async def _sse_chunks(self, payload):
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=10, sock_read=TIMEOUT)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                SSE_URL,
                headers=self._headers(),
                json=payload,
                ssl=get_ssl_context(),
                allow_redirects=False,
            ) as response:
                response.raise_for_status()
                if response.status != 200 or response.content_type != "text/event-stream":
                    raise ValueError(f"Unexpected Gnani stream response ({response.status}, {response.content_type})")
                event, lines = "", []
                async for raw in response.content:
                    line = raw.decode().rstrip("\r\n")
                    if line.startswith("event:"):
                        event = line[6:].strip()
                    elif line.startswith("data:"):
                        lines.append(line[5:].lstrip())
                    elif not line and lines:
                        data = json.loads("\n".join(lines))
                        if event == "error":
                            raise RuntimeError(f"Gnani synthesis failed: {data.get('message')}")
                        if event in ("chunk", "complete") and data.get("audio"):
                            yield base64.b64decode(data["audio"], validate=True)
                        if event == "complete" or data.get("is_final"):
                            return
                        event, lines = "", []
                raise ConnectionError("Gnani SSE stream closed before completion")

    async def push(self, message):
        meta = dict(message["meta_info"])
        text = message["data"] or ""
        if not self.stream:
            await super().push(create_ws_data_packet(text, meta))
            return
        if self._pending_text and self._pending_meta.get("sequence_id") != meta.get("sequence_id"):
            await super().push(create_ws_data_packet(self._pending_text, self._pending_meta))
            self._pending_text = ""
        self._pending_meta = meta
        joined = (self._pending_text + " " + text).strip()
        final = meta.get("end_of_llm_stream", False)
        # Timbre applies expressive tags to the rest of the request. Keep that span together.
        if final:
            units, self._pending_text = [joined], ""
        elif "<" in joined:
            units, self._pending_text = [], joined
        else:
            units, self._pending_text = split_flushable(joined, False)
        for unit in units:
            await super().push(create_ws_data_packet(unit, meta))

    async def _audio(self, text, meta):
        if not self.stream:
            yield await self._fetch_http_audio(text, meta)
            return
        self.synthesized_characters += len(text)
        payload = self._payload(text, self.audio_format, self.sampling_rate)
        chunks = self._ws_chunks(payload) if self.supports_websocket() else self._sse_chunks(payload)
        async with aclosing(chunks):
            async for audio in chunks:
                yield audio

    async def _next_chunk(self, stream):
        # Cancelling this read on barge-in releases the transport without cancelling the queue consumer.
        task = self._read_task = asyncio.create_task(anext(stream, None))
        try:
            await asyncio.wait([task])
            return None if task.cancelled() else task.result()
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            self._read_task = None

    async def generate(self):
        while True:
            message = await self.internal_queue.get()
            meta, text = dict(message["meta_info"]), message["data"]
            if self.task_manager_instance is not None and not self.should_synthesize_response(meta.get("sequence_id")):
                continue
            generation = self._generation
            final = meta.pop("end_of_llm_stream", False)
            meta.update(format=self.audio_format, text=text, text_synthesized="")
            last = None
            started = time.monotonic()
            if text and text.strip():
                async with aclosing(self._audio(text, meta)) as stream:
                    while generation == self._generation:
                        chunk = await self._next_chunk(stream)
                        if chunk is None or generation != self._generation:
                            break
                        if last is None:
                            meta["synthesizer_latency"] = time.monotonic() - started
                        else:
                            self._stamp_first_chunk(meta)
                            self._stamp_mark_id(meta)
                            yield create_ws_data_packet(last, meta)
                        last = chunk
                if generation == self._generation and last is None:
                    raise ValueError("Gnani completed synthesis without audio")
            if generation == self._generation and (last or final):
                # Playback acknowledgements concatenate these fragments across the turn.
                meta.update(text_synthesized=f"{text} " if text else "", end_of_llm_stream=final)
                self._stamp_first_chunk(meta)
                self._stamp_end_of_stream(meta)
                self._stamp_mark_id(meta)
                yield create_ws_data_packet(last or b"\x00", meta)

    async def handle_interruption(self):
        self._generation += 1
        self._pending_text = ""
        self.first_chunk_generated = False
        while not self.internal_queue.empty():
            self.internal_queue.get_nowait()
        if self._read_task is not None:
            self._read_task.cancel()
            await asyncio.gather(self._read_task, return_exceptions=True)

    async def close_connection(self):
        ws, self.websocket = self.websocket, None
        if ws is not None:
            await ws.close()

    async def cleanup(self):
        await self.handle_interruption()
        await self.close_connection()
