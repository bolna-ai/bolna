"""Gnani Prisma speech recognition for live audio and recorded turns."""

import asyncio
import audioop
import io
import json
import math
import os
import time
import wave

import aiohttp
import websockets

from bolna.enums import TelephonyProvider
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.ssl_context import get_ssl_context
from bolna.helpers.utils import create_ws_data_packet

from .base_transcriber import BaseTranscriber

logger = configure_logger(__name__)
GNANI_DEFAULT_MODEL = "prisma-v2.5"
GNANI_MODELS = (GNANI_DEFAULT_MODEL, "prisma-v3.0")
LOCALES = {"bn-IN", "en-IN", "gu-IN", "hi-IN", "kn-IN", "ml-IN", "mr-IN", "pa-IN", "ta-IN", "te-IN"}
STREAM_URL = "wss://api.vachana.ai/stt/v3/stream"
REST_URL = "https://api.vachana.ai/stt/v3"
WIRE_RATE = 16000
FRAME_BYTES = 1024


def validate_prisma_config(model, language, *, stream=True):
    if model not in GNANI_MODELS:
        raise ValueError("Gnani model must be prisma-v2.5 or prisma-v3.0")
    if language == "indic-auto" and (stream or model == "prisma-v3.0"):
        return
    locales = language.split(",")
    limit = 4 if model == "prisma-v3.0" else 1
    if len(locales) > limit or not all(locale in LOCALES for locale in locales):
        raise ValueError(f"Invalid Gnani language for {model}: {language!r}")


class GnaniTranscriber(BaseTranscriber):
    def __init__(
        self,
        telephony_provider,
        input_queue=None,
        output_queue=None,
        model=GNANI_DEFAULT_MODEL,
        language="en-IN",
        stream=True,
        sampling_rate=16000,
        encoding="linear16",
        **kwargs,
    ):
        super().__init__(input_queue)
        self.model = model
        self.language = language
        self.stream = stream
        validate_prisma_config(model, language, stream=stream)
        if telephony_provider in TelephonyProvider.telephony_values():
            sampling_rate = 8000
            encoding = "mulaw" if telephony_provider in TelephonyProvider.mulaw_values() else "linear16"
        self.sampling_rate = int(sampling_rate)
        if encoding not in ("mulaw", "linear16"):
            raise ValueError(f"Unsupported Gnani input encoding: {encoding}")
        if self.sampling_rate <= 0:
            raise ValueError(f"Unsupported Gnani sample rate: {sampling_rate}")
        self.encoding = encoding
        self.api_key = kwargs.get("transcriber_key") or os.getenv("GNANI_API_KEY")
        self.transcriber_output_queue = output_queue
        self.transcription_task = None
        self.meta_info = {}
        self.input_meta = {}
        self.turn_counter = 0
        self.open_turns = []
        self.delivered = set()
        self.drained = asyncio.Event()

    def get_meta_info(self):
        return self.meta_info

    def _auth_headers(self):
        if not self.api_key:
            raise ValueError("Gnani requires a transcriber_key or GNANI_API_KEY")
        return {"x-api-key-id": self.api_key}

    def _pcm(self, data):
        return audioop.ulaw2lin(data, 2) if self.encoding == "mulaw" else data

    def _recording(self, data):
        if data.startswith(b"RIFF") and data[8:12] == b"WAVE":
            with wave.open(io.BytesIO(data), "rb") as wav:
                duration = wav.getnframes() / wav.getframerate()
        else:
            pcm = self._pcm(data)
            if len(pcm) % 2:
                raise ValueError("Incomplete PCM sample in Gnani recording")
            duration = len(pcm) / (2 * self.sampling_rate)
            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as wav:
                wav.setparams((1, 2, self.sampling_rate, 0, "NONE", "not compressed"))
                wav.writeframes(pcm)
            data = buffer.getvalue()
        if not 0 < duration <= 30:
            raise ValueError("Gnani recorded turns must contain up to 30 seconds of audio")
        return data, duration

    async def _post_recording(self, session, data, meta):
        wav, duration = self._recording(data)
        form = aiohttp.FormData()
        form.add_field("audio_file", wav, filename="turn.wav", content_type="audio/wav")
        form.add_field("model", self.model)
        form.add_field("language_code", self.language)
        started = time.monotonic()
        async with session.post(
            REST_URL, data=form, headers=self._auth_headers(), ssl=get_ssl_context(), allow_redirects=False
        ) as response:
            response.raise_for_status()
            result = await response.json()
        if not result.get("success") or not isinstance(result.get("transcript"), str):
            raise ValueError("Gnani returned an invalid transcription response")
        elapsed = time.monotonic() - started
        meta = dict(
            meta, request_id=result.get("request_id"), transcriber_duration=duration, transcriber_latency=elapsed
        )
        if result.get("detected_language_code"):
            meta["detected_language_code"] = result["detected_language_code"]
        return create_ws_data_packet(result["transcript"].strip(), meta)

    async def transcribe_recording(self, data):
        """Transcribe one WAV or one turn of configured raw audio and return a Bolna data packet."""
        validate_prisma_config(self.model, self.language, stream=False)
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=45)) as session:
            return await self._post_recording(session, data, {})

    async def _recorded_turns(self):
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=45)) as session:
            while True:
                packet = await self.input_queue.get()
                self.meta_info = packet.get("meta_info") or {}
                if self.meta_info.get("eos"):
                    return
                result = await self._post_recording(session, packet["data"], self.meta_info)
                await self.transcriber_output_queue.put(result)

    async def _frames(self):
        buffer = bytearray()
        carry = b""
        resampler = None
        while True:
            packet = await self.input_queue.get()
            meta = packet.get("meta_info") or {}
            if meta.get("eos"):
                break
            data = packet.get("data")
            if not data:
                continue
            self.input_meta = meta
            pcm = carry + self._pcm(data)
            complete = len(pcm) - len(pcm) % 2
            pcm, carry = pcm[:complete], pcm[complete:]
            if self.sampling_rate != WIRE_RATE:
                # The service can announce a 16 kHz session even for an 8 kHz handshake.
                pcm, resampler = audioop.ratecv(pcm, 2, 1, self.sampling_rate, WIRE_RATE, resampler)
            buffer.extend(pcm)
            offset = 0
            while len(buffer) - offset >= FRAME_BYTES:
                yield bytes(buffer[offset : offset + FRAME_BYTES])
                offset += FRAME_BYTES
            del buffer[:offset]
        if carry:
            raise ValueError("Incomplete PCM sample at end of Gnani stream")
        if buffer:
            yield bytes(buffer).ljust(FRAME_BYTES, b"\0")
        if self.num_frames:
            # Gnani finalizes on silence; a WebSocket close would discard trailing speech.
            for _ in range(math.ceil(WIRE_RATE / 512)):
                yield bytes(FRAME_BYTES)

    async def _send(self, ws):
        loop = asyncio.get_running_loop()
        frame_seconds = 512 / WIRE_RATE
        deadline = loop.time()
        async for frame in self._frames():
            await asyncio.sleep(max(0, deadline - loop.time()))
            deadline = max(deadline, loop.time()) + frame_seconds
            await ws.send(frame)
            self.record_audio_frame(frame_seconds, time.time() * 1000)
            self.count_audio_sent(len(frame), WIRE_RATE)
        # Allow the last VAD event to arrive even if the connection had no open turn at EOS.
        if self.num_frames:
            await asyncio.sleep(0.5)
            await asyncio.wait_for(self.drained.wait(), timeout=8)

    def _start_turn(self, event):
        self.turn_counter += 1
        self.previous_request_id, self.current_request_id = self.current_request_id, self.generate_request_id()
        meta = dict(
            self.input_meta,
            request_id=self.current_request_id,
            previous_request_id=self.previous_request_id,
            asr_turn_id=self.turn_counter,
            origin="transcriber",
            user_start_ts_wall=time.time(),
        )
        self.meta_info = meta
        self.open_turns.append((event.get("utterance_id"), meta))
        self.drained.clear()
        return meta

    def _turn_for_event(self, event):
        # Older events omit the utterance id. Their segments are reported in order.
        utterance = event.get("utterance_id")
        for uid, meta in self.open_turns:
            if uid == utterance or uid is None or utterance is None:
                if event["type"] != "processing" or "user_stop_ts_wall" not in meta:
                    return meta
        return self._start_turn(event)

    def _finish_turn(self, event):
        meta = self._turn_for_event(event)
        self.open_turns = [(uid, turn) for uid, turn in self.open_turns if turn is not meta]
        if not self.open_turns:
            self.drained.set()
        return meta

    async def _receive(self, ws):
        async for raw in ws:
            event = json.loads(raw)
            kind = event.get("type")
            if kind == "speech_start":
                meta = self._start_turn(event)
                await self.transcriber_output_queue.put(create_ws_data_packet("speech_started", meta))
            elif kind == "processing":
                self._turn_for_event(event)["user_stop_ts_wall"] = time.time()
            elif kind == "transcript" and event.get("is_final", True):
                segment = event.get("segment_id") or event.get("utterance_id")
                if segment and segment in self.delivered:
                    continue
                if segment:
                    self.delivered.add(segment)
                meta = self._finish_turn(event)
                text = event["text"].strip()
                if event.get("detected_language_code"):
                    meta["detected_language_code"] = event["detected_language_code"]
                now = time.time()
                if "user_stop_ts_wall" in meta:
                    meta["transcriber_latency"] = now - meta["user_stop_ts_wall"]
                elif event.get("latency") is not None:
                    meta["transcriber_latency"] = event["latency"] / 1000
                self.meta_info = meta
                self._upsert_turn_latency(
                    {
                        "turn_id": meta["asr_turn_id"],
                        "asr_turn_start_epoch_ms": meta["user_start_ts_wall"] * 1000,
                        "asr_finalized_epoch_ms": now * 1000,
                        "final_transcript": text,
                        "user_speech_end_epoch_ms": meta["user_stop_ts_wall"] * 1000
                        if "user_stop_ts_wall" in meta
                        else None,
                    }
                )
                data = {"type": "transcript", "content": text} if text else {"type": "speech_ended"}
                await self.transcriber_output_queue.put(create_ws_data_packet(data, meta))
            elif kind == "error":
                # A single utterance can fail while the session remains usable.
                logger.warning("Gnani transcription error: %s", event.get("message"))
                if self.open_turns:
                    meta = self._finish_turn(event)
                    data = {"type": "speech_ended", "error": event.get("message")}
                    await self.transcriber_output_queue.put(create_ws_data_packet(data, meta))

    async def _stream(self):
        headers = {
            **self._auth_headers(),
            "model": self.model,
            "language_code": self.language,
            "x-sample-rate": str(WIRE_RATE),
        }
        if "," in self.language:
            # The streaming API accepts one locale or indic-auto and separate candidate hints.
            headers["language_code"] = "indic-auto"
            headers["x-language-candidates"] = self.language
        started = time.monotonic()
        async with websockets.connect(
            STREAM_URL, additional_headers=headers, ssl=get_ssl_context(STREAM_URL), open_timeout=10, close_timeout=2
        ) as ws:
            ready = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
            if ready.get("type") != "connected":
                raise ConnectionError(ready.get("message", "Gnani session failed to start"))
            config = ready.get("config", {})
            if config.get("sample_rate") != WIRE_RATE or config.get("chunk_size") != FRAME_BYTES // 2:
                raise ConnectionError("Gnani negotiated an unexpected PCM format")
            self.connection_time = round((time.monotonic() - started) * 1000)
            sender = asyncio.create_task(self._send(ws))
            receiver = asyncio.create_task(self._receive(ws))
            try:
                done, _ = await asyncio.wait([sender, receiver], return_when=asyncio.FIRST_COMPLETED)
                if receiver in done:
                    await receiver
                    raise ConnectionError(f"Gnani closed the stream ({ws.close_code}): {ws.close_reason}")
                await sender
            finally:
                for task in (sender, receiver):
                    task.cancel()
                await asyncio.gather(sender, receiver, return_exceptions=True)

    async def transcribe(self):
        self.reset_audio_frame_state()
        self.delivered.clear()
        self.open_turns.clear()
        self.meta_info = {}
        self.input_meta = {}
        self.drained.set()
        self.connection_error = None
        try:
            if self.stream:
                await self._stream()
            else:
                await self._recorded_turns()
        except Exception as exc:
            self.connection_error = str(exc) or type(exc).__name__
            logger.error("Gnani transcription failed: %s", self.connection_error)
        finally:
            await self.transcriber_output_queue.put(
                create_ws_data_packet("transcriber_connection_closed", self.closing_meta())
            )

    async def run(self):
        self.transcription_task = asyncio.create_task(self.transcribe())

    async def toggle_connection(self):
        self.connection_on = False
        if self.transcription_task and not self.transcription_task.done():
            self.transcription_task.cancel()
            await asyncio.gather(self.transcription_task, return_exceptions=True)

    async def cleanup(self):
        await self.toggle_connection()
