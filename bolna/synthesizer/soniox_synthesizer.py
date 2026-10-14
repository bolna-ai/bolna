import asyncio
import base64
import json
import os
import time
import uuid

import aiohttp
import websockets
from websockets.exceptions import InvalidHandshake

from .stream_synthesizer import StreamSynthesizer
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.ssl_context import get_ssl_context
from bolna.memory.cache.inmemory_scalar_cache import InmemoryScalarCache

logger = configure_logger(__name__)

SONIOX_TTS_HOST = "tts-rt.soniox.com"
SONIOX_TTS_DEFAULT_MODEL = "tts-rt-v2"

# 0.7–1.3 server-side; so a typo fails at agent setup, not mid-call.
SPEED_RANGE = (0.7, 1.3)

# pcm_mulaw/pcm_alaw are 8k-only; the linear formats accept any of these.
PCM_SAMPLE_RATES = (8000, 16000, 24000, 44100, 48000)

# Docs say 20-30s with no active stream and a drop at ~40s, but idle sockets go much sooner:
# an observed call lost one ~11s after connect.
KEEP_ALIVE_INTERVAL_S = 8
# Named rather than inlined into the sleep purely so tests can drive the loop.
KEEP_ALIVE_POLL_S = 1

# Text frames cap at 5000 characters. Buffered LLM chunks never approach this, but a
# pre-rendered clip (a long welcome message) can, so the one-shot path checks it.
MAX_TEXT_CHARS = 5000


class SonioxSynthesizer(StreamSynthesizer):
    def __init__(
        self,
        voice,
        voice_id=None,
        language="en",
        model=SONIOX_TTS_DEFAULT_MODEL,
        sampling_rate="24000",
        speed=None,
        reduce_silence=None,
        stream=False,
        buffer_size=400,
        caching=True,
        synthesizer_key=None,
        **kwargs,
    ):
        super().__init__(
            stream=stream,
            provider_name="soniox",
            buffer_size=buffer_size,
            **kwargs,
        )
        self.api_key = synthesizer_key or os.getenv("SONIOX_API_KEY")
        if not self.api_key:
            raise ValueError("Soniox API key is required, either as synthesizer_key or SONIOX_API_KEY")

        # Soniox addresses voices by name ("Adrian"); a cloned voice's id goes in the same
        # field, so voice_id wins when both are set.
        self.voice = voice_id or voice
        if not self.voice:
            raise ValueError("Soniox needs a voice name or a cloned voice id")

        self.caching = caching
        if caching:
            self.cache = InmemoryScalarCache()

        self.model = model
        self.language = (language or "en").split("-")[0]
        self.speed = float(speed) if speed is not None else None
        self.reduce_silence = reduce_silence

        self.use_mulaw = kwargs.get("use_mulaw", False)
        self.target_sample_rate = 8000 if self.use_mulaw else int(sampling_rate or 24000)
        self.sampling_rate = self.target_sample_rate

        self.host = os.getenv("SONIOX_TTS_HOST", SONIOX_TTS_HOST)
        protocol = os.getenv("SONIOX_TTS_HOST_PROTOCOL", "wss")
        self.ws_url = f"{protocol}://{self.host}/tts-websocket"
        self.api_url = f"https://{self.host}/tts"

        self._validate_options()

        # Per-turn stream bookkeeping. A stream_id may only be reused once the server has
        # confirmed `terminated`, so every turn mints a new one.
        self.stream_id = None
        self._stream_open = False
        # Sequence the open stream belongs to. A frame whose sequence_id no longer matches
        # belongs to a turn that has already been retired.
        self._turn_seq = None
        self._cancelled_streams = set()
        self._pending_cancels = []
        self._keepalive_task = None
        self._last_send_time = time.perf_counter()

    # ------------------------------------------------------------------
    # Config validation
    # ------------------------------------------------------------------

    def _validate_options(self):
        if self.speed is not None and not SPEED_RANGE[0] <= self.speed <= SPEED_RANGE[1]:
            raise ValueError(f"Soniox speed must be between {SPEED_RANGE[0]} and {SPEED_RANGE[1]}")
        # use_mulaw already pinned the rate to 8000 (pcm_mulaw is 8k-only), overriding any
        # configured sampling_rate, so only the linear path can carry a bad rate here.
        if not self.use_mulaw and self.target_sample_rate not in PCM_SAMPLE_RATES:
            raise ValueError(f"Soniox sample_rate must be one of {PCM_SAMPLE_RATES}")

    def get_sleep_time(self):
        return 0.01

    # ------------------------------------------------------------------
    # Audio plumbing
    # ------------------------------------------------------------------

    def _wire_audio_format(self):
        """Ask for exactly what the output leg needs, so no in-process resampling is wanted."""
        return "pcm_mulaw" if self.use_mulaw else "pcm_s16le"

    def _get_audio_format(self):
        return "mulaw" if self.use_mulaw else "pcm"

    def _process_audio_chunk(self, chunk):
        # The wire format is already the target format — decoding base64 is the whole job.
        return chunk or None

    def _get_http_audio_format(self):
        # The one-shot path asks for wav so its header can describe itself to callers that
        # convert (see synthesize()).
        return "wav"

    def _process_http_audio(self, audio):
        return audio or None

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def _stream_config(self, stream_id):
        """The first frame of a stream: credentials and render settings together."""
        config = {
            "api_key": self.api_key,
            "stream_id": stream_id,
            "model": self.model,
            "language": self.language,
            "voice": self.voice,
            "audio_format": self._wire_audio_format(),
            "sample_rate": self.target_sample_rate,
        }
        if self.speed is not None:
            config["speed"] = self.speed
        if self.reduce_silence is not None:
            config["reduce_silence"] = bool(self.reduce_silence)
        return config

    async def establish_connection(self):
        try:
            start = time.perf_counter()
            ws = await asyncio.wait_for(
                websockets.connect(self.ws_url, ssl=get_ssl_context(self.ws_url)),
                timeout=10.0,
            )
            elapsed = round((time.perf_counter() - start) * 1000)
            if not self.connection_time:
                self.connection_time = elapsed
            # A live socket retires whatever the last one died of. A fatal handshake
            # rejection returns before this, so it cannot clear one that must stop the call.
            self.connection_error = None
            # Nothing is sent yet: the config frame opens a *stream*, not the connection, and
            # the first turn's sender emits it.
            self.stream_id = None
            self._stream_open = False
            self._turn_seq = None
            self._cancelled_streams.clear()
            self._pending_cancels.clear()
            logger.info(
                f"Connected to Soniox TTS in {elapsed}ms "
                f"(model={self.model}, voice={self.voice}, "
                f"format={self._wire_audio_format()}@{self.target_sample_rate})"
            )
            if self._keepalive_task is None or self._keepalive_task.done():
                self._keepalive_task = asyncio.create_task(self._keepalive_loop())
            return ws
        except asyncio.TimeoutError:
            logger.error("Timeout while connecting to Soniox TTS websocket")
        except InvalidHandshake as e:
            error_msg = str(e)
            if "401" in error_msg or "403" in error_msg:
                logger.error(f"Soniox TTS authentication failed: {e}")
            elif "404" in error_msg:
                logger.error(f"Soniox TTS endpoint not found: {e}")
            else:
                logger.error(f"Soniox TTS handshake failed: {e}")
            # A handshake rejection will not fix itself on retry, so it is surfaced rather
            # than left to monitor_connection's failure budget.
            self.connection_error = str(e)
        except Exception as e:
            logger.error(f"Failed to connect to Soniox TTS: {e}")
        return None

    async def _keepalive_loop(self):
        """Hold the idle connection open between turns."""
        while not self.conversation_ended and not self.connection_error:
            await asyncio.sleep(KEEP_ALIVE_POLL_S)
            # Started before monitor_connection assigns the socket, so wait the gap out
            # rather than exiting: this task is what keeps the connection alive.
            if self._stream_open or not self._is_ws_connected():
                continue
            if time.perf_counter() - self._last_send_time < KEEP_ALIVE_INTERVAL_S:
                continue
            await self._send_frame({"keep_alive": True})

    async def _send_frame(self, payload):
        """Send one frame; the caller settles the turn on a False.

        Not the shared _send_json: that records connection_error, which _generate_ws_loop
        treats as fatal and never clears. A socket dying mid-send is transient — the redial
        settles it (cf. kalpa _send_frame)."""
        if not self._is_ws_connected():
            logger.info("Soniox TTS websocket is not connected; dropping frame")
            return False
        try:
            await self.websocket.send(json.dumps(payload))
            self._last_send_time = time.perf_counter()
            return True
        except Exception as e:
            logger.error(f"Soniox TTS send failed; the redial settles it: {e}")
            return False

    # ------------------------------------------------------------------
    # Turn lifecycle
    # ------------------------------------------------------------------

    def _on_push(self, meta_info, text):
        """Runs in push order, before the sender task. A new sequence arriving while a stream
        is still open means the previous turn was retired without an explicit interruption;
        its stream is abandoned so its text cannot bleed into this turn."""
        seq = meta_info.get("sequence_id")
        if self._stream_open and self._turn_seq is not None and self._turn_seq != seq:
            logger.info(f"Soniox stream {self.stream_id} superseded by seq={seq}; abandoning it")
            if self.stream_id:
                self._cancelled_streams.add(self.stream_id)
                self._pending_cancels.append(self.stream_id)
            self._stream_open = False
            self.stream_id = None

    async def sender(self, text, sequence_id, end_of_llm_stream=False):
        try:
            if self.conversation_ended:
                return
            if not self.should_synthesize_response(sequence_id):
                logger.info(f"Not synthesizing: sequence_id {sequence_id} not current")
                return

            await self._wait_for_ws()
            if not self._is_ws_connected():
                return

            while self._pending_cancels:
                await self._send_frame({"stream_id": self._pending_cancels.pop(0), "cancel": True})

            if not self._stream_open:
                self.stream_id = f"{sequence_id}-{uuid.uuid4().hex[:8]}"
                self._turn_seq = sequence_id
                if not await self._send_frame(self._stream_config(self.stream_id)):
                    return
                self._stream_open = True
                self.first_chunk_generated = False
                logger.info(f"Opened Soniox TTS stream {self.stream_id} for seq={sequence_id}")

            if self.ws_send_time is None:
                self.ws_send_time = time.perf_counter()

            frame = {"stream_id": self.stream_id, "text": text or "", "text_end": bool(end_of_llm_stream)}
            if not await self._send_frame(frame):
                return

            if end_of_llm_stream:
                self.last_text_sent = True
                logger.info(f"Closed text window on Soniox TTS stream {self.stream_id}")

        except asyncio.CancelledError:
            logger.info("Soniox TTS sender task was cancelled.")
        except Exception as e:
            logger.error(f"Unexpected error in Soniox TTS sender: {e}")

    async def handle_interruption(self):
        """Cancel the open stream. Soniox sends no further audio after a cancel, so unlike
        providers that keep flushing, there are no in-flight frames left to filter — but the
        stream_id is still quarantined until its `terminated` arrives."""
        if not self._stream_open or not self.stream_id:
            return
        cancelled = self.stream_id
        self._cancelled_streams.add(cancelled)
        self._stream_open = False
        self.stream_id = None
        self._turn_seq = None
        self.first_chunk_generated = False
        self.current_turn_start_time = None
        await self._send_frame({"stream_id": cancelled, "cancel": True})
        logger.info(f"Cancelled Soniox TTS stream {cancelled}")

    # ------------------------------------------------------------------
    # Receiver
    # ------------------------------------------------------------------

    async def receiver(self):
        while True:
            if not self._is_ws_connected():
                await asyncio.sleep(0.05)
                continue
            try:
                message = await self.websocket.recv()
            except Exception as e:
                logger.error(f"Soniox TTS receiver lost the socket: {e}")
                self._stream_open = False
                await asyncio.sleep(0.05)
                continue

            try:
                event = json.loads(message)
            except (TypeError, ValueError):
                logger.error("Soniox TTS sent a non-JSON frame; ignoring")
                continue

            stream_id = event.get("stream_id")
            stale = stream_id in self._cancelled_streams

            if event.get("error_type") or event.get("error_code"):
                logger.error(
                    f"Soniox TTS error on stream={stream_id}: "
                    f"{event.get('error_type')} {event.get('error_message')} "
                    f"(code={event.get('error_code')} request_id={event.get('request_id')})"
                )

                if not stale and self.stream_id is not None and stream_id == self.stream_id:
                    self._stream_open = False
                    self.stream_id = None
                    yield b"\x00"
                continue

            if event.get("terminated"):
                self._cancelled_streams.discard(stream_id)
                if stream_id == self.stream_id:
                    self._stream_open = False
                    self.stream_id = None
                continue

            audio_b64 = event.get("audio")
            if audio_b64:
                if stale:
                    continue  # belongs to a cancelled or superseded turn
                try:
                    chunk = base64.b64decode(audio_b64)
                except Exception as e:
                    logger.error(f"Could not decode Soniox TTS audio: {e}")
                    continue
                if chunk:
                    yield chunk

            if event.get("audio_end"):
                if not stale:
                    yield b"\x00"

    # ------------------------------------------------------------------
    # One-shot HTTP
    # ------------------------------------------------------------------

    async def _generate_http(self, text, audio_format="wav", sample_rate=None):
        if not text:
            return None
        if len(text) > MAX_TEXT_CHARS:
            logger.error(f"Soniox TTS rejects text over {MAX_TEXT_CHARS} chars; got {len(text)}")
            return None

        payload = {
            "model": self.model,
            "language": self.language,
            "voice": self.voice,
            "audio_format": audio_format,
            "text": text,
        }
        if sample_rate:
            payload["sample_rate"] = sample_rate
        if self.speed is not None:
            payload["speed"] = self.speed
        if self.reduce_silence is not None:
            payload["reduce_silence"] = bool(self.reduce_silence)

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.api_url, headers=headers, json=payload) as response:
                    if response.status != 200:
                        body = await response.text()
                        logger.error(f"Soniox TTS HTTP {response.status}: {body[:300]}")
                        return None
                    return await response.read()
        except Exception as e:
            logger.error(f"Soniox TTS HTTP request failed: {e}")
            return None

    async def synthesize(self, text):
        """One-shot render for prewarm/handoff clips. WAV, so its header describes the rate to
        callers that convert (a headerless PCM body gets guessed wrong)."""
        return await self._generate_http(text)

    async def synthesize_pcm_clip(self, text, sample_rate):
        """Native linear PCM at an arbitrary supported rate — no decode step for the caller."""
        if sample_rate not in PCM_SAMPLE_RATES:
            return None
        return await self._generate_http(text, audio_format="pcm_s16le", sample_rate=sample_rate)

    async def synthesize_telephony_clip(self, text):
        """Mu-law 8000 straight from the API, skipping the pydub/ffmpeg decode the base class
        would otherwise need. None on non-telephony configs."""
        if not self.use_mulaw:
            return None
        return await self._generate_http(text, audio_format="pcm_mulaw", sample_rate=8000)

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    async def cleanup(self):
        if self._keepalive_task is not None:
            self._keepalive_task.cancel()
            self._keepalive_task = None
        self._stream_open = False
        self.stream_id = None
        self._cancelled_streams.clear()
        await super().cleanup()
