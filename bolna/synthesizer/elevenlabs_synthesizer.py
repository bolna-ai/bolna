import asyncio
import json
import os
import re
import time
import uuid
from websockets.exceptions import InvalidHandshake
import base64

import aiohttp
import websockets

from .stream_synthesizer import StreamSynthesizer
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.ssl_context import get_ssl_context
from bolna.helpers.utils import create_ws_data_packet, resample
from bolna.memory.cache.inmemory_scalar_cache import InmemoryScalarCache

logger = configure_logger(__name__)


class ElevenlabsBase(StreamSynthesizer):
    """Credentials, wire format and HTTP synthesis shared by the ElevenLabs synthesizers."""

    def __init__(
        self,
        voice,
        voice_id,
        model="eleven_turbo_v2_5",
        audio_format="mp3",
        sampling_rate="16000",
        stream=False,
        buffer_size=400,
        temperature=0.5,
        similarity_boost=0.75,
        speed=1.0,
        style=0,
        synthesizer_key=None,
        caching=True,
        **kwargs,
    ):
        super().__init__(
            stream=True,  # ElevenLabs always streams
            provider_name="elevenlabs",
            buffer_size=buffer_size,
            **kwargs,
        )
        self.api_key = os.environ["ELEVENLABS_API_KEY"] if synthesizer_key is None else synthesizer_key
        self.voice = voice_id
        self.model = model
        self.stream = True
        self.sampling_rate = sampling_rate
        self.speed = speed
        self.style = style
        self.audio_format = "mp3"
        self.use_mulaw = kwargs.get("use_mulaw", True)
        self.temperature = temperature
        self.similarity_boost = similarity_boost
        self.caching = caching
        if self.caching:
            self.cache = InmemoryScalarCache()

        self.elevenlabs_host = os.getenv("ELEVENLABS_API_HOST", "api.elevenlabs.io")
        # Set from the x-trace-id response header on connect; both sockets log against it.
        self.ws_trace_id = None
        if self.use_mulaw:
            self.wire_format = "ulaw_8000"
            self.wire_pcm_rate = None
        else:
            # Raw PCM on the wire: ElevenLabs WS chunks are not cut on MP3 frame boundaries,
            # so decoding each chunk as standalone MP3 (pydub/ffmpeg) crashes mid-stream and
            # kills the synth loop. PCM at the target rate needs no decode and usually no
            # resample — same pattern as cartesia/sarvam on the web/freeswitch path.
            rate = int(self.sampling_rate)
            self.wire_pcm_rate = rate if rate in (16000, 22050, 24000, 44100) else 24000
            self.wire_format = f"pcm_{self.wire_pcm_rate}"

    def _get_audio_format(self):
        return "mulaw" if self.wire_format == "ulaw_8000" else "wav"

    def _process_audio_chunk(self, chunk):
        # ulaw_8000 arrives ready to use; pcm needs at most a rate conversion (a passthrough
        # when wire_pcm_rate == sampling_rate, which is the normal web/freeswitch case)
        if self.wire_format == "ulaw_8000":
            return chunk
        return resample(chunk, int(self.sampling_rate), format="pcm", original_sample_rate=self.wire_pcm_rate)

    def _unpack_receiver_message(self, item):
        """ElevenLabs receiver yields (audio, text_synthesized) tuples."""
        audio, text_synthesized = item
        return audio, {"text_synthesized": text_synthesized}

    def _get_output_format(self):
        return self.wire_format

    async def synthesize(self, text):
        return await self._generate_http(text, format="mp3_44100_128")

    async def synthesize_pcm_clip(self, text, sample_rate):
        # Natively rendered rates only.
        if int(sample_rate) not in (16000, 22050, 24000, 44100):
            return None
        return await self._generate_http(text, format=f"pcm_{int(sample_rate)}")

    async def synthesize_telephony_clip(self, text):
        """One-shot render in the telephony wire format (mu-law 8000) — no
        decode/transcode step (and no ffmpeg), unlike the MP3 the plain synthesize()
        returns. None on non-mulaw configs so callers fall back to synthesize()."""
        if not self.use_mulaw:
            return None
        return await self._generate_http(text)

    async def _generate_http(self, text, format=None):
        payload = {
            "text": text,
            "model_id": self.model,
            "voice_settings": {
                "stability": self.temperature,
                "similarity_boost": self.similarity_boost,
                "optimize_streaming_latency": 3,
                "speed": self.speed,
                "style": self.style,
            },
        }
        headers = {"xi-api-key": self.api_key}
        fmt = format or self._get_output_format()
        url = f"{self.api_url}{fmt}"
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    return await response.read()
                else:
                    logger.error(f"Error: {response.status} - {await response.text()}")
                    return None


class ElevenlabsSynthesizer(ElevenlabsBase):
    """Eleven v1/v2 models over the multi-stream-input socket, one context per turn."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ws_url = (
            f"wss://{self.elevenlabs_host}/v1/text-to-speech/{self.voice}/multi-stream-input"
            f"?model_id={self.model}&output_format={self.wire_format}"
            f"&inactivity_timeout=170&sync_alignment=true&optimize_streaming_latency=4"
        )
        self.api_url = f"https://{self.elevenlabs_host}/v1/text-to-speech/{self.voice}/stream?optimize_streaming_latency=2&output_format="

        # One context per turn: closed at end_of_llm_stream and on interruption, so each
        # turn gets a fresh context_id. Closing frees the slot, staying well under
        # ElevenLabs' 5-concurrent-context cap.
        self.context_id = None
        self.context_ids_to_ignore = set()
        self._eos_context_id = None  # context the last end-of-stream was emitted for
        self.current_turn_context_id = None  # survives close_context, unlike context_id
        self.ws_send_time = None
        self.current_turn_ttfb = None
        self.eos_accum_context_id = None  # context whose spoken chars are being accumulated
        self.eos_accum_text = ""  # spoken-so-far for that context (end-of-stream match)

    def _on_push(self, meta_info, text):
        # Mint only for pushes that will actually synthesize — a superseded push must not
        # advance current_turn_context_id, or the prior turn's real isFinal gets suppressed.
        if not self.context_id and self.should_synthesize_response(meta_info.get("sequence_id")):
            self.context_id = str(uuid.uuid4())
            self.current_turn_context_id = self.context_id

    async def handle_interruption(self):
        try:
            # Also covers a context already closed at end_of_llm_stream but still draining frames.
            if self.current_turn_context_id:
                self.context_ids_to_ignore.add(self.current_turn_context_id)
                self.current_turn_context_id = None
                # The interrupted context's end-of-stream is now dropped, so the
                # next turn must be re-detected as new to clear stale queue entries.
                self.current_turn_start_time = None
            if self.context_id:
                self.context_ids_to_ignore.add(self.context_id)
                interrupt_message = {"context_id": self.context_id, "close_context": True}
                await self.websocket.send(json.dumps(interrupt_message))
                self.context_id = None
        except Exception:
            pass

    async def sender(self, text, sequence_id, end_of_llm_stream=False):
        try:
            if self.conversation_ended:
                return
            if not self.should_synthesize_response(sequence_id):
                logger.info(f"Not synthesizing: sequence_id {sequence_id} not current")
                await self.flush_synthesizer_stream()
                return

            await self._wait_for_ws()

            if text != "":
                for text_chunk in self.text_chunker(text):
                    if not self.should_synthesize_response(sequence_id):
                        logger.info(f"Not synthesizing (inner): sequence_id {sequence_id} not current")
                        await self.flush_synthesizer_stream()
                        return
                    try:
                        if self.ws_send_time is None:
                            self.ws_send_time = time.perf_counter()
                            logger.info(f"WS send trace_id={self.ws_trace_id} first_text_sent")
                        await self.websocket.send(json.dumps({"text": text_chunk, "context_id": self.context_id}))
                    except Exception as e:
                        logger.info(f"Error sending chunk: {e}")
                        self.connection_error = str(e)
                        return

            if end_of_llm_stream:
                self.last_text_sent = True
                try:
                    await self.websocket.send(json.dumps({"text": "", "context_id": self.context_id, "flush": True}))
                    # Closing the context makes ElevenLabs emit isFinal, which the receiver uses
                    # as end-of-stream. The context's remaining frames are still delivered. The
                    # next turn opens a fresh context; voice_settings carry over from the connection BOS.
                    if self.context_id:
                        await self.websocket.send(json.dumps({"context_id": self.context_id, "close_context": True}))
                        self.context_id = None
                except Exception as e:
                    logger.info(f"Error sending end-of-stream signal: {e}")
                    self.connection_error = str(e)

        except asyncio.CancelledError:
            logger.info("Sender task was cancelled.")
        except Exception as e:
            logger.error(f"Unexpected error in sender: {e}")

    async def receiver(self):
        """Yields (audio_chunk, text_spoken) tuples, or (b'\\x00', '') for end-of-stream."""
        audio_chunk_count = 0
        last_recv_time = None
        not_connected_since = None
        # Bail out instead of busy-spinning on a persistent recv() error. A stuck
        # recv (e.g. "cannot call recv while another coroutine is already running
        # recv" when two coroutines race the same websocket) would otherwise loop
        # here every few ms with no sleep, flooding logs and starving the event
        # loop so the call never tears down.
        consecutive_errors = 0
        max_consecutive_errors = 10
        while True:
            try:
                if self.conversation_ended:
                    return
                if not self._is_ws_connected():
                    if self.connection_error:
                        return
                    now = time.perf_counter()
                    if not_connected_since is None:
                        not_connected_since = now
                    elif now - not_connected_since > 30:
                        logger.error("ElevenLabs receiver: WebSocket never connected after 30s, giving up.")
                        self.connection_error = self.connection_error or "WebSocket never connected"
                        return
                    logger.info("WebSocket is not connected, skipping receive.")
                    await asyncio.sleep(0.10)
                    continue
                else:
                    not_connected_since = None

                recv_start = time.perf_counter()
                response = await self.websocket.recv()
                recv_duration = (time.perf_counter() - recv_start) * 1000
                data = json.loads(response)
                consecutive_errors = 0  # successful recv — reset the error backoff

                ctx = data.get("contextId")
                if ctx in self.context_ids_to_ignore:
                    continue

                if "audio" in data and data["audio"] and self.ws_send_time is not None:
                    audio_chunk_count += 1
                    if audio_chunk_count == 1:
                        time_since_send = (time.perf_counter() - self.ws_send_time) * 1000
                        logger.info(
                            f"WS recv FIRST trace_id={self.ws_trace_id} recv_wait={recv_duration:.0f}ms time_since_send={time_since_send:.0f}ms"
                        )
                    elif recv_duration > 200:
                        gap = (recv_start - last_recv_time) * 1000 if last_recv_time else 0
                        logger.info(
                            f"WS recv SLOW chunk={audio_chunk_count} trace_id={self.ws_trace_id} recv_wait={recv_duration:.0f}ms gap={gap:.0f}ms"
                        )
                    last_recv_time = time.perf_counter()

                logger.info("response for isFinal: {}".format(data.get("isFinal", False)))

                if "audio" in data and data["audio"]:
                    chunk = base64.b64decode(data["audio"])
                    try:
                        text_spoken = "".join(data.get("alignment", {}).get("chars", []))
                    except Exception:
                        text_spoken = ""
                    # Accumulate spoken text per context for the end-of-stream match below.
                    if ctx != self.eos_accum_context_id:
                        self.eos_accum_context_id = ctx
                        self.eos_accum_text = ""
                    self.eos_accum_text += text_spoken
                    yield chunk, text_spoken

                emit_eos = False
                if "isFinal" in data and data["isFinal"]:
                    logger.info(f"WS recv isFinal trace_id={self.ws_trace_id}")
                    audio_chunk_count = 0
                    last_recv_time = None
                    emit_eos = True

                elif self.last_text_sent:
                    try:
                        current_norm = self.normalize_text(self.current_text.strip()).replace('"', "").strip()
                        spoken_norm = self.normalize_text(self.eos_accum_text.strip()).replace('"', "").strip()
                        # Strip whitespace before compare: ElevenLabs alignment splits
                        # "first-time" into "first- time", breaking endswith.
                        current_cmp = re.sub(r"\s+", "", current_norm)
                        spoken_cmp = re.sub(r"\s+", "", spoken_norm)
                        # Require ~the whole turn spoken before trusting the suffix match, else a
                        # repeated closer ("Sure." twice, final push "Sure.") matches one segment early.
                        spoken_enough = (
                            self.current_sequence_chars <= 0
                            or len(self.eos_accum_text) >= 0.9 * self.current_sequence_chars
                        )
                        logger.info(
                            f"EOS check spoken_chars={len(self.eos_accum_text)} seq_chars={self.current_sequence_chars} enough={spoken_enough}"
                        )
                        # End the stream only once the WHOLE turn text has been spoken, not when a
                        # truncated frame fragment (e.g. "s.") coincidentally suffixes it (87da790e).
                        if current_cmp and spoken_enough and spoken_cmp.endswith(current_cmp):
                            logger.info("send end_of_synthesizer_stream")
                            emit_eos = True
                    except Exception as e:
                        logger.error(f"Error matching spoken text - {e}")
                        emit_eos = True
                else:
                    logger.info("No audio data in the response")

                # A stale (previous-turn) context's frames must never end the current turn's stream.
                is_stale_context = (
                    ctx is not None and self.current_turn_context_id is not None and ctx != self.current_turn_context_id
                )
                if emit_eos and is_stale_context:
                    logger.info(f"Suppressing end-of-stream from stale context {ctx}")
                # isFinal and the text-match can both fire within one turn; emit end-of-stream
                # only once per context.
                elif emit_eos and not (ctx is not None and ctx == self._eos_context_id):
                    self._eos_context_id = ctx
                    yield b"\x00", ""

            except websockets.exceptions.ConnectionClosed:
                break
            except Exception as e:
                consecutive_errors += 1
                logger.error(f"Error occurred in receiver - {e}")
                if consecutive_errors >= max_consecutive_errors:
                    logger.error(
                        f"ElevenLabs receiver: {consecutive_errors} consecutive errors, giving up to avoid busy-spin."
                    )
                    self.connection_error = self.connection_error or str(e)
                    return
                # Back off so a persistent error can't peg the event loop.
                await asyncio.sleep(0.1)

    async def establish_connection(self):
        try:
            start_time = time.perf_counter()
            websocket = await asyncio.wait_for(
                websockets.connect(self.ws_url, ssl=get_ssl_context(self.ws_url)), timeout=10.0
            )
            if hasattr(websocket, "response") and hasattr(websocket.response, "headers"):
                self.ws_trace_id = websocket.response.headers.get("x-trace-id")
                logger.info(f"Elevenlabs WebSocket connected trace_id={self.ws_trace_id}")
            bos_message = {
                "text": " ",
                "voice_settings": {
                    "stability": self.temperature,
                    "similarity_boost": self.similarity_boost,
                    "speed": self.speed,
                    "style": self.style,
                },
                "generation_config": {
                    "chunk_length_schedule": [50, 80, 120, 150],
                },
                "xi_api_key": self.api_key,
            }
            await websocket.send(json.dumps(bos_message))
            if not self.connection_time:
                self.connection_time = round((time.perf_counter() - start_time) * 1000)
            logger.info(f"Connected to {self.ws_url}")
            return websocket
        except asyncio.TimeoutError:
            logger.error("Timeout while connecting to ElevenLabs websocket")
            return None
        except InvalidHandshake as e:
            error_msg = str(e)
            if "401" in error_msg or "403" in error_msg:
                logger.error(f"ElevenLabs authentication failed: Invalid or expired API key - {e}")
            else:
                logger.error(f"ElevenLabs handshake failed: {e}")
            self.connection_error = str(e)
            return None
        except Exception as e:
            logger.error(f"Failed to connect to ElevenLabs: {e}")
            return None


# v3 accepts any stability value but only these three change anything.
STABILITY_PRESETS = (0.0, 0.5, 1.0)
DEFAULT_STABILITY = 0.5
CLOSE_TIMEOUT_S = 1
RECONNECT_POLL_INTERVAL_S = 0.05
# The endpoint ignores inactivity_timeout and drops idle sockets at 20s.
KEEP_ALIVE_INTERVAL_S = 8


class ElevenlabsV3Synthesizer(ElevenlabsBase):
    """Eleven v3 over the text-to-dialogue socket: one flush per turn, no contexts."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ws_url = (
            f"wss://{self.elevenlabs_host}/v1/text-to-dialogue/stream-input"
            f"?model_id={self.model}&output_format={self.wire_format}&sync_alignment=true"
        )
        # v3 rejects optimize_streaming_latency with a 400.
        self.api_url = f"https://{self.elevenlabs_host}/v1/text-to-speech/{self.voice}/stream?output_format="
        # Snapped once so the inherited HTTP path matches the streamed voice. temperature is
        # optional in the config and reaches the constructor unvalidated, so it may be None.
        stability = DEFAULT_STABILITY if self.temperature is None else self.temperature
        self.temperature = min(STABILITY_PRESETS, key=lambda preset: abs(preset - stability))
        self._new_turn_pending = True
        self._interrupted = False
        # last_text_sent stays true between turns, so it cannot say whether a turn is still
        # open. This does, and keeps a dropped socket from ending an already-ended turn twice.
        self._turn_eos_emitted = True
        self._connect_lock = asyncio.Lock()
        self._keep_alive_task = None
        self._reconnect_task = None
        self._last_send_time = time.perf_counter()

    def get_sleep_time(self):
        return 0.01

    async def establish_connection(self):
        try:
            start_time = time.perf_counter()
            websocket = await asyncio.wait_for(
                websockets.connect(
                    self.ws_url,
                    ssl=get_ssl_context(self.ws_url),
                    additional_headers={"xi-api-key": self.api_key},
                ),
                timeout=10.0,
            )
            if hasattr(websocket, "response") and hasattr(websocket.response, "headers"):
                self.ws_trace_id = websocket.response.headers.get("x-trace-id")
                logger.info(f"Elevenlabs v3 WebSocket connected trace_id={self.ws_trace_id}")
            # First message only. stability is the sole setting v3 honours.
            await websocket.send(
                json.dumps(
                    {
                        "voices": [self.voice],
                        "voice_settings": {"stability": self.temperature},
                    }
                )
            )
            self._last_send_time = time.perf_counter()
            self._new_turn_pending = True
            # Teardown is over once a replacement is up.
            self._interrupted = False
            if not self.connection_time:
                self.connection_time = round((time.perf_counter() - start_time) * 1000)
            logger.info(f"Connected to {self.ws_url}")
            return websocket
        except asyncio.TimeoutError:
            logger.error("Timeout while connecting to ElevenLabs v3 websocket")
            return None
        except InvalidHandshake as e:
            error_msg = str(e)
            if "401" in error_msg or "403" in error_msg:
                logger.error(f"ElevenLabs v3 authentication failed: invalid or expired API key - {e}")
            else:
                logger.error(f"ElevenLabs v3 handshake failed: {e}")
            self.connection_error = str(e)
            return None
        except Exception as e:
            logger.error(f"Failed to connect to ElevenLabs v3: {e}")
            return None

    async def _ensure_connection(self):
        """Reconnect if down, serialised so two callers cannot both dial and leak a socket."""
        async with self._connect_lock:
            if self.conversation_ended:
                return False
            if self._is_ws_connected():
                return True
            websocket = await self.establish_connection()
            if websocket is None:
                return False
            self.websocket = websocket
            return True

    async def monitor_connection(self):
        consecutive_failures = 0
        while consecutive_failures < 3 and not self.conversation_ended:
            if not self._is_ws_connected():
                logger.info("Re-establishing ElevenLabs v3 connection...")
                if await self._ensure_connection():
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                    logger.warning(f"ElevenLabs v3 connection failed (attempt {consecutive_failures}/3)")
                    if consecutive_failures >= 3:
                        logger.error("Max connection failures reached for ElevenLabs v3")
                        self.connection_error = self.connection_error or "Max connection failures reached"
                        break
            if self._keep_alive_task is None or self._keep_alive_task.done():
                self._keep_alive_task = asyncio.create_task(self._keep_alive_loop())
            await asyncio.sleep(1)

    async def _keep_alive_loop(self):
        """Hold the socket open through long stretches of caller speech."""
        while not self.conversation_ended and not self.connection_error:
            await asyncio.sleep(1)
            if not self._is_ws_connected():
                continue
            if time.perf_counter() - self._last_send_time < KEEP_ALIVE_INTERVAL_S:
                continue
            try:
                await self.websocket.send(json.dumps({"keep_alive": True}))
                self._last_send_time = time.perf_counter()
            except Exception as e:
                logger.info(f"ElevenLabs v3 keep_alive failed: {e}")

    async def handle_interruption(self):
        """Drop the socket and dial a replacement, since nothing here cancels a turn."""
        # No network I/O: the caller awaits this on the interruption path. Detaching the
        # socket synchronously also stops the receiver reading the abandoned turn.
        try:
            self.current_turn_start_time = None
            self._new_turn_pending = True
            self._interrupted = True
            old_ws, self.websocket = self.websocket, None
            if self._reconnect_task and not self._reconnect_task.done():
                self._reconnect_task.cancel()
            # Referenced so the task is not garbage collected before it reconnects.
            self._reconnect_task = asyncio.create_task(self._recycle_socket(old_ws))
        except Exception as e:
            logger.info(f"Error handling ElevenLabs v3 interruption: {e}")

    def _classify_lost_socket(self, where):
        """Classify a lost socket: ours to abandon quietly, the provider's to report."""
        # A swallowed provider drop leaves the turn with no end-of-stream, so the final mark
        # never reaches the output handler and playback stays marked as in progress.
        self._new_turn_pending = True
        if self._interrupted:
            logger.info(f"ElevenLabs v3 socket closed by barge-in {where}, abandoning turn")
            return
        logger.error(f"ElevenLabs v3 socket dropped {where}")
        self.connection_error = f"socket dropped {where}"

    async def _recycle_socket(self, old_ws):
        """Close the abandoned socket and dial its replacement, off the barge-in path."""
        if old_ws is not None:
            try:
                await asyncio.wait_for(old_ws.close(), timeout=CLOSE_TIMEOUT_S)
            except Exception:
                pass
        await self._ensure_connection()

    async def sender(self, text, sequence_id, end_of_llm_stream=False):
        try:
            if self.conversation_ended:
                return
            if not self.should_synthesize_response(sequence_id):
                logger.info(f"Not synthesizing: sequence_id {sequence_id} not current")
                await self.flush_synthesizer_stream()
                return

            # Tighter than the 1s default: every barge-in redials, and the next turn should
            # not wait a full poll tick on a socket that reconnects in about 200ms.
            await self._wait_for_ws(poll_interval=RECONNECT_POLL_INTERVAL_S)

            if text != "":
                for text_chunk in self.text_chunker(text):
                    if not self.should_synthesize_response(sequence_id):
                        logger.info(f"Not synthesizing (inner): sequence_id {sequence_id} not current")
                        await self.flush_synthesizer_stream()
                        return
                    try:
                        if self.ws_send_time is None:
                            self.ws_send_time = time.perf_counter()
                            self._turn_eos_emitted = False
                            logger.info(f"WS send trace_id={self.ws_trace_id} first_text_sent")
                        # Only the first fragment of a turn; per-fragment new_turn would
                        # break intonation inside one utterance.
                        payload = {"text": text_chunk, "voice_id": self.voice}
                        if self._new_turn_pending:
                            payload["new_turn"] = True
                            self._new_turn_pending = False
                        ws = self.websocket
                        if ws is None:
                            self._classify_lost_socket("mid-send")
                            return
                        await ws.send(json.dumps({"inputs": [payload]}))
                        self._last_send_time = time.perf_counter()
                    except websockets.exceptions.ConnectionClosed:
                        self._classify_lost_socket("mid-send")
                        return
                    except Exception as e:
                        logger.info(f"Error sending chunk: {e}")
                        self.connection_error = str(e)
                        return

            if end_of_llm_stream:
                self.last_text_sent = True
                try:
                    # Closes the turn with is_final_audio_for_turn. close_socket would end
                    # the session rather than the turn.
                    ws = self.websocket
                    if ws is None:
                        self._classify_lost_socket("before flush")
                        return
                    await ws.send(json.dumps({"flush": True}))
                    self._last_send_time = time.perf_counter()
                    self._new_turn_pending = True
                except websockets.exceptions.ConnectionClosed:
                    self._classify_lost_socket("before flush")
                except Exception as e:
                    logger.info(f"Error sending end-of-stream signal: {e}")
                    self.connection_error = str(e)

        except asyncio.CancelledError:
            logger.info("Sender task was cancelled.")
        except Exception as e:
            logger.error(f"Unexpected error in sender: {e}")

    async def receiver(self):
        """Yields (audio_chunk, text_spoken) tuples, or (b'\\x00', '') for end-of-stream."""
        audio_chunk_count = 0
        not_connected_since = None
        consecutive_errors = 0
        max_consecutive_errors = 10
        while True:
            try:
                if self.conversation_ended:
                    return
                if not self._is_ws_connected():
                    if self.connection_error:
                        return
                    now = time.perf_counter()
                    if not_connected_since is None:
                        not_connected_since = now
                    elif now - not_connected_since > 30:
                        logger.error("ElevenLabs v3 receiver: WebSocket never connected after 30s, giving up.")
                        self.connection_error = self.connection_error or "WebSocket never connected"
                        return
                    await asyncio.sleep(0.10)
                    continue
                else:
                    not_connected_since = None

                recv_start = time.perf_counter()
                response = await self.websocket.recv()
                recv_duration = (time.perf_counter() - recv_start) * 1000
                data = json.loads(response)
                consecutive_errors = 0

                if data.get("error"):
                    logger.error(f"ElevenLabs v3 error: {data.get('error')} - {data.get('message')}")
                    self.connection_error = data.get("message") or data.get("error")
                    return

                if data.get("audio"):
                    if self.ws_send_time is not None:
                        audio_chunk_count += 1
                        if audio_chunk_count == 1:
                            time_since_send = (time.perf_counter() - self.ws_send_time) * 1000
                            logger.info(
                                f"WS recv FIRST trace_id={self.ws_trace_id} recv_wait={recv_duration:.0f}ms "
                                f"time_since_send={time_since_send:.0f}ms"
                            )
                        elif recv_duration > 200:
                            logger.info(
                                f"WS recv SLOW chunk={audio_chunk_count} trace_id={self.ws_trace_id} "
                                f"recv_wait={recv_duration:.0f}ms"
                            )
                    chunk = base64.b64decode(data["audio"])
                    try:
                        text_spoken = "".join(data.get("alignment", {}).get("chars", []))
                    except Exception:
                        text_spoken = ""
                    yield chunk, text_spoken

                # Standalone message, no audio, once per flush. The only per-turn end
                # marker; is_final arrives only when the session closes.
                if data.get("is_final_audio_for_turn"):
                    logger.info(f"WS recv is_final_audio_for_turn trace_id={self.ws_trace_id}")
                    audio_chunk_count = 0
                    self._turn_eos_emitted = True
                    yield b"\x00", ""

            except websockets.exceptions.ConnectionClosed:
                # Keep looping rather than breaking as v1 does: barge-in closes the socket
                # routinely, and returning here would end generate() for the rest of the call.
                if self.conversation_ended or self.connection_error:
                    return
                if self._interrupted:
                    logger.info("ElevenLabs v3 WebSocket closed by barge-in, waiting for reconnect")
                else:
                    # Dropped after the flush but before is_final_audio_for_turn. End the turn
                    # explicitly, or playback stays marked as in progress for the whole call.
                    logger.error("ElevenLabs v3 WebSocket dropped by provider, ending the turn")
                    if self.last_text_sent and not self._turn_eos_emitted:
                        self._turn_eos_emitted = True
                        yield b"\x00", ""
                audio_chunk_count = 0
                await asyncio.sleep(0.05)
            except Exception as e:
                consecutive_errors += 1
                logger.error(f"Error occurred in receiver - {e}")
                if consecutive_errors >= max_consecutive_errors:
                    logger.error(
                        f"ElevenLabs v3 receiver: {consecutive_errors} consecutive errors, giving up to avoid busy-spin."
                    )
                    self.connection_error = self.connection_error or str(e)
                    return
                await asyncio.sleep(0.1)

    async def cleanup(self):
        for task in (self._keep_alive_task, self._reconnect_task):
            if task:
                task.cancel()
        self._keep_alive_task = None
        self._reconnect_task = None
        await super().cleanup()
