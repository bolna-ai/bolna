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


class ElevenlabsSynthesizer(StreamSynthesizer):
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
        self.ws_trace_id = None
        self.current_turn_ttfb = None
        self.eos_accum_context_id = None  # context whose spoken chars are being accumulated
        self.eos_accum_text = ""  # spoken-so-far for that context (end-of-stream match)

    # ------------------------------------------------------------------
    # StreamSynthesizer hooks
    # ------------------------------------------------------------------

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

    def _on_push(self, meta_info, text):
        # Mint only for pushes that will actually synthesize — a superseded push must not
        # advance current_turn_context_id, or the prior turn's real isFinal gets suppressed.
        if not self.context_id and self.should_synthesize_response(meta_info.get("sequence_id")):
            self.context_id = str(uuid.uuid4())
            self.current_turn_context_id = self.context_id

    # ------------------------------------------------------------------
    # Format helper
    # ------------------------------------------------------------------

    def _get_output_format(self):
        return self.wire_format

    # ------------------------------------------------------------------
    # Interruption
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # sender / receiver
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # HTTP fallback
    # ------------------------------------------------------------------

    async def synthesize(self, text):
        return await self._generate_http(text, format="mp3_44100_128")

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


# Out-of-preset floats are accepted silently, so snap before sending.
STABILITY_PRESETS = (0.0, 0.5, 1.0)

# The 20s idle cap is not adjustable here: inactivity_timeout is ignored on this endpoint.
KEEP_ALIVE_INTERVAL = 8


class ElevenlabsV3Synthesizer(ElevenlabsSynthesizer):
    """Eleven v3, which 403s on multi-stream-input and is served only from text-to-dialogue.

    Voices register in a first message rather than the URL, text goes as ``inputs`` arrays,
    turns end on ``is_final_audio_for_turn``, and there are no contexts. Everything outside
    the wire protocol is inherited.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ws_url = (
            f"wss://{self.elevenlabs_host}/v1/text-to-dialogue/stream-input"
            f"?model_id={self.model}&output_format={self.wire_format}&sync_alignment=true"
        )
        # Nulled so no inherited code path reads a context this endpoint does not have.
        self.context_id = None
        self.current_turn_context_id = None
        self._new_turn_pending = True
        self._connect_lock = asyncio.Lock()
        self._keep_alive_task = None
        self._reconnect_task = None
        self._last_send_time = time.perf_counter()

    def get_sleep_time(self):
        return 0.01

    def _on_push(self, meta_info, text):
        """No context to mint; turn boundaries are marked with new_turn on the wire."""
        return

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

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
            # First message only. stability is the sole setting v3 honours; similarity_boost,
            # speed and style measurably do nothing.
            await websocket.send(
                json.dumps(
                    {
                        "voices": [self.voice],
                        "voice_settings": {"stability": self._snapped_stability()},
                    }
                )
            )
            self._last_send_time = time.perf_counter()
            self._new_turn_pending = True
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

    def _snapped_stability(self):
        return min(STABILITY_PRESETS, key=lambda preset: abs(preset - self.temperature))

    async def _ensure_connection(self):
        """Reconnect if down, serialised so the barge-in redial and the monitor loop
        cannot both dial and leak a socket."""
        async with self._connect_lock:
            if self._is_ws_connected():
                return True
            websocket = await self.establish_connection()
            if websocket is None:
                return False
            self.websocket = websocket
            return True

    async def monitor_connection(self):
        consecutive_failures = 0
        while consecutive_failures < 3:
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
            if time.perf_counter() - self._last_send_time < KEEP_ALIVE_INTERVAL:
                continue
            try:
                await self.websocket.send(json.dumps({"keep_alive": True}))
                self._last_send_time = time.perf_counter()
            except Exception as e:
                logger.info(f"ElevenLabs v3 keep_alive failed: {e}")

    # ------------------------------------------------------------------
    # Interruption
    # ------------------------------------------------------------------

    async def handle_interruption(self):
        """Drop the socket, since no control message on this endpoint cancels a turn.

        The redial is backgrounded because awaiting it would stall barge-in cleanup for
        up to the 10s connect timeout.
        """
        try:
            self.current_turn_start_time = None
            self._new_turn_pending = True
            if self._is_ws_connected():
                await self.websocket.close()
            # Referenced so the task cannot be garbage collected before it reconnects.
            self._reconnect_task = asyncio.create_task(self._ensure_connection())
        except Exception as e:
            logger.info(f"Error handling ElevenLabs v3 interruption: {e}")

    # ------------------------------------------------------------------
    # sender / receiver
    # ------------------------------------------------------------------

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
                        # Only the turn's first fragment closes the previous prosody segment;
                        # per-fragment new_turn would break intonation inside one utterance.
                        payload = {"text": text_chunk, "voice_id": self.voice}
                        if self._new_turn_pending:
                            payload["new_turn"] = True
                            self._new_turn_pending = False
                        await self.websocket.send(json.dumps({"inputs": [payload]}))
                        self._last_send_time = time.perf_counter()
                    except websockets.exceptions.ConnectionClosed:
                        # Expected: barge-in closed the socket. Recording a connection error
                        # here would surface as SynthesizerError and hang up the call on every
                        # interruption. The reconnect is already in flight.
                        logger.info("ElevenLabs v3 socket closed mid-send, abandoning turn")
                        self._new_turn_pending = True
                        return
                    except Exception as e:
                        logger.info(f"Error sending chunk: {e}")
                        self.connection_error = str(e)
                        return

            if end_of_llm_stream:
                self.last_text_sent = True
                try:
                    # Generates everything buffered and closes the turn with
                    # is_final_audio_for_turn. close_socket would end the session, not the turn.
                    await self.websocket.send(json.dumps({"flush": True}))
                    self._last_send_time = time.perf_counter()
                    self._new_turn_pending = True
                except websockets.exceptions.ConnectionClosed:
                    logger.info("ElevenLabs v3 socket closed before flush, abandoning turn")
                    self._new_turn_pending = True
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

                # Standalone message, no audio, once per flush. The only per-turn end marker:
                # is_final arrives only when the session closes.
                if data.get("is_final_audio_for_turn"):
                    logger.info(f"WS recv is_final_audio_for_turn trace_id={self.ws_trace_id}")
                    audio_chunk_count = 0
                    yield b"\x00", ""

            except websockets.exceptions.ConnectionClosed:
                # Keep looping rather than breaking as v1 does: barge-in closes the socket
                # routinely, and returning here would end generate() for the rest of the call.
                if self.conversation_ended or self.connection_error:
                    return
                logger.info("ElevenLabs v3 WebSocket closed, waiting for reconnect")
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
