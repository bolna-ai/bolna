"""
Kalpa Labs TTS: streaming WSS /v1/tts/{voice_id}/stream, one-shot POST /v1/tts/{voice_id}.

Kalpa's conversational speech models render named voices from GET /v1/voices

Models:
1. kalpa-tts-multilingual-beta-v0.1 speaks English and Hindi, including code-switched
Hinglish with no language parameter
2. kalpa-tts-beta-v0.1 is English-only.

Flow
1. The client authenticates with an initializeConnection frame (the handshake itself is unauthenticated),
2. The server confirms with sessionCreated
3. Each sendText frame appends text and flush=true renders the buffered utterance.
4. Audio arrives as responseAudio frames carrying base64 raw 24 kHz mono s16le PCM.
5. Every flushed utterance terminates with exactly one responseDone (status "completed", or "cancelled" after a cancelResponse).
6. For telephony, mu-law encodes audio from 24KHz to 8KHz
"""

import asyncio
import base64
import json
import os
import time

import aiohttp
import websockets
from websockets.exceptions import InvalidHandshake

from .stream_synthesizer import StreamSynthesizer
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.ssl_context import get_ssl_context
from bolna.helpers.utils import audio_to_mulaw8k, pcm_to_ulaw, resample
from bolna.memory.cache.inmemory_scalar_cache import InmemoryScalarCache

logger = configure_logger(__name__)

# The session's true rate arrives in sessionCreated; this is the documented default.
KALPA_NATIVE_SAMPLE_RATE = 24000
MULAW_SAMPLE_RATE = 8000

KALPA_DEFAULT_MODEL = "kalpa-tts-multilingual-beta-v0.1"

# Kalpa rejects utterances longer than this; we truncate rather than fail a live turn.
MAX_TEXT_CHARS = 8000

# How long a flush waits for the previous response's responseDone. Cancels settle in
# milliseconds, so hitting this means something is wrong — flush anyway and let the
# server arbitrate rather than silently dropping the turn (current gateways queue the
# flush in that case, so nothing is lost even then).
RESPONSE_IDLE_TIMEOUT = 10.0

AUDIO_QUALITIES = {"low", "medium", "high"}


class KalpaSynthesizer(StreamSynthesizer):
    def __init__(
        self,
        voice=None,
        voice_id=None,
        model=KALPA_DEFAULT_MODEL,
        temperature=None,
        top_k=None,
        acoustic_temperature=None,
        max_new_tokens=None,
        audio_quality=None,
        sampling_rate="24000",
        stream=False,
        buffer_size=400,
        caching=True,
        synthesizer_key=None,
        **kwargs,
    ):
        super().__init__(
            stream=stream,
            provider_name="kalpa",
            buffer_size=buffer_size,
            **kwargs,
        )
        self.api_key = os.environ["KALPA_API_KEY"] if synthesizer_key is None else synthesizer_key
        if not self.api_key:
            raise ValueError("Kalpa API key is required, either as synthesizer_key or KALPA_API_KEY")

        self.voice = voice
        # The API addresses voices by opaque id; a bare display name ("Kiara") is resolved
        # against GET /v1/voices on first connect.
        self.voice_id = voice_id
        if not self.voice_id and not self.voice:
            raise ValueError("Kalpa needs a voice_id or a voice name (see GET /v1/voices)")

        self.model = model
        # Only explicitly-configured knobs are sent; Kalpa's defaults are the tuned
        # production sampling.
        self.params = {
            key: value
            for key, value in (
                ("temperature", temperature),
                ("top_k", top_k),
                ("acoustic_temperature", acoustic_temperature),
                ("max_new_tokens", max_new_tokens),
                ("audio_quality", audio_quality),
            )
            if value is not None
        }

        self.use_mulaw = kwargs.get("use_mulaw", False)
        # Telephony always renders at 8 kHz mu-law; web keeps the configured PCM rate.
        self.target_sample_rate = MULAW_SAMPLE_RATE if self.use_mulaw else int(sampling_rate)
        self.sampling_rate = self.target_sample_rate
        self.native_sample_rate = KALPA_NATIVE_SAMPLE_RATE

        self.caching = caching
        if caching:
            self.cache = InmemoryScalarCache()

        self.kalpa_host = os.getenv("KALPA_API_HOST", "api.kalpalabs.ai")

        # Fail fast on a misconfigured agent rather than mid-call.
        self._validate_options()

        # Aggregation state: buffer a whole turn's chunks, flush once on end_of_llm_stream.
        # _buffer_seq tracks which turn owns the buffer so a superseded turn (new
        # sequence_id before its end_of_llm_stream) can't leak its half-buffered text.
        self._text_buffer = []
        self._buffer_seq = None
        # Set while no response is generating on the socket (one in flight per connection).
        # The counter tracks outstanding flushes: normally 0/1, briefly 2 when the idle
        # timeout lets a newer turn flush past a wedged response — responses settle in
        # flush order, so only the done that drains it back to 0 owns the slot.
        self._response_idle = asyncio.Event()
        self._response_idle.set()
        self._inflight_flushes = 0
        # After a cancel, audio frames already on the wire keep arriving until the
        # response's own responseDone; ids in this set are dropped instead of played.
        self._ignored_response_ids = set()
        self._current_response_id = None
        # A barge-in can land before responseCreated delivers the in-flight response's id;
        # this flag marks the abandonment id-independently until that response settles.
        self._abandoned_in_flight = False

    def get_sleep_time(self):
        return 0.01

    # ------------------------------------------------------------------
    # Config validation
    # ------------------------------------------------------------------

    def _validate_options(self):
        """Mirror the server's GenParams ranges so a typo fails at agent setup, not on the
        first turn of a live call. The model id is deliberately not validated against a
        fixed list — the catalog (GET /v1/models) evolves server-side."""
        temperature = self.params.get("temperature")
        if temperature is not None and not 0.0 <= float(temperature) <= 1.5:
            raise ValueError("Kalpa temperature must be between 0.0 and 1.5")
        acoustic = self.params.get("acoustic_temperature")
        if acoustic is not None and not 0.0 <= float(acoustic) <= 1.5:
            raise ValueError("Kalpa acoustic_temperature must be between 0.0 and 1.5")
        top_k = self.params.get("top_k")
        if top_k is not None and int(top_k) < 1:
            raise ValueError("Kalpa top_k must be >= 1")
        max_new_tokens = self.params.get("max_new_tokens")
        if max_new_tokens is not None and not 16 <= int(max_new_tokens) <= 2048:
            raise ValueError("Kalpa max_new_tokens must be between 16 and 2048")
        audio_quality = self.params.get("audio_quality")
        if audio_quality is not None and audio_quality not in AUDIO_QUALITIES:
            raise ValueError(f"Kalpa audio_quality must be one of {sorted(AUDIO_QUALITIES)}")

    # ------------------------------------------------------------------
    # StreamSynthesizer hooks
    # ------------------------------------------------------------------

    def _get_audio_format(self):
        return "mulaw" if self.use_mulaw else "pcm"

    def _process_audio_chunk(self, chunk):
        """Resample Kalpa's 24 kHz PCM to the target rate; mu-law encode for telephony."""
        if not chunk:
            return None
        try:
            audio = resample(
                chunk,
                self.target_sample_rate,
                format="pcm",
                original_sample_rate=self.native_sample_rate,
            )
        except Exception as e:
            logger.error(f"Error resampling Kalpa audio: {e}")
            return None
        return pcm_to_ulaw(audio) if self.use_mulaw else audio

    def _on_push(self, meta_info, text):
        """Runs synchronously in push order (before the sender task): if a new turn starts
        while the previous one is still buffered, drop the stale buffer so it can't prepend
        onto this turn's text."""
        seq = meta_info.get("sequence_id")
        if self._buffer_seq is not None and self._buffer_seq != seq and self._text_buffer:
            logger.info(
                f"Dropping {len(self._text_buffer)} unflushed Kalpa chunk(s) from superseded seq={self._buffer_seq}"
            )
            self._text_buffer = []
        self._buffer_seq = seq

    # ------------------------------------------------------------------
    # Voice resolution
    # ------------------------------------------------------------------

    async def _resolve_voice_id(self):
        """Return the opaque voice id, resolving a display name via GET /v1/voices once.

        Names match case-insensitively, on the full name ("Kiara (hindi)") or its base
        before the qualifier ("Kiara") — so agent configs don't have to carry UUIDs.
        """
        if self.voice_id:
            return self.voice_id

        headers = {"Authorization": f"Bearer {self.api_key}"}
        url = f"https://{self.kalpa_host}/v1/voices"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"Kalpa GET /v1/voices failed: {resp.status} - {await resp.text()}")
                body = await resp.json()
        voices = body.get("data", []) if isinstance(body, dict) else body

        wanted = self.voice.strip().lower()
        matches = [v for v in voices if v.get("name", "").lower() == wanted]
        if not matches:
            matches = [v for v in voices if v.get("name", "").split(" (")[0].lower() == wanted]
        if len(matches) == 1:
            self.voice_id = matches[0]["id"]
            logger.info(f"Resolved Kalpa voice {self.voice!r} -> {matches[0]['name']!r} ({self.voice_id})")
            return self.voice_id

        names = sorted(v.get("name", "") for v in voices)
        if not matches:
            raise ValueError(f"Unknown Kalpa voice {self.voice!r}. Available: {names}")
        raise ValueError(f"Ambiguous Kalpa voice {self.voice!r} matches {[v['name'] for v in matches]}")

    # ------------------------------------------------------------------
    # Interruption
    # ------------------------------------------------------------------

    async def handle_interruption(self):
        """Drop the buffered turn and cancel the in-flight response. Cancel is idempotent
        server-side, so it is unconditionally safe to fire; the cancelled response still
        terminates with its own responseDone (status "cancelled"), which frees the
        in-flight slot without emitting an end-of-stream sentinel. On current gateways a
        bare cancel also wipes undelivered server-side state (queued flush, buffered
        text) — moot here, since this integration never leaves either behind."""
        self._text_buffer = []
        self._buffer_seq = None
        if not self._response_idle.is_set():
            self._abandoned_in_flight = True
            if self._current_response_id is not None:
                self._ignored_response_ids.add(self._current_response_id)
        # The cancelled turn's end-of-stream is never forwarded, so the next turn must be
        # re-detected as new for stale text_queue entries to be pruned — even when the
        # cancel send below fails with the socket.
        self.current_turn_start_time = None
        try:
            ws = self.websocket
            if ws is not None and ws.state is websockets.protocol.State.OPEN:
                await ws.send(json.dumps({"type": "cancelResponse"}))
                logger.info("Sent cancelResponse to Kalpa TTS WebSocket")
        except Exception as e:
            logger.error(f"Error handling Kalpa interruption: {e}")

    # ------------------------------------------------------------------
    # sender / receiver
    # ------------------------------------------------------------------

    async def sender(self, text, sequence_id, end_of_llm_stream=False):
        try:
            if self.conversation_ended:
                return
            if not self.should_synthesize_response(sequence_id):
                logger.info(f"Not synthesizing: sequence_id {sequence_id} not current")
                return

            # Appends happen before any await: sender tasks start in push order and a
            # non-EOS sender never suspends, so chunks cannot aggregate out of order even
            # while the socket is reconnecting. Only the EOS task below ever waits.
            if text:
                # Guard the ownership invariant: a chunk whose turn's buffer was already
                # dropped (barge-in between task creation and start) must not append.
                if self._buffer_seq != sequence_id:
                    logger.info(f"Dropping Kalpa chunk for superseded seq={sequence_id} (buffer seq={self._buffer_seq})")
                    return
                self._text_buffer.append(text)

            # Keep buffering until the LLM turn ends: one atomic text+flush frame per turn, so
            # an abandoned turn's fragments never sit in a server-side buffer and every turn
            # settles with exactly one responseDone.
            if not end_of_llm_stream:
                return

            # The streaming LLM wrappers split chunks with rsplit(" ", 1) and drop the
            # boundary space, so it has to be restored here.
            full_text = " ".join(self._text_buffer).strip()
            self._text_buffer = []
            self._buffer_seq = None
            self.last_text_sent = True

            if not full_text:
                return
            if len(full_text) > MAX_TEXT_CHARS:
                logger.warning(f"Kalpa utterance exceeds {MAX_TEXT_CHARS} chars; truncating")
                # Cut at a word boundary so the caller doesn't hear a clipped syllable.
                cut = full_text.rfind(" ", 0, MAX_TEXT_CHARS + 1)
                full_text = full_text[:cut] if cut > 0 else full_text[:MAX_TEXT_CHARS]

            await self._wait_for_ws()

            # The wait suspends for real while the socket reconnects, and a barge-in can
            # retire this sequence in that gap — the captured turn must not flush then.
            if not self.should_synthesize_response(sequence_id):
                logger.info(f"Not synthesizing (post-wait): sequence_id {sequence_id} not current")
                return

            # One response in flight per connection: wait for the previous turn's
            # responseDone (a cancelled turn still gets one) before flushing. Event.set()
            # wakes every parked sender, so re-check and re-park until the slot is truly
            # free — otherwise two turns could ride one done into overlapping flushes.
            deadline = time.perf_counter() + RESPONSE_IDLE_TIMEOUT
            while not self._response_idle.is_set():
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    logger.warning("Kalpa: previous response still in flight after 10s; flushing anyway")
                    # Giving up on the wedged response retires its abandonment too —
                    # otherwise this turn's response would inherit it at responseCreated
                    # and lose its audio and completion sentinel. Its id (when known)
                    # stays ignored.
                    self._abandoned_in_flight = False
                    break
                try:
                    await asyncio.wait_for(self._response_idle.wait(), timeout=remaining)
                except asyncio.TimeoutError:
                    pass

            # The waits above can span a barge-in that retires this sequence without
            # cancelling the task; re-check before anything reaches the socket.
            if not self.should_synthesize_response(sequence_id):
                logger.info(f"Not synthesizing (inner): sequence_id {sequence_id} not current")
                return

            self._inflight_flushes += 1
            self._response_idle.clear()
            if self.ws_send_time is None:
                self.ws_send_time = time.perf_counter()
            try:
                await self._send_json({"type": "sendText", "text": full_text, "flush": True})
            except Exception:
                # _send_json already logged and set connection_error. Nothing was flushed,
                # so no responseDone will free the slot — release it here.
                self._inflight_flushes -= 1
                if self._inflight_flushes == 0:
                    self._response_idle.set()

        except asyncio.CancelledError:
            logger.info("Kalpa sender task was cancelled.")
        except Exception as e:
            logger.error(f"Unexpected error in Kalpa sender: {e}")

    def _settle_lost_flushes(self):
        """The socket died with responses outstanding: free the slot, reset the flush
        bookkeeping, and return how many end-of-stream sentinels settle the lost turns
        (oldest first; an abandoned turn is skipped — its pipeline already dropped it)."""
        lost = 0
        if not self._response_idle.is_set():
            self._response_idle.set()
            lost = max(1, self._inflight_flushes)
            if self._abandoned_in_flight:
                lost -= 1
        self._inflight_flushes = 0
        self._abandoned_in_flight = False
        return lost

    async def receiver(self):
        not_connected_since = None
        while True:
            try:
                if self.conversation_ended:
                    return
                if not self._is_ws_connected():
                    # A socket that dies between recvs is seen here, not by the
                    # ConnectionClosed handler below — settle its turns here too.
                    for _ in range(self._settle_lost_flushes()):
                        yield b"\x00"
                    if self.connection_error:
                        return
                    now = time.perf_counter()
                    if not_connected_since is None:
                        not_connected_since = now
                    elif now - not_connected_since > 30:
                        logger.error("Kalpa receiver: WebSocket never connected after 30s, giving up.")
                        self.connection_error = self.connection_error or "WebSocket never connected"
                        return
                    logger.info("Kalpa WebSocket is not connected, skipping receive.")
                    await asyncio.sleep(0.1)
                    continue
                else:
                    not_connected_since = None

                response = await self.websocket.recv()
                data = self._loads_event(response)
                if data is None:
                    continue

                event = data.get("type")
                if event == "responseAudio":
                    if data.get("response_id") in self._ignored_response_ids:
                        continue
                    chunk = self._decode_audio(data.get("pcm_b64"))
                    if chunk:
                        yield chunk
                elif event == "responseDone":
                    if self._inflight_flushes:
                        self._inflight_flushes -= 1
                    rid = data.get("response_id")
                    if rid == self._current_response_id:
                        self._current_response_id = None
                    status = data.get("status")
                    logger.info(f"Kalpa responseDone status={status} response_id={rid}")
                    if self._inflight_flushes:
                        # A late done for a retired response (the idle-timeout overlap):
                        # the newer flush still owns the slot — releasing it or emitting
                        # a sentinel would finish the newer turn before its audio.
                        self._ignored_response_ids.discard(rid)
                        continue
                    self._response_idle.set()
                    was_abandoned = self._abandoned_in_flight
                    self._abandoned_in_flight = False
                    if rid in self._ignored_response_ids or was_abandoned:
                        # This response's turn was abandoned at the barge-in; even a
                        # "completed" done (the cancel lost the race) must not stamp
                        # end-of-stream onto the next turn.
                        self._ignored_response_ids.discard(rid)
                    elif status == "completed":
                        yield b"\x00"
                    # "cancelled" terminates a turn we interrupted; handle_interruption()
                    # already abandoned it, so forwarding a sentinel would stamp
                    # end-of-stream on a turn the pipeline dropped.
                elif event == "error":
                    err = data.get("error") or {}
                    if data.get("fatal"):
                        # The server closes the socket after a fatal error; ConnectionClosed
                        # is what ends this loop, and auth failures won't fix themselves.
                        logger.error(f"Kalpa fatal error: {err}")
                        if err.get("type") == "authentication_error":
                            self.connection_error = err.get("message") or "authentication error"
                    else:
                        # A non-fatal error with a flush in flight means that turn will
                        # never produce audio — terminate it and free the slot. When idle
                        # (e.g. a rejected cancelResponse) a sentinel would pop the next
                        # turn's meta_info, so just log.
                        logger.error(f"Kalpa TTS error: {err}")
                        if not self._response_idle.is_set():
                            if self._inflight_flushes:
                                self._inflight_flushes -= 1
                            if self._inflight_flushes == 0:
                                self._response_idle.set()
                                if not self._abandoned_in_flight:
                                    yield b"\x00"
                                self._abandoned_in_flight = False
                elif event == "sessionCreated":
                    # Normally consumed inside establish_connection; tolerated here in case
                    # a future server pushes an unsolicited session update.
                    self.native_sample_rate = int(data.get("sample_rate") or self.native_sample_rate)
                elif event == "responseCreated":
                    self._current_response_id = data.get("response_id")
                    if self._abandoned_in_flight:
                        # The barge-in landed before this frame delivered the id; the
                        # response is already abandoned, so its audio must not play.
                        self._ignored_response_ids.add(self._current_response_id)
                    logger.info(f"Kalpa responseCreated response_id={self._current_response_id}")
                else:
                    logger.info(f"Ignoring Kalpa event: {data}")

            except websockets.exceptions.ConnectionClosed:
                logger.info("Kalpa WebSocket connection closed")
                # Responses that die with the socket never get their responseDone;
                # monitor_connection re-establishes the socket for the next turn.
                for _ in range(self._settle_lost_flushes()):
                    yield b"\x00"
                break
            except Exception as e:
                logger.error(f"Error occurred in Kalpa receiver - {e}")

    def _decode_audio(self, pcm_b64):
        if not pcm_b64:
            return None
        try:
            return base64.b64decode(pcm_b64)
        except Exception as e:
            logger.error(f"Kalpa sent undecodable audio: {e}")
            return None

    @staticmethod
    def _loads_event(data):
        try:
            parsed = json.loads(data)
        except (TypeError, ValueError):
            logger.error(f"Kalpa sent a non-JSON frame: {data!r}")
            return None
        return parsed if isinstance(parsed, dict) else None

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def _initialize_message(self):
        message = {"type": "initializeConnection", "api_key": self.api_key}
        if self.model:
            message["model"] = self.model
        if self.params:
            message["params"] = self.params
        return message

    async def establish_connection(self):
        """Connect, authenticate, and consume the init reply — so receiver() only ever sees
        response frames. Returns the ready websocket, or None (monitor_connection retries;
        deterministic rejections also set connection_error so it doesn't retry a bad key)."""
        try:
            start_time = time.perf_counter()
            voice_id = await self._resolve_voice_id()
            ws_url = f"wss://{self.kalpa_host}/v1/tts/{voice_id}/stream"
            websocket_conn = await asyncio.wait_for(
                websockets.connect(ws_url, ssl=get_ssl_context(ws_url)),
                timeout=10.0,
            )
            try:
                # The handshake is unauthenticated by design; auth rides the first frame.
                await websocket_conn.send(json.dumps(self._initialize_message()))
                reply = self._loads_event(await asyncio.wait_for(websocket_conn.recv(), timeout=10.0)) or {}
            except Exception:
                await websocket_conn.close()
                raise

            if reply.get("type") != "sessionCreated":
                err = reply.get("error") or {}
                message = err.get("message") or f"unexpected init reply: {reply}"
                logger.error(f"Kalpa session rejected: {message}")
                # A bad key, model, or voice won't fix itself on retry; an inference-side
                # outage (e.g. voice seed temporarily unavailable) might.
                if err.get("type") in ("authentication_error", "invalid_request", "not_found"):
                    self.connection_error = message
                await websocket_conn.close()
                return None

            self.native_sample_rate = int(reply.get("sample_rate") or self.native_sample_rate)
            # A fresh connection has nothing in flight.
            self._response_idle.set()
            self._inflight_flushes = 0
            self._ignored_response_ids.clear()
            self._current_response_id = None
            self._abandoned_in_flight = False
            if not self.connection_time:
                self.connection_time = round((time.perf_counter() - start_time) * 1000)
            logger.info(
                f"Connected to Kalpa TTS (model={reply.get('model')}, voice={reply.get('voice_id')}, "
                f"{self.native_sample_rate} Hz)"
            )
            return websocket_conn
        except asyncio.TimeoutError:
            logger.error("Timeout while connecting to Kalpa TTS websocket")
            return None
        except ValueError as e:
            # Voice resolution failed against the live catalog — retrying won't change it.
            logger.error(str(e))
            self.connection_error = str(e)
            return None
        except InvalidHandshake as e:
            logger.error(f"Kalpa TTS handshake failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to connect to Kalpa TTS: {e}")
            return None

    # ------------------------------------------------------------------
    # HTTP (one-shot renders: non-streaming loop, prewarm/handoff clips, caching)
    # ------------------------------------------------------------------

    async def _generate_http(self, text):
        """POST /v1/tts/{voice_id}; the response JSON carries base64 16-bit PCM WAV."""
        try:
            voice_id = await self._resolve_voice_id()
        except Exception as e:
            logger.error(f"Kalpa voice resolution failed: {e}")
            return None

        payload = {"text": text}
        if self.model:
            payload["model"] = self.model
        if self.params:
            payload["params"] = self.params
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        url = f"https://{self.kalpa_host}/v1/tts/{voice_id}"
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                request_id = response.headers.get("X-Request-ID")
                if response.status != 200:
                    logger.error(
                        f"Kalpa TTS HTTP error: {response.status} request_id={request_id} - {await response.text()}"
                    )
                    return None
                data = await response.json()
        audio = (data.get("audio") or {}).get("data_b64")
        if not audio:
            logger.error(f"Kalpa TTS HTTP response carried no audio (request_id={request_id})")
            return None
        return self._decode_audio(audio)

    async def synthesize(self, text):
        """One-shot render used by prewarm/handoff paths. The API returns WAV, whose
        self-describing header lets downstream mu-law conversion pick the right rate."""
        return await self._generate_http(text)

    async def synthesize_telephony_clip(self, text):
        """Mu-law 8000 one-shot for handoff/prewarm clips, converted in-process so the
        caller skips the pydub/ffmpeg decode. None on non-telephony configs."""
        if not self.use_mulaw:
            return None
        audio = await self._generate_http(text)
        if not audio:
            return None
        return self._process_http_audio(audio)

    def _process_http_audio(self, audio):
        """Convert the one-shot WAV to the format the non-streaming loop should emit."""
        if not audio:
            # The non-streaming loop has no None guard: a None packet crashes the output
            # handler's b64encode and mutes the session. b"\x00" just ends the turn.
            return b"\x00"
        if self.use_mulaw:
            return audio_to_mulaw8k(audio, rate_hint=self.native_sample_rate, format_hint="wav")
        if self.target_sample_rate != self.native_sample_rate:
            return resample(audio, self.target_sample_rate, format="wav")
        return audio

    def _get_http_audio_format(self):
        return "mulaw" if self.use_mulaw else "wav"
