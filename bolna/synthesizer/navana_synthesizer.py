"""
Navana TTS: streaming WSS /v1, one-shot POST /tts/bytes.

A socket opens with one `hello` (language, voice, output format) that the server answers with
`ready`, then carries text frames that are each synthesized whole, in the order sent. Every audio
chunk follows a JSON header naming its frame's seq and whether it is that frame's last chunk.

There is no flush and no cancel. A frame is spoken as its own utterance, so text is held to sentence
ends, and a barge-in that still has audio owed replaces the socket rather than wait it out.
"""

import asyncio
import json
import os
import re
import time
import uuid

import aiohttp
import websockets
from websockets.exceptions import ConnectionClosed, InvalidHandshake

from .elevenlabs_synthesizer import split_flushable
from .stream_synthesizer import StreamSynthesizer
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.ssl_context import get_ssl_context
from bolna.helpers.utils import pcm_to_ulaw, pcm_to_wav_bytes
from bolna.memory.cache.inmemory_scalar_cache import InmemoryScalarCache

logger = configure_logger(__name__)

NAVANA_TTS_HOST = "tts.navana.ai"

NAVANA_TTS_DEFAULT_MODEL = "navana-tts"
# The distilled model: faster, slightly lower quality.
NAVANA_TTS_FAST_MODEL = "navana-tts-fast"
NAVANA_TTS_MODELS = (NAVANA_TTS_DEFAULT_MODEL, NAVANA_TTS_FAST_MODEL)

# Every voice speaks every language, so a voice id never needs to match the language.
NAVANA_TTS_DEFAULT_VOICE = "default_female"
NAVANA_TTS_LANGUAGES = frozenset({"bn", "en", "gu", "hi", "kn", "ml", "mr", "or", "ta", "te"})
# bolna also carries Odia as "od".
_LANGUAGE_ALIASES = {"od": "or"}

SAMPLE_RATES = (8000, 16000, 24000)

CONNECT_TIMEOUT_S = 10.0
HTTP_TIMEOUT_S = 30.0
# How often an idle receiver checks whether the call has failed, ended, or stalled.
RECEIVER_POLL_S = 0.25
# A socket that owes audio but has sent no chunk for this long is treated as dropped, or its turn would
# never end. A chunk normally follows within a few hundred ms.
STALL_S = 5.0

# Error codes no reconnect can fix, so the call ends with the reason instead of retrying. The rest
# (idle_timeout, concurrency_limit, server_overloaded, triton_5xx, ...) close the socket and
# monitor_connection opens a new one.
FATAL_ERRORS = frozenset(
    {"unauthorized", "insufficient_funds", "unknown_voice", "bad_request", "protocol_violation", "bad_json"}
)


class NavanaSynthesizer(StreamSynthesizer):
    def __init__(
        self,
        voice=NAVANA_TTS_DEFAULT_VOICE,
        voice_id=None,
        model=NAVANA_TTS_DEFAULT_MODEL,
        language="hi",
        sampling_rate="24000",
        stream=False,
        buffer_size=400,
        caching=True,
        synthesizer_key=None,
        **kwargs,
    ):
        super().__init__(
            stream=stream,
            provider_name="navana",
            buffer_size=buffer_size,
            **kwargs,
        )
        self.api_key = synthesizer_key or os.getenv("NAVANA_API_KEY")
        if not self.api_key:
            raise ValueError("Navana API key is required, either as synthesizer_key or NAVANA_API_KEY")

        # voice_id is what Navana addresses (a stock id or a cloned cv_ id); voice is its display name.
        self.voice = voice_id or voice or NAVANA_TTS_DEFAULT_VOICE
        self.model = model or NAVANA_TTS_DEFAULT_MODEL
        self.language = self._normalise_language(language)

        self.caching = caching
        if caching:
            self.cache = InmemoryScalarCache()

        self.use_mulaw = kwargs.get("use_mulaw", False)
        self.sampling_rate = 8000 if self.use_mulaw else int(sampling_rate or 24000)

        host = os.getenv("NAVANA_TTS_HOST", NAVANA_TTS_HOST)
        self.ws_url = f"wss://{host}/v1"
        self.api_url = f"https://{host}/tts/bytes"
        # Prefixed so Navana can tell bolna sessions apart in its own logs.
        self.client_session_id = f"bolna-{uuid.uuid4().hex[:12]}"

        self._validate_options()

        # One lock for every frame on the wire, so concurrent sender tasks keep push order.
        self.send_lock = asyncio.Lock()
        self.connect_lock = asyncio.Lock()
        self.events = asyncio.Queue()
        self.readers = set()
        # Closes left to finish on their own; held here because the event loop keeps only weak references.
        self.background_tasks = set()
        self.reconnect_task = None
        # Last send or chunk while frames are owed, for the stall check.
        self.last_progress = time.monotonic()

        # Text held back until it ends a sentence, for the sequence it belongs to.
        self.pending_text = ""
        self.pending_sequence_id = None
        # Frame seq -> bolna sequence_id, for frames sent on the live socket whose last chunk is owed.
        self.next_frame_seq = 0
        self.frame_owner = {}
        self.last_sent_frame = None
        # Frame seq -> sequences whose turn ends once that frame's last chunk arrives.
        self.final_frames = {}

    @staticmethod
    def _normalise_language(language):
        """Reduce bolna's region-qualified code ("hi-IN") to the bare code Navana wants."""
        if not isinstance(language, str) or not language.strip():
            return "hi"
        primary = re.split(r"[-_]", language.strip().lower(), maxsplit=1)[0]
        return _LANGUAGE_ALIASES.get(primary, primary)

    def _validate_options(self):
        if self.language not in NAVANA_TTS_LANGUAGES:
            raise ValueError(f"Navana language must be one of {sorted(NAVANA_TTS_LANGUAGES)}, got {self.language!r}")
        if self.model not in NAVANA_TTS_MODELS:
            raise ValueError(f"Navana model must be one of {NAVANA_TTS_MODELS}, got {self.model!r}")
        # use_mulaw already pinned the rate to 8000, so only the linear path can carry a bad rate.
        if self.sampling_rate not in SAMPLE_RATES:
            raise ValueError(f"Navana sample_rate must be one of {SAMPLE_RATES}, got {self.sampling_rate}")

    def get_sleep_time(self):
        return 0.01

    # ------------------------------------------------------------------
    # Audio format
    # ------------------------------------------------------------------

    def _get_audio_format(self):
        return "mulaw" if self.use_mulaw else "pcm"

    def _output_format(self):
        """Always linear PCM at the call's rate. Navana's own mulaw output is not G.711, so
        telephony audio is encoded here instead."""
        return f"{self.sampling_rate}:pcm16"

    def _process_audio_chunk(self, chunk):
        """Each chunk is whole int16 samples at the call's rate, so it converts on its own with no
        resampling; only telephony needs mu-law."""
        if not chunk:
            return None
        if len(chunk) % 2:
            logger.warning(f"Navana TTS sent an odd-length PCM chunk ({len(chunk)} bytes); dropping its last byte")
            chunk = chunk[:-1]
        return pcm_to_ulaw(chunk) if self.use_mulaw else chunk

    def _unpack_receiver_message(self, item):
        return item if isinstance(item, tuple) else (item, {})

    def _headers(self):
        return {"Authorization": f"Bearer {self.api_key}", "User-Agent": "bolna"}

    def _speech_options(self):
        """Knobs shared by the hello and the one-shot body. The hello rejects any field it doesn't
        know, `speed` included, so nothing else goes in here until it does."""
        options = {"lang": self.language, "voice": self.voice}
        if self.model == NAVANA_TTS_FAST_MODEL:
            options["use_fast"] = True
        return options

    # ------------------------------------------------------------------
    # Turn bookkeeping
    # ------------------------------------------------------------------

    def _take_flushable(self, text, sequence_id, end_of_llm_stream):
        """Units of this turn's text to send now; the unfinished sentence stays held."""
        carried = []
        if sequence_id != self.pending_sequence_id:
            if self.pending_text and self.should_synthesize_response(self.pending_sequence_id):
                carried = [self.pending_text]
            self.pending_text, self.pending_sequence_id = "", sequence_id
        # The LLM wrappers cut pushes at a space and drop it.
        joined = " ".join(f"{self.pending_text} {text or ''}".split())
        units, self.pending_text = split_flushable(joined, end_of_llm_stream)
        return carried + units

    def _end_of_stream(self, sequence_id):
        """The turn's sentinel, or the retired form for a turn the pipeline has already dropped."""
        if self.should_synthesize_response(sequence_id):
            return b"\x00"
        return b"\x00", {"retired_sequence_id": sequence_id}

    def _close_turn(self, sequence_id):
        """Frames are served in order, so a turn ends with the last frame on the wire, whoever sent it."""
        last = self.last_sent_frame
        if last is not None and last in self.frame_owner:
            self.final_frames.setdefault(last, []).append(sequence_id)
        else:
            self.events.put_nowait(b"\x00")

    def _end_lost_turn(self, ws, frame_seq, sequence_id):
        """The turn's last text died with `ws`, so no frame will ever arrive to end it."""
        if ws is self.websocket:
            # Ended when the socket is dropped: by its reader once drained, or by the reconnect.
            self.final_frames.setdefault(frame_seq, []).append(sequence_id)
        else:
            # Already detached, and its leftover frames are ignored, so nothing is left to wait for.
            self.events.put_nowait(self._end_of_stream(sequence_id))

    def _drain_events(self):
        """Drop whatever is queued for the generator. In place: receiver() may be waiting on this queue."""
        while not self.events.empty():
            self.events.get_nowait()

    def _spawn(self, coro):
        task = asyncio.create_task(coro)
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)
        return task

    def _redial(self):
        """Open the next socket now rather than on monitor_connection's next 1 s check."""
        if self.conversation_ended or self.connection_error:
            return
        if self.reconnect_task is None or self.reconnect_task.done():
            self.reconnect_task = asyncio.create_task(self.establish_connection())

    def _settle_owed_turns(self):
        """Drop the socket's state. Owed audio is lost with it, but every turn already closed still
        gets its end, or playback would stay marked in progress for the rest of the call."""
        for sequence_ids in self.final_frames.values():
            for sequence_id in sequence_ids:
                self.events.put_nowait(self._end_of_stream(sequence_id))
        self._reset_socket_state()

    def _reset_socket_state(self):
        self.frame_owner = {}
        self.final_frames = {}
        self.last_sent_frame = None

    # ------------------------------------------------------------------
    # sender
    # ------------------------------------------------------------------

    async def sender(self, text, sequence_id, end_of_llm_stream=False):
        try:
            if self.conversation_ended:
                return
            if not self.should_synthesize_response(sequence_id):
                logger.info(f"Not synthesizing: sequence_id {sequence_id} not current")
                return

            # At the first push, so first-audio latency counts the hold.
            if self.ws_send_time is None:
                self.ws_send_time = time.perf_counter()
            # Before any await, so pushes reach the buffer in order.
            units = self._take_flushable(text, sequence_id, end_of_llm_stream)
            if end_of_llm_stream:
                self.last_text_sent = True

            async with self.send_lock:
                await self._wait_for_ws()
                if not self._is_ws_connected():
                    return
                for unit in units:
                    if not self.should_synthesize_response(sequence_id):
                        logger.info(f"Not synthesizing (inner): sequence_id {sequence_id} not current")
                        return
                    if not await self._send_text(unit, sequence_id, ends_turn=end_of_llm_stream):
                        return
                if end_of_llm_stream:
                    self._close_turn(sequence_id)

        except asyncio.CancelledError:
            logger.info("Navana sender task was cancelled.")
        except Exception as e:
            logger.error(f"Unexpected error in Navana sender: {e}")

    async def _send_text(self, unit, sequence_id, ends_turn=False):
        ws = self.websocket
        frame_seq = self.next_frame_seq
        self.next_frame_seq += 1
        self.frame_owner[frame_seq] = sequence_id
        self.last_sent_frame = frame_seq
        # Claim before sending: text lost into a socket that dies mid-send is still this turn's.
        self.current_turn_socket = ws
        try:
            await ws.send(json.dumps({"type": "text", "seq": frame_seq, "target_text": unit}))
            self.last_progress = time.monotonic()
            return True
        except ConnectionClosed:
            # Not fatal: monitor_connection replaces the socket. Frame seqs are never reused, so
            # this pop can only remove the frame that never reached Navana.
            logger.warning("Navana TTS socket closed mid-send")
            self.frame_owner.pop(frame_seq, None)
            if ends_turn:
                self._end_lost_turn(ws, frame_seq, sequence_id)
            return False

    # ------------------------------------------------------------------
    # receiver
    # ------------------------------------------------------------------

    async def receiver(self):
        """Yields what the socket readers queue: audio, b'\\x00' at a turn's end, or a retired turn's end."""
        while not self.conversation_ended:
            try:
                item = await asyncio.wait_for(self.events.get(), timeout=RECEIVER_POLL_S)
            except asyncio.TimeoutError:
                if self.connection_error:
                    return
                self._check_stall()
                continue
            yield item

    def _check_stall(self):
        """Drop a socket that owes audio but has sent nothing for STALL_S, as if it had closed."""
        ws = self.websocket
        if ws is None or not self.frame_owner or time.monotonic() - self.last_progress < STALL_S:
            return
        logger.error(f"Navana TTS sent no audio for {STALL_S}s with {len(self.frame_owner)} frames owed")
        self._spawn(self._close_quietly(ws))
        self._on_socket_closed(ws)

    async def _read(self, ws):
        """Pump one socket's frames into the event queue until it closes."""
        header = {}
        try:
            async for message in ws:
                if isinstance(message, (bytes, bytearray)):
                    self._on_audio(ws, header, bytes(message))
                    header = {}
                    continue
                try:
                    frame = json.loads(message)
                except ValueError:
                    frame = None
                if not isinstance(frame, dict):
                    logger.error(f"Navana TTS sent a text frame that is not a JSON object: {message!r}")
                    continue
                if frame.get("type") == "audio":
                    header = frame
                elif frame.get("type") == "error":
                    self._on_error(ws, frame)
        except ConnectionClosed:
            pass
        except Exception as e:
            # The socket is still open, so close it here: otherwise it looks connected, monitor_connection
            # never replaces it, and senders keep writing to a socket nobody reads.
            logger.error(f"Navana TTS reader stopped: {e!r}")
            self._spawn(self._close_quietly(ws))
        finally:
            self._on_socket_closed(ws)

    def _on_audio(self, ws, header, audio):
        # A replaced socket drains frames for a turn the next socket no longer owes.
        if ws is not self.websocket:
            return
        self.last_progress = time.monotonic()
        seq = header.get("seq")
        owner = self.frame_owner.get(seq)
        if owner is not None and audio and self.should_synthesize_response(owner):
            self.events.put_nowait(audio)
        if header.get("is_last_chunk"):
            self.frame_owner.pop(seq, None)
            for sequence_id in self.final_frames.pop(seq, []):
                self.events.put_nowait(self._end_of_stream(sequence_id))

    def _on_error(self, ws, frame):
        code, message = frame.get("code"), frame.get("message")
        # lang_corrected is the one non-fatal frame: the audio follows in another language.
        if frame.get("fatal") is False:
            logger.warning(f"Navana TTS warning {code}: {message}")
            return
        if ws is not self.websocket:
            return
        logger.error(f"Navana TTS error {code}: {message}")
        if code in FATAL_ERRORS:
            self.connection_error = f"Navana TTS {code}: {message}"

    def _on_socket_closed(self, ws):
        """The live socket went away under us: idle timeout, server error, or a crashed or stalled reader.
        The protocol has no keepalive frame, so the replacement dials at once instead of waiting for
        monitor_connection's next check."""
        if ws is not self.websocket:
            return
        self.websocket = None
        self._settle_owed_turns()
        self._redial()

    # ------------------------------------------------------------------
    # Interruption
    # ------------------------------------------------------------------

    async def handle_interruption(self):
        """Drop held text and queued audio; with audio still owed, swap the socket, since only a close
        stops Navana."""
        self.pending_text, self.pending_sequence_id = "", None
        # Queued audio, and any queued end, belong to the reply just cut off. Left in place they would be
        # paired with the next turn's meta, so the caller hears the end of the interrupted reply.
        self._drain_events()
        if not self.frame_owner:
            # Everything sent has finished rendering, so the socket owes nothing and stays.
            self.final_frames = {}
            return
        ws, self.websocket = self.websocket, None
        self._reset_socket_state()
        logger.info("Navana TTS: replacing the socket to drop an interrupted turn's audio")
        if ws is not None:
            # Not awaited: the close handshake must not delay the replacement.
            self._spawn(self._close_quietly(ws))
        self._redial()

    async def close_connection(self):
        """Close a multilingual leg's socket. Also stop a redial a barge-in started, or it would reopen
        the socket closed here, and drop what is queued for turns the leg will not finish."""
        if self.reconnect_task is not None:
            self.reconnect_task.cancel()
            self.reconnect_task = None
        self.pending_text, self.pending_sequence_id = "", None
        self._reset_socket_state()
        self._drain_events()
        await super().close_connection()

    @staticmethod
    async def _close_quietly(ws):
        try:
            await ws.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def _hello(self):
        hello = {"type": "hello", "output_format": self._output_format(), "client_session_id": self.client_session_id}
        hello.update(self._speech_options())
        return hello

    async def establish_connection(self):
        # monitor_connection and a barge-in replacement can both get here; only one may connect.
        async with self.connect_lock:
            if self.conversation_ended:
                return None
            if self._is_ws_connected():
                return self.websocket
            return await self._connect()

    async def _connect(self):
        websocket = None
        try:
            start_time = time.perf_counter()
            websocket = await asyncio.wait_for(
                websockets.connect(
                    self.ws_url,
                    additional_headers=self._headers(),
                    ssl=get_ssl_context(self.ws_url),
                    compression=None,
                ),
                timeout=CONNECT_TIMEOUT_S,
            )
            await websocket.send(json.dumps(self._hello()))
            ready = json.loads(await asyncio.wait_for(websocket.recv(), timeout=CONNECT_TIMEOUT_S))
            if ready.get("type") != "ready":
                code, message = ready.get("code"), ready.get("message")
                logger.error(f"Navana TTS refused the session: {code}: {message}")
                if code in FATAL_ERRORS:
                    self.connection_error = f"Navana TTS {code}: {message}"
                await self._close_quietly(websocket)
                return None
            if self.conversation_ended:
                # The call ended during the handshake; installing this socket would keep a session open.
                await self._close_quietly(websocket)
                return None
            if not self.connection_time:
                self.connection_time = round((time.perf_counter() - start_time) * 1000)
            logger.info(f"Connected to {self.ws_url} session_id={ready.get('session_id')}")

            # Anything still waiting belongs to the dead socket this one replaces.
            self._settle_owed_turns()
            self.websocket = websocket
            self.last_progress = time.monotonic()
            reader = asyncio.create_task(self._read(websocket))
            self.readers.add(reader)
            reader.add_done_callback(self.readers.discard)
            # Wakes senders waiting in _wait_for_ws now, not at its next poll.
            self.websocket_ready.set()
            return websocket
        except asyncio.CancelledError:
            # Not an Exception, so the close below would be skipped and the half-open session leak.
            if websocket is not None:
                self._spawn(self._close_quietly(websocket))
            raise
        except asyncio.TimeoutError:
            logger.error("Timeout while connecting to Navana TTS websocket")
        except InvalidHandshake as e:
            error_msg = str(e)
            logger.error(f"Navana TTS handshake failed: {e}")
            if "401" in error_msg or "403" in error_msg:
                self.connection_error = error_msg
        except ConnectionClosed as e:
            logger.error(f"Navana TTS closed the socket before ready: {e}")
        except Exception as e:
            logger.error(f"Failed to connect to Navana TTS: {e}")
        if websocket is not None:
            await self._close_quietly(websocket)
        return None

    async def cleanup(self):
        # Before the base closes the socket: a redial finishing during that close would otherwise install
        # a live socket, and its session, after the call.
        self.conversation_ended = True
        if self.reconnect_task is not None:
            self.reconnect_task.cancel()
        await super().cleanup()
        for task in self.readers:
            if not task.done():
                task.cancel()

    # ------------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------------

    async def _render(self, text, output_format):
        """One-shot render; raw audio in `output_format`, or None on failure."""
        payload = {"text": text, "output_format": output_format, **self._speech_options()}
        headers = {**self._headers(), "Content-Type": "application/json"}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=HTTP_TIMEOUT_S),
                ) as response:
                    if response.status == 200:
                        return await response.read()
                    logger.error(
                        f"Navana TTS HTTP error: {response.status} request_id={response.headers.get('X-Request-Id')}"
                        f" - {(await response.text())[:300]}"
                    )
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.error(f"Navana TTS HTTP request failed: {e!r}")
        return None

    async def _generate_http(self, text):
        return await self._render(text, self._output_format())

    async def synthesize_telephony_clip(self, text):
        """Mu-law 8000 encoded in-process from 8 kHz PCM, so handoff/prewarm clips skip the
        pydub/ffmpeg decode."""
        audio = await self._render(text, "8000:pcm16")
        if not audio or len(audio) % 2:
            return None
        return pcm_to_ulaw(audio)

    async def synthesize_pcm_clip(self, text, sample_rate):
        if int(sample_rate) not in SAMPLE_RATES:
            return None
        return await self._render(text, f"{int(sample_rate)}:pcm16")

    async def synthesize(self, text):
        """WAV-wrapped: callers decode this with a rate guessed off the synthesizer, which headerless
        PCM would get wrong."""
        audio = await self._render(text, f"{self.sampling_rate}:pcm16")
        if not audio:
            return None
        return pcm_to_wav_bytes(audio, sample_rate=self.sampling_rate)
