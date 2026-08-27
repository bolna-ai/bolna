"""
Kalpa Labs TTS: streaming WSS /v1/tts/{voice_id}/stream, one-shot POST /v1/tts/{voice_id}.

Kalpa's conversational speech models render named voices from GET /v1/voices

Models:
1. kalpa-tts-multilingual-beta-v0.1 speaks English and Hindi, including code-switched
Hinglish with no language parameter
2. kalpa-tts-beta-v0.1 is English-only.

Flow
1. The client authenticates with an initializeConnection frame (the handshake itself is
   unauthenticated). Its generation_config.chunk_length_schedule opts the connection into
   server-side segmentation: once buffered text crosses the schedule's next threshold and ends
   at a complete sentence, that part starts rendering while the rest is still arriving.
2. The server confirms with sessionCreated.
3. LLM chunks are streamed as sendText frames the moment they arrive; flush=true (sent at
   end_of_llm_stream) ends the utterance and renders whatever remains.
4. Audio arrives as responseAudio frames carrying base64 raw 24 kHz mono s16le PCM. The whole
   utterance stays one response on the wire — one responseCreated, one responseDone — but its
   first audio can arrive long before the flush.
5. Every utterance terminates with exactly one responseDone (status "completed", or
   "cancelled" after a cancelResponse).
6. For telephony, mu-law encodes audio from 24KHz to 8KHz
"""

import asyncio
import base64
import json
import os
import time
from collections import deque

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

# Buffered-character thresholds before each successive part of an utterance may start
# rendering (the last value repeats). The same schedule ElevenLabs runs in this codebase, and
# also Kalpa's server-side default — sent explicitly, because omitting generation_config
# altogether falls back to generate-only-on-flush and loses the early first audio.
DEFAULT_CHUNK_LENGTH_SCHEDULE = [50, 80, 120, 150]

# How long a new utterance waits for the previous response's responseDone. Cancels settle in
# milliseconds, so hitting this means the connection's state is wedged or unknowable; the
# socket is closed to reset it (the receiver settles the lost turn, monitor_connection
# redials) rather than flushing into a session we no longer understand.
RESPONSE_IDLE_TIMEOUT = 10.0

AUDIO_QUALITIES = {"low", "medium", "high"}

# Voice display names resolve to catalog ids once per process, not once per call: the task
# manager builds a fresh synthesizer for every call, and the GET /v1/voices round trip sits
# on the connect path. The catalog is global (never key-scoped), so (host, name) is the
# whole key; failures are never cached, so a typo keeps failing loudly and a voice added to
# the catalog later is picked up without a restart.
_VOICE_IDS = {}  # (host, lowercased display name) -> voice id


class KalpaSynthesizer(StreamSynthesizer):
    def __init__(
        self,
        voice=None,
        voice_id=None,
        model=KALPA_DEFAULT_MODEL,
        temperature=None,
        acoustic_temperature=None,
        max_new_tokens=None,
        audio_quality=None,
        chunk_length_schedule=None,
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
                ("acoustic_temperature", acoustic_temperature),
                ("max_new_tokens", max_new_tokens),
                ("audio_quality", audio_quality),
            )
            if value is not None
        }
        self.chunk_length_schedule = list(chunk_length_schedule or DEFAULT_CHUNK_LENGTH_SCHEDULE)

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

        # Text is streamed to the server as the LLM produces it (segmentation renders it
        # early), so the client tracks the open utterance rather than a local buffer. One
        # utterance occupies the connection from its first sendText to its responseDone.
        # The lock releases senders in acquisition order, so frames leave in push order even
        # when an earlier sender suspends mid-await (slot wait, reconnect).
        self._send_lock = asyncio.Lock()
        self._turn_seq = None  # sequence that owns the open (un-flushed) utterance
        self._turn_chunks = []  # wire fragments already sent for it (reconnect replay)
        self._turn_chars = 0
        self._turn_truncated = False
        self._turn_dead = False  # utterance abandoned mid-stream; drop its stragglers
        self._turn_glue = " "  # what the wrappers' split consumed before the next chunk
        # A connection generation stamps which socket the turn's text went to; a mismatch at
        # send time means the server lost its buffer and the whole turn must be resent.
        self._conn_gen = 0
        self._turn_conn_gen = 0
        # A superseded turn can leave un-flushed text in the server buffer (no interruption
        # ran); the generation it happened on gates a defensive wipe before the next turn.
        self._dirty_conn_gen = None
        # Bumped by handle_interruption() before anything else; senders retire at their
        # next await boundary on a mismatch — including on task-manager paths that cancel
        # first and only invalidate the sequence later. Each sender's epoch is captured at
        # PUSH time (_on_push runs synchronously when its task is created, and pushes and
        # senders are strictly FIFO, so the deque pairs them): a sender task created just
        # before the barge-in but first scheduled after it must still count as
        # pre-interruption work.
        self._interrupt_gen = 0
        self._sender_epochs = deque()
        # THE SLOT: one utterance occupies the connection from its first sendText until its
        # settle. _slot_seq names the owning sequence; _slot_abandoned marks a barge-in on
        # it (its completion then settles silently). Freed ONLY by _settle_slot() (and the
        # establish_connection reset) so the state cannot half-change.
        self._response_idle = asyncio.Event()
        self._response_idle.set()
        self._slot_seq = None
        self._slot_abandoned = False
        # THE WIRE RESPONSE: the server serializes responses on a connection, so at most
        # one is open (responseCreated..responseDone). Whether it serves the slot-owning
        # utterance is decided ONCE, at its responseCreated; audio, done, and error frames
        # all correlate against this single record, so a stale or foreign frame can never
        # touch the active slot.
        self._wire_rid = None
        self._wire_serves_slot = False

    def get_sleep_time(self):
        return 0.01

    # ------------------------------------------------------------------
    # Config validation
    # ------------------------------------------------------------------

    def _validate_options(self):
        """Mirror the server's ranges so a typo fails at agent setup, not on the first turn
        of a live call. The model id is deliberately not validated against a fixed list —
        the catalog (GET /v1/models) evolves server-side."""
        temperature = self.params.get("temperature")
        if temperature is not None and not 0.0 <= float(temperature) <= 1.5:
            raise ValueError("Kalpa temperature must be between 0.0 and 1.5")
        acoustic = self.params.get("acoustic_temperature")
        if acoustic is not None and not 0.0 <= float(acoustic) <= 1.5:
            raise ValueError("Kalpa acoustic_temperature must be between 0.0 and 1.5")
        max_new_tokens = self.params.get("max_new_tokens")
        if max_new_tokens is not None and not 16 <= int(max_new_tokens) <= 2048:
            raise ValueError("Kalpa max_new_tokens must be between 16 and 2048")
        audio_quality = self.params.get("audio_quality")
        if audio_quality is not None and audio_quality not in AUDIO_QUALITIES:
            raise ValueError(f"Kalpa audio_quality must be one of {sorted(AUDIO_QUALITIES)}")
        schedule = self.chunk_length_schedule
        if not 1 <= len(schedule) <= 10 or any(not 50 <= int(t) <= 2000 for t in schedule):
            raise ValueError("Kalpa chunk_length_schedule must be 1-10 thresholds between 50 and 2000")

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
        """Runs synchronously in push order (before the sender task): stamp the interrupt
        epoch this push belongs to, and detect supersession — a new turn starting while an
        older utterance is still open means that turn was retired without an interruption.
        Mark it dead so its stragglers drop instead of appending to this turn's utterance,
        and remember that its text may still sit in the server buffer."""
        self._sender_epochs.append(self._interrupt_gen)
        seq = meta_info.get("sequence_id")
        if self._turn_seq is not None and self._turn_seq != seq and not self._turn_dead:
            logger.info(f"Marking un-flushed Kalpa utterance from superseded seq={self._turn_seq} dead")
            self._turn_dead = True
            if self._turn_chunks and self._turn_conn_gen == self._conn_gen:
                self._dirty_conn_gen = self._conn_gen

    # ------------------------------------------------------------------
    # Voice resolution
    # ------------------------------------------------------------------

    async def _resolve_voice_id(self):
        """Return the opaque voice id, resolving a display name via GET /v1/voices once
        per process (see _VOICE_IDS).

        Names match case-insensitively, on the full name ("Kiara (hindi)") or its base
        before the qualifier ("Kiara") — so agent configs don't have to carry UUIDs.
        """
        if self.voice_id:
            return self.voice_id

        wanted = self.voice.strip().lower()
        cached = _VOICE_IDS.get((self.kalpa_host, wanted))
        if cached:
            self.voice_id = cached
            return cached

        headers = {"Authorization": f"Bearer {self.api_key}"}
        url = f"https://{self.kalpa_host}/v1/voices"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"Kalpa GET /v1/voices failed: {resp.status} - {await resp.text()}")
                body = await resp.json()
        voices = body.get("data", []) if isinstance(body, dict) else body

        matches = [v for v in voices if v.get("name", "").lower() == wanted]
        if not matches:
            matches = [v for v in voices if v.get("name", "").split(" (")[0].lower() == wanted]
        if len(matches) == 1:
            self.voice_id = matches[0]["id"]
            _VOICE_IDS[(self.kalpa_host, wanted)] = self.voice_id
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
        """Abandon the open utterance and whatever the server holds for it.

        The epoch bump comes first: it retires every pending sender at its next await
        boundary, so this cancel cannot race a parked sender into the socket even on
        task-manager paths that interrupt before invalidating the sequence (it is why this
        method is safe without taking the send lock, which a parked sender may hold).

        Then three states, two moves:
        - A response is open on the wire: bare cancelResponse. It aborts the generation
          and wipes buffered text; the response still settles with its own responseDone,
          and _slot_abandoned keeps that settle silent (even a "completed" done when the
          cancel lost the race) and drops its still-arriving audio.
        - The utterance was flushed but its responseCreated hasn't arrived yet: same
          cancel. The flush committed the utterance server-side, so a response (and its
          done) is guaranteed.
        - Text was streamed but never flushed and no response has been seen: a cancel
          alone could settle with no responseDone at all (segmentation decides whether
          the server started rendering), wedging the slot until the idle timeout. Flush
          first: that commits the utterance and reduces this to the previous case — the
          cancel's done frees the slot in milliseconds on the same connection, where a
          socket reset here cost the next turn the full reconnect.
        """
        self._interrupt_gen += 1
        turn_open = self._turn_seq is not None and not self._turn_dead
        self._reset_turn()
        self._dirty_conn_gen = None
        # The cancelled turn's end-of-stream is never forwarded, so the next turn must be
        # re-detected as new for stale text_queue entries to be pruned — even when the
        # cancel send below fails with the socket.
        self.current_turn_start_time = None
        try:
            flush_first = False
            if not self._response_idle.is_set():
                self._slot_abandoned = True
                flush_first = self._wire_rid is None and turn_open
            ws = self.websocket
            if ws is not None and ws.state is websockets.protocol.State.OPEN:
                if flush_first:
                    await ws.send(json.dumps({"type": "sendText", "flush": True}))
                await ws.send(json.dumps({"type": "cancelResponse"}))
                logger.info("Sent cancelResponse to Kalpa TTS WebSocket")
        except Exception as e:
            # The send died with the socket; the receiver's closure path settles the turn.
            logger.error(f"Error handling Kalpa interruption: {e}")

    # ------------------------------------------------------------------
    # sender / receiver
    # ------------------------------------------------------------------

    def _reset_turn(self):
        self._turn_seq = None
        self._turn_chunks = []
        self._turn_chars = 0
        self._turn_truncated = False
        self._turn_dead = False
        self._turn_glue = " "

    async def _send_frame(self, payload):
        """Send one frame. Unlike the shared _send_json, a failure here is not recorded as
        connection_error: a socket dying mid-send is transient — the receiver settles or
        replays the turn and monitor_connection redials — while connection_error is
        reserved for fatal conditions that must stop the whole synthesizer."""
        try:
            await self.websocket.send(json.dumps(payload))
        except Exception as e:
            logger.error(f"Kalpa send failed ({payload.get('type')}): {e}")
            raise

    @staticmethod
    def _truncate_at_word(text, limit):
        """Cut at a word boundary so the caller doesn't hear a clipped syllable."""
        if len(text) <= limit:
            return text
        logger.warning(f"Kalpa text exceeds {limit} chars; truncating at a word boundary")
        cut = text.rfind(" ", 0, limit + 1)
        return text[:cut] if cut > 0 else text[:limit]

    async def _close_ws(self, reason):
        """Deterministic reset: the receiver notices the closure and settles the lost turn,
        monitor_connection redials, and establish_connection reinitializes the state."""
        ws = self.websocket
        if ws is None:
            return
        logger.warning(f"Closing Kalpa TTS socket: {reason}")
        try:
            await ws.close()
        except Exception as e:
            logger.error(f"Error closing Kalpa TTS socket: {e}")

    def _retired(self, sequence_id, interrupt_gen):
        """True once a sender's work must not reach the socket: the conversation ended, an
        interruption ran after the sender started (the epoch moved), or the pipeline
        retired the sequence. Re-checked after every await — a parked sender can resume
        long after the world changed."""
        return (
            self.conversation_ended
            or interrupt_gen != self._interrupt_gen
            or not self.should_synthesize_response(sequence_id)
        )

    async def _wait_for_idle_slot(self, sequence_id, interrupt_gen):
        """Wait for the previous utterance's settle to free the connection; returns
        "free", "retired", or "reset". Settles arrive in milliseconds, so hitting the
        timeout means one that is wedged or can never arrive; the socket is closed to
        reset the session ("reset" — the caller parks on the reconnect, after which this
        turn's text is replayed on the fresh connection)."""
        deadline = time.perf_counter() + RESPONSE_IDLE_TIMEOUT
        while not self._response_idle.is_set():
            if self._retired(sequence_id, interrupt_gen):
                return "retired"
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                await self._close_ws("previous response still unsettled after 10s")
                return "reset"
            try:
                await asyncio.wait_for(self._response_idle.wait(), timeout=remaining)
            except asyncio.TimeoutError:
                pass
        return "free"

    async def _open_utterance(self, sequence_id, interrupt_gen):
        """Ensure `sequence_id` owns an open utterance on a live connection; returns
        (opened, replay). replay=True means this turn's earlier chunks were sent on a
        connection that died — the server lost them, so the caller resends the whole turn."""
        while True:
            if self._retired(sequence_id, interrupt_gen):
                return False, False
            if self._turn_dead and self._turn_seq == sequence_id:
                return False, False
            if self._turn_seq == sequence_id and self._turn_conn_gen == self._conn_gen and self._is_ws_connected():
                return True, False
            await self._wait_for_ws()
            if self.connection_error or self._retired(sequence_id, interrupt_gen):
                return False, False
            if self._turn_dead and not self._response_idle.is_set():
                # A superseded turn holds the slot but can never settle it (it will never
                # flush); run the barge-in triage handle_interruption never got to run:
                # flush first when no response is known (a bare cancel there could settle
                # nothing) — either way the cancel's done frees the slot in milliseconds.
                self._slot_abandoned = True
                self._dirty_conn_gen = None  # the cancel wipes the buffered text too
                try:
                    if self._wire_rid is None:
                        await self._send_frame({"type": "sendText", "flush": True})
                    await self._send_frame({"type": "cancelResponse"})
                except Exception:
                    # the send died with the socket, and with it the dead turn's
                    # residue; park on the redial and retry — bailing out here would
                    # lose the new turn's chunk before anything retained it
                    continue
            # A new utterance (or a replay after a reconnect) claims the connection's slot.
            slot = await self._wait_for_idle_slot(sequence_id, interrupt_gen)
            if slot == "retired" or self._retired(sequence_id, interrupt_gen):
                # the wait can span a barge-in that retires this sequence; the wake-up on
                # the freed slot must not claim it for a turn the pipeline dropped
                return False, False
            if slot == "reset" or not self._is_ws_connected():
                continue  # the socket died (or was reset) while parked; redial and retry
            if self._turn_seq != sequence_id:
                self._reset_turn()
                self._turn_seq = sequence_id
            if self._dirty_conn_gen == self._conn_gen:
                # A superseded turn left un-flushed text in the server buffer; wipe it
                # before this turn's first chunk or the utterances merge.
                self._dirty_conn_gen = None
                try:
                    await self._send_frame({"type": "cancelResponse"})
                except Exception:
                    continue  # died with the socket (which wiped the buffer); redial and retry
            replay = bool(self._turn_chunks)
            self._turn_conn_gen = self._conn_gen
            self._response_idle.clear()
            self._slot_seq = sequence_id
            self._slot_abandoned = False
            # Generation may start at any segment boundary from here on, so synth latency
            # is measured from the turn's first text frame.
            if self.ws_send_time is None:
                self.ws_send_time = time.perf_counter()
            return True, replay

    def _wire_text(self, text, replay):
        """Account `text` against the open utterance and return what should go on the wire.

        The streaming LLM wrappers split chunks with rsplit(" ", 1): when the buffer had a
        space, the boundary space is consumed and must be restored before the next chunk —
        but a chunk with no space at all was an unbreakable token (long URL, number) cut
        mid-way, and the next chunk continues it directly, so gluing a space in would
        mispronounce it. The utterance cap is enforced across the whole turn: the chunk
        that crosses it is cut at a word boundary and everything after is dropped (the
        flush still lands). A replay returns the whole turn so far — the reconnect handed
        us a server with an empty buffer."""
        if not self._turn_truncated:
            piece = (self._turn_glue if self._turn_chars else "") + text
            self._turn_glue = " " if " " in text else ""
            budget = MAX_TEXT_CHARS - self._turn_chars
            if len(piece) > budget:
                piece = self._truncate_at_word(piece, budget)
                self._turn_truncated = True
            if piece:
                self._turn_chunks.append(piece)
                self._turn_chars += len(piece)
        else:
            piece = ""
        return "".join(self._turn_chunks) if replay else piece

    async def sender(self, text, sequence_id, end_of_llm_stream=False):
        try:
            # The epoch was stamped at push time (see _on_push): an interruption bumps it
            # first, so a sender pushed before the barge-in retires even when its task is
            # first scheduled after the bump and the sequence is only invalidated later.
            # (Direct calls without a push fall back to the live epoch.)
            interrupt_gen = self._sender_epochs.popleft() if self._sender_epochs else self._interrupt_gen
            if self._retired(sequence_id, interrupt_gen):
                logger.info(f"Not synthesizing: sequence_id {sequence_id} not current")
                return

            async with self._send_lock:
                # The lock wait can span a barge-in; nothing may reach the socket after it.
                if self._retired(sequence_id, interrupt_gen):
                    logger.info(f"Not synthesizing (post-lock): sequence_id {sequence_id} not current")
                    return

                if text and not (self._turn_dead and self._turn_seq == sequence_id):
                    opened, replay = await self._open_utterance(sequence_id, interrupt_gen)
                    if not opened:
                        return
                    wire_text = self._wire_text(text, replay)
                    if wire_text:
                        try:
                            await self._send_frame({"type": "sendText", "text": wire_text})
                        except Exception:
                            # The socket died mid-send; the chunk stays in _turn_chunks, so
                            # the next send (or this call's flush below) replays the turn
                            # on the fresh connection.
                            pass

                if end_of_llm_stream:
                    await self._finish_utterance(sequence_id, interrupt_gen)

        except asyncio.CancelledError:
            logger.info("Kalpa sender task was cancelled.")
        except Exception as e:
            logger.error(f"Unexpected error in Kalpa sender: {e}")

    async def _finish_utterance(self, sequence_id, interrupt_gen):
        """The LLM turn ended: flush the utterance. A turn that never opened (the LLM's last
        buffer is often empty) has nothing to flush; a dead one is finally forgotten."""
        if self._turn_seq != sequence_id:
            return
        if self._turn_dead:
            self._reset_turn()
            return
        opened, replay = await self._open_utterance(sequence_id, interrupt_gen)
        if not opened:
            return
        frame = {"type": "sendText", "flush": True}
        if replay:
            frame["text"] = "".join(self._turn_chunks)
        self.last_text_sent = True
        try:
            await self._send_frame(frame)
        except Exception:
            # Died at the flush: the receiver's settle path terminates the turn (the slot
            # stays claimed until then, so nothing else flushes into the gap).
            pass
        self._reset_turn()

    def _sentinel_owns_queue_head(self):
        """A completion sentinel is positional: the shared stream generator stamps it onto
        the next queued meta_info. If a newer turn's metadata is already queued (its pushes
        enqueue synchronously while its sender parks on the slot), the settling turn's own
        metas were consumed by its audio — emitting would mark the newer turn complete
        before it renders, so the settling turn is dropped without a completion instead."""
        if not self.text_queue:
            return True
        head_seq = self.text_queue[0].get("sequence_id")
        if head_seq == self._slot_seq:
            return True
        logger.warning(
            f"Suppressing Kalpa end-of-stream for lost seq={self._slot_seq}: queue head belongs to seq={head_seq}"
        )
        return False

    def _settle_slot(self, emit):
        """THE transition that frees the connection slot — every settle path (done, error,
        socket death) funnels through here so the state cannot half-change. Returns 1 when
        the settling turn's end-of-stream sentinel should be emitted: never for an
        abandoned (barged-in) turn — its pipeline already dropped it — and never when the
        queue head already belongs to a newer turn (the sentinel is positional)."""
        if self._response_idle.is_set():
            return 0
        emit = emit and not self._slot_abandoned and self._sentinel_owns_queue_head()
        self._response_idle.set()
        self._slot_seq = None
        self._slot_abandoned = False
        return 1 if emit else 0

    def _settle_lost_utterance(self):
        """The socket died. Settle whatever it was carrying and return how many
        end-of-stream sentinels to emit (0 or 1):

        - a flushed utterance awaiting its responseDone settles now (an abandoned one
          settles silently via _settle_slot);
        - an open utterance whose response had started already played audio; replaying its
          text would repeat what the caller heard, so the turn ends here instead — it is
          marked dead and settled;
        - an open utterance with no response yet lost nothing audible: keep its text for
          replay and settle silently — the turn is still live."""
        emit = False
        if not self._response_idle.is_set():
            if self._turn_seq is None:
                emit = True
            elif not self._turn_dead and self._wire_rid is not None and self._wire_serves_slot:
                self._turn_dead = True
                emit = True
        lost = self._settle_slot(emit)
        self._wire_rid = None
        self._wire_serves_slot = False
        return lost

    async def receiver(self):
        not_connected_since = None
        ws = None
        while True:
            try:
                if self.conversation_ended:
                    return
                if not self._is_ws_connected():
                    # A socket that dies between recvs is seen here, not by the
                    # ConnectionClosed handler below — settle its turn here too.
                    for _ in range(self._settle_lost_utterance()):
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

                # Pin the iteration to the socket it reads from: a closed socket drains its
                # buffered frames before raising ConnectionClosed, so after a reset this
                # loop can still be reading the replaced socket while the new connection
                # already serves the next turn. A stale done freeing (and "completing") the
                # newer turn's slot is exactly the corruption this guard prevents.
                ws = self.websocket
                response = await ws.recv()
                if ws is not self.websocket:
                    logger.info("Dropping frame from a replaced Kalpa socket")
                    continue
                data = self._loads_event(response)
                if data is None:
                    continue

                event = data.get("type")
                if event == "responseAudio":
                    # Only the open wire response, and only while it serves a live
                    # (un-abandoned) slot: stale, foreign, orphaned, or barged-in audio
                    # never plays.
                    if data.get("response_id") != self._wire_rid or not self._wire_serves_slot or self._slot_abandoned:
                        continue
                    chunk = self._decode_audio(data.get("pcm_b64"))
                    if chunk:
                        yield chunk
                elif event == "responseDone":
                    rid = data.get("response_id")
                    status = data.get("status")
                    logger.info(f"Kalpa responseDone status={status} response_id={rid}")
                    if rid is None or rid != self._wire_rid:
                        # created always precedes done on this ordered socket, so a done
                        # that does not name the open wire response is stale or foreign —
                        # it must not free, or positionally complete, a slot a newer turn
                        # may own. If it strands the connection, the idle timeout resets it.
                        logger.warning(f"Ignoring Kalpa responseDone for unmatched response_id={rid}")
                        continue
                    serves = self._wire_serves_slot
                    self._wire_rid = None
                    self._wire_serves_slot = False
                    # A "cancelled" done (or any done of an abandoned turn, even
                    # "completed" when the cancel lost the race) settles silently.
                    if serves and self._settle_slot(emit=status == "completed"):
                        yield b"\x00"
                elif event == "error":
                    err = data.get("error") or {}
                    if data.get("fatal"):
                        # The server closes the socket after a fatal error; ConnectionClosed
                        # is what ends this loop, and auth failures won't fix themselves.
                        logger.error(f"Kalpa fatal error: {err}")
                        if err.get("type") == "authentication_error":
                            self.connection_error = err.get("message") or "authentication error"
                    else:
                        logger.error(f"Kalpa TTS error: {err}")
                        rid = data.get("response_id")
                        if rid is not None:
                            # Response-scoped: only the open wire response can be killed —
                            # a delayed error for an already-settled response touches
                            # nothing.
                            if rid == self._wire_rid:
                                serves = self._wire_serves_slot
                                self._wire_rid = None
                                self._wire_serves_slot = False
                                if self._turn_seq is not None:
                                    self._turn_dead = True  # drop the broken turn's stragglers
                                if serves and self._settle_slot(emit=True):
                                    yield b"\x00"
                        elif not self._response_idle.is_set() and self._wire_rid is None:
                            # Connection-scoped rejection (e.g. a rejected flush) with no
                            # response to carry the settle: the slot-owning utterance can
                            # never finish — end it. With a response open its own done
                            # still arrives, and when idle a sentinel would pop the next
                            # turn's meta_info, so both just log.
                            if self._turn_seq is not None:
                                self._turn_dead = True
                            if self._settle_slot(emit=True):
                                yield b"\x00"
                elif event == "sessionCreated":
                    # Normally consumed inside establish_connection; tolerated here in case
                    # a future server pushes an unsolicited session update.
                    self.native_sample_rate = int(data.get("sample_rate") or self.native_sample_rate)
                elif event == "responseCreated":
                    # With segmentation this can arrive well before the flush. Whether the
                    # response serves the slot-owning utterance is decided HERE, once: the
                    # server serializes responses, so a response created while our slot is
                    # claimed can only be ours (and one created while the slot is idle is
                    # an orphan whose frames must not play). A barge-in after this point
                    # flips _slot_abandoned rather than rewriting this record.
                    self._wire_rid = data.get("response_id")
                    self._wire_serves_slot = not self._response_idle.is_set()
                    logger.info(
                        f"Kalpa responseCreated response_id={self._wire_rid} serves_slot={self._wire_serves_slot}"
                    )
                else:
                    logger.info(f"Ignoring Kalpa event: {data}")

            except websockets.exceptions.ConnectionClosed:
                if ws is not None and ws is not self.websocket:
                    # A socket we already replaced finished dying; the live connection's
                    # state was reset at establish, so there is nothing to settle here.
                    logger.info("Kalpa: replaced socket finished closing")
                    continue
                logger.info("Kalpa WebSocket connection closed")
                # A turn that dies with the socket never gets its responseDone;
                # monitor_connection re-establishes the socket. Keep looping rather than
                # returning: SynthesizerPool iterates generate() exactly once, so a
                # receiver that ends on a transient closure would leave that language
                # silent for the rest of the call.
                for _ in range(self._settle_lost_utterance()):
                    yield b"\x00"
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
        message = {
            "type": "initializeConnection",
            "api_key": self.api_key,
            "generation_config": {"chunk_length_schedule": self.chunk_length_schedule},
        }
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
            # A fresh connection has nothing in flight; an open turn's text is NOT reset —
            # its sender sees the new generation and replays it here.
            self._conn_gen += 1
            self._response_idle.set()
            self._slot_seq = None
            self._slot_abandoned = False
            self._wire_rid = None
            self._wire_serves_slot = False
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

        # Same cap as the streaming path: the API rejects longer utterances outright, and
        # in the non-streaming loop that rejection would silently mute the whole turn.
        payload = {"text": self._truncate_at_word(text, MAX_TEXT_CHARS)}
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
