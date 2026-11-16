"""
Kalpa Labs TTS: streaming WSS /v1/tts/{voice_id}/stream, one-shot POST /v1/tts/{voice_id}.

Kalpa's conversational speech models render named voices from GET /v1/voices

Models:
1. kalpa-tts-1 (the default) speaks English and Hindi, including code-switched
Hinglish with no language parameter
2. kalpa-tts-beta-v0.1 is English-only.
kalpa-tts-multilingual-beta-v0.1 is deprecated; Kalpa serves it with kalpa-tts-1.

Flow
1. The client authenticates with an initializeConnection frame (the handshake itself is
   unauthenticated). Its generation_config.chunk_length_schedule opts the connection into
   server-side segmentation: once buffered text crosses the schedule's next threshold and ends
   at a complete sentence (or, past twice the threshold, a clause or word gap), that part
   starts rendering while the rest is still arriving.
2. The server confirms with sessionCreated.
3. LLM chunks are streamed as sendText frames the moment they arrive; flush=true (sent at
   end_of_llm_stream) ends the utterance and renders whatever remains.
4. Audio arrives as responseAudio frames carrying base64 raw 24 kHz mono s16le PCM.
   The whole utterance stays one response on the wire — one responseCreated, one responseDone —
   but its first audio can arrive long before the flush.
5. A flushed utterance ends with one responseDone: "completed", or "cancelled" when a
   cancelResponse stopped it or the server gave up on it (an error naming the response comes
   first, sometimes before its responseCreated). An utterance that fails before its flush gets
   a fatal error and the socket closes.
6. Every cancelResponse is answered with exactly one cancelled frame, after the last frame of
   whatever it stopped; the connection is reused for the next utterance only after that.
7. For telephony the 24 kHz audio is resampled to 8 kHz here — one continuous resampler per
   turn, so chunk seams don't click — and mu-law encoded.
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
from bolna.helpers.utils import StreamingPcmResampler, audio_to_mulaw8k, pcm_to_ulaw, resample
from bolna.memory.cache.inmemory_scalar_cache import InmemoryScalarCache

logger = configure_logger(__name__)

# The session's true rate arrives in sessionCreated; this is the documented default.
KALPA_NATIVE_SAMPLE_RATE = 24000
MULAW_SAMPLE_RATE = 8000

KALPA_DEFAULT_MODEL = "kalpa-tts-1"

# Kalpa rejects utterances longer than this; we truncate rather than fail a live turn.
MAX_TEXT_CHARS = 8000

# Buffered-character thresholds before each successive part of an utterance may start
# rendering (the last value repeats). The same schedule ElevenLabs runs in this codebase, and
# also Kalpa's server-side default — sent explicitly, because omitting generation_config
# altogether falls back to generate-only-on-flush and loses the early first audio.
DEFAULT_CHUNK_LENGTH_SCHEDULE = [50, 80, 120, 150]

# How long a new utterance waits on the previous response while the server sends nothing. A
# response that is still generating keeps sending frames and is never cut; silence this long
# means the connection's state is wedged or unknowable, so the socket is closed to reset it
# (the receiver settles the lost turn, monitor_connection redials) rather than flushing into a
# session we no longer understand.
RESPONSE_IDLE_TIMEOUT = 10.0
# While waiting, how often to check whether the pipeline has dropped the response holding the
# connection without an interruption (e.g. a new transcript cancelled its LLM turn); such a
# response is cancelled rather than waited out.
SLOT_OWNER_CHECK_INTERVAL = 0.25

# An "overloaded" refusal of a flushed utterance that has played nothing yet is resent after a
# pause, this many times; past that the turn ends.
OVERLOADED_RETRIES = 1
OVERLOADED_RETRY_DELAY = 1.0

# One-shot renders retry 408, 429, 502 and 503 responses, honouring Retry-After up to
# the cap so a live turn is never parked for long.
HTTP_RETRY_STATUSES = frozenset({408, 429, 502, 503})
HTTP_RETRIES = 2
HTTP_RETRY_MAX_DELAY = 2.0

# Failures of the account itself: no retry or reconnect can fix them.
ACCOUNT_ERRORS = frozenset({"authentication_error", "allowance_exhausted", "account_suspended"})
# Init rejections that fail every retry the same way (bad key, model or voice; account blocked).
PERMANENT_INIT_ERRORS = ACCOUNT_ERRORS | {"invalid_request", "not_found"}

AUDIO_QUALITIES = {"low", "medium", "high"}

# Voice display names resolve to catalog ids once per process, not once per call: the task
# manager builds a fresh synthesizer for every call, and the GET /v1/voices round trip sits
# on the connect path. The catalog is global (never key-scoped), so (host, model, name) is the
# whole key; failures are never cached, so a typo keeps failing loudly and a voice added to
# the catalog later is picked up without a restart.
_VOICE_IDS = {}  # (host, model, lowercased display name) -> voice id


def _retry_delay(retry_after, attempt):
    """Retry-After seconds when the server sent them, else a short backoff; capped either way."""
    try:
        delay = float(retry_after)
    except (TypeError, ValueError):
        delay = 0.5 * 2**attempt
    return min(max(delay, 0.0), HTTP_RETRY_MAX_DELAY)


class KalpaSynthesizer(StreamSynthesizer):
    def __init__(
        self,
        voice=None,
        voice_id=None,
        model=KALPA_DEFAULT_MODEL,
        temperature=None,
        acoustic_temperature=None,
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
        # One resampler carried across a turn's chunks (reset when its slot frees): resampling
        # each chunk on its own leaves a click at every chunk seam.
        self._resampler = None
        self._resampler_rate = None

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
        # Its flush is on the wire. Set before the send returns: the receiver can already see
        # the response's outcome while the send is still awaiting.
        self._turn_flushed = False
        self._turn_glue = " "  # what the wrappers' split consumed before the next chunk
        # A connection generation stamps which socket the turn's text went to; a mismatch at
        # send time means the server lost its buffer and the whole turn must be resent.
        self._conn_gen = 0
        self._turn_conn_gen = 0
        # A turn abandoned without a cancel (superseded, or killed by an error) can leave
        # un-flushed text in the server buffer; the generation it happened on gates a bare
        # cancel that wipes it before the next turn.
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
        # it (it then settles silently, on its cancel's acknowledgement). Claimed only by
        # _claim_slot() and freed only by _settle_slot() (and the establish_connection reset),
        # so the state cannot half-change.
        self._response_idle = asyncio.Event()
        self._response_idle.set()
        self._slot_seq = None
        self._slot_claims = 0  # counts claims: canned turns share sequence_id -1
        self._slot_abandoned = False
        self._slot_text = None  # the flushed utterance's text, kept for an overloaded resend
        self._slot_audio = False  # whether any of the slot's audio has played
        self._slot_retries = 0
        # A flushed utterance lost with its socket before any audio played keeps the slot and is
        # resent once on the next connection: _slot_carried spends that once, _slot_carry_due
        # marks the resend as still owed.
        self._slot_carried = False
        self._slot_carry_due = False
        # THE WIRE RESPONSE: the server serializes responses on a connection, so at most
        # one is open (responseCreated..responseDone). Whether it serves the slot-owning
        # utterance is decided ONCE, at its responseCreated; audio, done, and error frames
        # all correlate against this single record, so a stale or foreign frame can never
        # touch the active slot.
        self._wire_rid = None
        self._wire_serves_slot = False
        # cancelResponse frames sent on this connection and not yet answered. Kalpa sends each
        # "cancelled" after the last frame of whatever that cancel stopped, so a response frame
        # arriving while one is outstanding is residue from before the cancel.
        self._acks_pending = 0
        # Responses already done or failed on this connection: a later frame naming one (a
        # failed response's trailing created/done) must not bind to, or settle, the slot.
        self._finished_rids = set()
        self._last_frame_at = 0.0  # when the live socket last sent anything
        # Sequences whose end-of-stream the receiver still owes: one a redial settled, or one
        # whose queue head already belongs to a newer turn (see _settle_slot).
        self._lost_eos = []
        self._retry_task = None

    def get_sleep_time(self):
        return 0.01

    async def cleanup(self):
        if self._retry_task is not None:
            self._retry_task.cancel()  # a pending overloaded resend must not outlive the call
        await super().cleanup()

    async def close_connection(self):
        """A multilingual leg switching away closes on purpose: a reply on the socket is ended as
        lost, not resent whenever the leg reopens."""
        self._slot_carried = True
        if self._slot_carry_due:
            self._slot_carry_due = False
            lost_seq = self._slot_seq
            if self._settle_slot(emit=True):
                self._lost_eos.append(lost_seq)
        await super().close_connection()

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
        """Resample Kalpa's 24 kHz PCM to the target rate, continuously across the turn's
        chunks; mu-law encode for telephony."""
        if not chunk:
            return None
        try:
            if self._resampler is None or self._resampler_rate != self.native_sample_rate:
                self._resampler = StreamingPcmResampler(self.native_sample_rate, self.target_sample_rate)
                self._resampler_rate = self.native_sample_rate
            audio = self._resampler.process(chunk)
            if not audio:
                return None  # a tiny chunk can be held back whole until the next one
            return pcm_to_ulaw(audio) if self.use_mulaw else audio
        except Exception as e:
            logger.error(f"Error converting Kalpa audio: {e}")
            return None

    def _on_push(self, meta_info, text):
        """Runs synchronously in push order (before the sender task): stamp the interrupt
        epoch this push belongs to, and detect supersession — a new turn starting while an
        older utterance is still open means that turn was retired without an interruption.
        Mark it dead so its stragglers drop instead of appending to this turn's utterance."""
        self._sender_epochs.append(self._interrupt_gen)
        seq = meta_info.get("sequence_id")
        if self._turn_seq is not None and self._turn_seq != seq and not self._turn_dead and not self._turn_flushed:
            logger.info(f"Marking un-flushed Kalpa utterance from superseded seq={self._turn_seq} dead")
            self._kill_open_turn()

    def _kill_open_turn(self):
        """The open utterance can never finish: drop its stragglers, and remember that its
        un-flushed text may still sit in the server buffer."""
        if self._turn_seq is None or self._turn_dead or self._turn_flushed:
            return
        self._turn_dead = True
        if self._turn_chunks and self._turn_conn_gen == self._conn_gen:
            self._dirty_conn_gen = self._conn_gen

    # ------------------------------------------------------------------
    # Voice resolution
    # ------------------------------------------------------------------

    async def _resolve_voice_id(self):
        """Return the opaque voice id, resolving a display name via GET /v1/voices?model=
        once per process (see _VOICE_IDS).

        Names match case-insensitively, on the full name ("Kiara (hindi)") or its base
        before the qualifier ("Kiara") — so agent configs don't have to carry UUIDs. With a
        model configured, only the voices offered for it are candidates, so a voice the model
        cannot render fails here, with the model's own voice list.
        """
        if self.voice_id:
            return self.voice_id

        wanted = self.voice.strip().lower()
        cache_key = (self.kalpa_host, self.model, wanted)
        cached = _VOICE_IDS.get(cache_key)
        if cached:
            self.voice_id = cached
            return cached

        headers = {"Authorization": f"Bearer {self.api_key}"}
        url = f"https://{self.kalpa_host}/v1/voices"
        params = {"model": self.model} if self.model else None
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url, headers=headers, params=params, timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status != 200:
                    message = f"Kalpa GET /v1/voices failed: {resp.status} - {await resp.text()}"
                    # A bad key or model, or a blocked account, fails every retry the same way.
                    if 400 <= resp.status < 500 and resp.status not in HTTP_RETRY_STATUSES:
                        raise ValueError(message)
                    raise RuntimeError(message)
                body = await resp.json()
        voices = body.get("data", []) if isinstance(body, dict) else body

        matches = [v for v in voices if v.get("name", "").lower() == wanted]
        if not matches:
            matches = [v for v in voices if v.get("name", "").split(" (")[0].lower() == wanted]
        if len(matches) == 1:
            self.voice_id = matches[0]["id"]
            _VOICE_IDS[cache_key] = self.voice_id
            logger.info(f"Resolved Kalpa voice {self.voice!r} -> {matches[0]['name']!r} ({self.voice_id})")
            return self.voice_id

        names = sorted(v.get("name", "") for v in voices)
        if not matches:
            raise ValueError(f"Unknown Kalpa voice {self.voice!r} for model {self.model!r}. Available: {names}")
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

        Then one bare cancelResponse: Kalpa stops the in-flight response, drops buffered text
        and queued flushes, and answers with "cancelled" after the stopped response's last
        frame — or at once when nothing was in flight. Until that answer the connection is
        abandoned: an occupied slot's still-arriving audio is dropped, and no new utterance
        starts.
        """
        self._interrupt_gen += 1
        self._reset_turn()
        self._dirty_conn_gen = None  # the cancel wipes the buffered text too
        # The cancelled turn's end-of-stream is never forwarded, so the next turn must be
        # re-detected as new for stale text_queue entries to be pruned — even when the
        # cancel send below fails with the socket.
        self.current_turn_start_time = None
        ws = self.websocket
        if ws is None or ws.state is not websockets.protocol.State.OPEN:
            if not self._response_idle.is_set():
                self._slot_abandoned = True  # the receiver's closure path settles it silently
            return
        held = self._abandon_slot()
        try:
            await self._send_cancel()
            logger.info("Sent cancelResponse to Kalpa TTS WebSocket")
        except Exception as e:
            # The send died with the socket; the receiver's closure path settles the turn.
            logger.error(f"Error handling Kalpa interruption: {e}")
            self._release_hold(held)

    # ------------------------------------------------------------------
    # sender / receiver
    # ------------------------------------------------------------------

    def _reset_turn(self):
        self._turn_seq = None
        self._turn_chunks = []
        self._turn_chars = 0
        self._turn_truncated = False
        self._turn_dead = False
        self._turn_flushed = False
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

    async def _send_cancel(self):
        """A bare cancelResponse, counted before it leaves so the receiver can tell which
        response frames precede its answer."""
        self._acks_pending += 1
        try:
            await self._send_frame({"type": "cancelResponse"})
        except Exception:
            # floor: the answer can land while a dying socket's send is still raising
            self._acks_pending = max(0, self._acks_pending - 1)
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
        "free", "retired", or "reset". A response still generating keeps the wait alive with
        its frames; RESPONSE_IDLE_TIMEOUT of silence means one that is wedged or can never
        settle, and the socket is closed to reset the session ("reset" — the caller parks on
        the reconnect, after which this turn's text is replayed on the fresh connection)."""
        started = time.perf_counter()
        while not self._response_idle.is_set():
            if self._retired(sequence_id, interrupt_gen):
                return "retired"
            await self._cancel_dropped_owner()
            remaining = max(started, self._last_frame_at) + RESPONSE_IDLE_TIMEOUT - time.perf_counter()
            if remaining <= 0:
                await self._close_ws(f"previous response silent for {RESPONSE_IDLE_TIMEOUT:g}s")
                return "reset"
            try:
                await asyncio.wait_for(self._response_idle.wait(), timeout=min(remaining, SLOT_OWNER_CHECK_INTERVAL))
            except asyncio.TimeoutError:
                pass
        return "free"

    async def _cancel_dropped_owner(self):
        """Some task-manager paths drop a sequence without an interruption (a new transcript
        cancelling the LLM turn). A response the pipeline no longer wants would otherwise hold
        the connection for the rest of its generation while its audio is discarded downstream:
        cancel it the way handle_interruption would — the answer frees the slot in milliseconds."""
        if (
            self._slot_abandoned
            or self._slot_seq is None
            or self.should_synthesize_response(self._slot_seq)
            or not self._is_ws_connected()
        ):
            return
        logger.info(f"Cancelling Kalpa response for dropped seq={self._slot_seq}")
        self._abandon_slot()
        self._dirty_conn_gen = None  # the cancel wipes the buffered text too
        try:
            await self._send_cancel()
        except Exception:
            pass  # died with the socket; the closure settles the abandoned slot silently

    def _claim_slot(self, sequence_id):
        self._response_idle.clear()
        self._slot_seq = sequence_id
        self._slot_claims += 1
        self._slot_abandoned = False
        self._slot_text = None
        self._slot_audio = False
        self._slot_retries = 0
        self._slot_carried = False
        self._slot_carry_due = False

    def _free_slot(self):
        self._response_idle.set()
        self._slot_seq = None
        self._slot_abandoned = False
        self._slot_text = None
        self._slot_audio = False
        self._slot_retries = 0
        self._slot_carried = False
        self._slot_carry_due = False
        if self._resampler is not None:
            # The next turn's audio must not start with this one's held-back samples (~1 ms at
            # 24 -> 8 kHz, dropped rather than flushed, as for Maya and Sarvam).
            self._resampler.reset()

    def _abandon_slot(self):
        """Before a cancel goes out: drop an occupied slot's still-arriving frames, or hold an
        idle connection, so that no new utterance starts before the cancel is answered — the
        point at which Kalpa documents the connection as reusable. Returns the hold's claim
        token, or None when the slot was already occupied."""
        held = None
        if self._response_idle.is_set():
            self._claim_slot(None)
            held = self._slot_claims
        self._slot_abandoned = True
        return held

    def _release_hold(self, held):
        """The cancel a hold was taken for never left: no answer will free it."""
        if held is not None and held == self._slot_claims and not self._acks_pending:
            self._settle_slot(emit=False)

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
            if self._turn_dead and not self._response_idle.is_set() and not self._slot_abandoned:
                # A superseded turn holds the slot but can never settle it (it will never
                # flush); cancel it the way handle_interruption would have — the answer
                # frees the slot in milliseconds. (An abandoned slot has its cancel out already.)
                self._abandon_slot()
                self._dirty_conn_gen = None  # the cancel wipes the buffered text too
                try:
                    await self._send_cancel()
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
            if self._turn_dead and self._turn_seq == sequence_id:
                return False, False  # the receiver ended this turn while it was parked
            if self._dirty_conn_gen == self._conn_gen:
                # An abandoned turn left un-flushed text in the server buffer: wipe it, then
                # wait for the answer, before this turn's first chunk — or the utterances merge.
                self._dirty_conn_gen = None
                held = self._abandon_slot()
                try:
                    await self._send_cancel()
                except Exception:
                    self._release_hold(held)  # died with the socket, which wiped the buffer too
                continue
            if self._turn_seq != sequence_id:
                self._reset_turn()
                self._turn_seq = sequence_id
            replay = bool(self._turn_chunks)
            self._turn_conn_gen = self._conn_gen
            self._claim_slot(sequence_id)
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
        buffer is often empty) has nothing to flush; a dead one is forgotten once it no longer
        holds the slot (until then its marker makes the next turn cancel it)."""
        if self._turn_seq != sequence_id:
            return
        if self._turn_dead:
            if self._response_idle.is_set():
                self._reset_turn()
            return
        while True:
            opened, replay = await self._open_utterance(sequence_id, interrupt_gen)
            if not opened:
                return
            text = "".join(self._turn_chunks)
            frame = {"type": "sendText", "flush": True}
            if replay:
                frame["text"] = text
            self.last_text_sent = True
            self._slot_text = text  # kept until the response plays: an overloaded refusal resends it
            self._turn_flushed = True
            try:
                await self._send_frame(frame)
            except Exception:
                self._turn_flushed = False
                if (self._response_idle.is_set() and self._turn_conn_gen != -1) or self._slot_retries:
                    # The receiver ended the turn while the send was in flight (its done or
                    # error, or a heard turn's socket death), or a refusal's resend owns its
                    # text: nothing is left to replay.
                    self._reset_turn()
                    return
                # Died at the flush. The turn keeps its text, so once the receiver settles the
                # lost socket the next connection replays it whole — unless its audio had
                # started, in which case the receiver ends it instead.
                self._turn_conn_gen = -1
                continue
            if self._turn_conn_gen == -1:
                # The receiver settled the dead socket under the send and nothing was heard:
                # the send returned, but the frame is lost with the socket — replay it.
                self._turn_flushed = False
                continue
            self._reset_turn()
            return

    def _schedule_resend(self):
        """Resend the slot's flushed text after OVERLOADED_RETRY_DELAY. The slot stays claimed
        meanwhile, so later turns keep their order behind it.

        No send lock: a later turn's sender holds it while parked on this very slot. None is
        needed either — while the claim stands no other sender can put text on the wire, and a
        barge-in's cancel (lock-free too) is seen by the checks right before the send."""
        seq, claim, text = self._slot_seq, self._slot_claims, self._slot_text
        conn_gen, interrupt_gen = self._conn_gen, self._interrupt_gen
        logger.warning(f"Kalpa is overloaded; resending seq={seq} in {OVERLOADED_RETRY_DELAY:g}s")

        async def resend():
            await asyncio.sleep(OVERLOADED_RETRY_DELAY)
            if self._response_idle.is_set() or self._slot_claims != claim or self._slot_abandoned:
                return  # settled meanwhile (a barge-in, a lost socket, the end of the call)
            if self._conn_gen != conn_gen or not self._is_ws_connected():
                return  # the closure path settles it
            if self._retired(seq, interrupt_gen):
                self._settle_slot(emit=False)  # the pipeline dropped the turn; nothing to end
                return
            try:
                await self._send_frame({"type": "sendText", "text": text, "flush": True})
            except Exception:
                pass  # the socket died; its closure settles the slot

        self._retry_task = asyncio.create_task(resend())

    def _sentinel_owns_queue_head(self, sequence_id):
        """A bare completion sentinel is positional: the shared stream generator stamps it onto
        the next queued meta_info. If a newer turn's metadata is already queued (its pushes
        enqueue synchronously while its sender parks on the slot), the settling turn's own
        metas were consumed by its audio, and a bare sentinel would mark the newer turn
        complete before it renders."""
        return not self.text_queue or self.text_queue[0].get("sequence_id") == sequence_id

    def _eos_for(self, sequence_id):
        """The end-of-stream item for a settling turn: the bare sentinel when the queue head is
        its own, else one carrying its sequence, which the shared generator emits under that
        sequence without taking the newer turn's meta (as Sarvam does)."""
        if self._sentinel_owns_queue_head(sequence_id):
            return b"\x00"
        logger.info(f"Ending Kalpa seq={sequence_id} under its own sequence: a newer turn heads the queue")
        return b"\x00", {"retired_sequence_id": sequence_id}

    def _unpack_receiver_message(self, item):
        return item if isinstance(item, tuple) else (item, {})

    def _settle_slot(self, emit):
        """THE transition that frees the connection slot — every settle path (done, error,
        cancel answer, socket death) funnels through here so the state cannot half-change.
        Returns 1 when the settling turn's bare end-of-stream sentinel should be emitted: never
        for an abandoned (barged-in) turn — its pipeline already dropped it. When a newer turn
        already heads the queue, the receiver emits it under the turn's own sequence instead."""
        if self._response_idle.is_set():
            return 0
        emit = emit and not self._slot_abandoned
        if emit and not self._sentinel_owns_queue_head(self._slot_seq):
            if self._slot_seq is not None:
                self._lost_eos.append(self._slot_seq)
            emit = False
        self._free_slot()
        return 1 if emit else 0

    def _fail_slot(self, error_type):
        """The slot's response failed; returns the sentinels to emit (0 or 1). An overloaded
        refusal of a flushed utterance that has played nothing is resent after a pause;
        anything else ends the turn."""
        if (
            error_type == "overloaded"
            and (self._turn_seq is None or self._turn_flushed)
            and self._slot_text
            and not self._slot_audio
            and self._slot_retries < OVERLOADED_RETRIES
        ):
            self._slot_retries += 1
            self._schedule_resend()
            return 0
        self._kill_open_turn()
        return self._settle_slot(emit=True)

    def _settle_lost_utterance(self):
        """The socket died. Settle whatever it was carrying and return how many
        end-of-stream sentinels to emit (0 or 1):

        - a flushed utterance that played nothing lost nothing audible: it keeps the slot, so
          later turns stay behind it, and establish_connection resends it whole — once;
        - any other flushed utterance awaiting its responseDone settles now (an abandoned one
          settles silently via _settle_slot);
        - an open utterance that already played audio would repeat it if replayed, so the
          turn ends here instead — it is marked dead and settled;
        - an open utterance that played nothing lost nothing audible: keep its text for
          replay and settle silently — the turn is still live."""
        emit = False
        if not self._response_idle.is_set():
            if self._turn_seq is None:
                if self._slot_carry_due:
                    return 0  # already settled for this loss; the resend is owed
                if self._slot_text and not self._slot_audio and not self._slot_abandoned and not self._slot_carried:
                    self._slot_carried = self._slot_carry_due = True
                    self._wire_rid = None
                    self._wire_serves_slot = False
                    return 0
                emit = True
            elif not self._turn_dead and self._slot_audio:
                self._turn_dead = True
                emit = True
            else:
                # Nothing heard: the turn replays on the next connection — also when its flush
                # send is still in flight and returns after this.
                self._turn_conn_gen = -1
        lost = self._settle_slot(emit)
        self._wire_rid = None
        self._wire_serves_slot = False
        return lost

    def _carry_wanted(self):
        """Resend a lost reply only while the pipeline still wants it; a barged-in or dropped one
        just frees the slot (its pipeline already let it go)."""
        return (
            not self._slot_abandoned and not self.conversation_ended and self.should_synthesize_response(self._slot_seq)
        )

    def _handle_cancel_answer(self):
        """One "cancelled" arrived. When it was the last one owed, everything those cancels
        stopped has drained, and an abandoned turn settles silently."""
        self._acks_pending = max(0, self._acks_pending - 1)
        if not self._acks_pending and self._slot_abandoned:
            self._wire_rid = None
            self._wire_serves_slot = False
            self._settle_slot(emit=False)

    async def receiver(self):
        not_connected_since = None
        ws = None
        while True:
            try:
                if self.conversation_ended:
                    return
                while self._lost_eos:
                    yield self._eos_for(self._lost_eos.pop(0))
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
                self._last_frame_at = time.perf_counter()
                data = self._loads_event(response)
                if data is None:
                    continue

                event = data.get("type")
                rid = data.get("response_id")
                if event == "cancelled":
                    self._handle_cancel_answer()
                    continue
                if rid is not None and not data.get("fatal") and (self._acks_pending or rid in self._finished_rids):
                    # From before an outstanding cancel, or from a response that already
                    # finished: it can neither play nor touch the slot.
                    continue

                if event == "responseAudio":
                    # Only the open wire response, and only while it serves a live
                    # (un-abandoned) slot: stale, foreign, orphaned, or barged-in audio
                    # never plays.
                    if rid != self._wire_rid or not self._wire_serves_slot or self._slot_abandoned:
                        continue
                    chunk = self._decode_audio(data.get("pcm_b64"))
                    if chunk:
                        self._slot_audio = True
                        yield chunk
                elif event == "responseDone":
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
                    self._finished_rids.add(rid)
                    if serves:
                        # Every done ends its turn — "cancelled" too when we never asked:
                        # the server gave up on it. Only a barge-in's own turn (abandoned)
                        # settles silently.
                        self._kill_open_turn()
                        if self._settle_slot(emit=True):
                            yield b"\x00"
                elif event == "error":
                    err = data.get("error") or {}
                    error_type = err.get("type")
                    if data.get("fatal"):
                        # The server closes the socket after a fatal error; ConnectionClosed
                        # is what settles the turn, and account failures won't fix themselves.
                        logger.error(f"Kalpa fatal error: {err}")
                        if error_type in ACCOUNT_ERRORS:
                            self.connection_error = err.get("message") or error_type
                        continue
                    logger.error(f"Kalpa TTS error: {err}")
                    if rid is not None:
                        # Response-scoped: only the slot's own response can be failed — a
                        # delayed error for an already-settled response touches nothing.
                        if rid == self._wire_rid:
                            serves = self._wire_serves_slot
                            self._wire_rid = None
                            self._wire_serves_slot = False
                            self._finished_rids.add(rid)
                            if serves:
                                for _ in range(self._fail_slot(error_type)):
                                    yield b"\x00"
                        elif self._wire_rid is None and not self._response_idle.is_set() and not self._slot_abandoned:
                            # A response refused before it started generating is reported before
                            # its responseCreated (a cancelled done follows). Responses are
                            # serialized and every older one has finished or been cancelled,
                            # so an unknown one can only be the slot's own.
                            self._finished_rids.add(rid)
                            for _ in range(self._fail_slot(error_type)):
                                yield b"\x00"
                    elif not self._response_idle.is_set() and self._wire_rid is None and not self._slot_abandoned:
                        # Connection-scoped rejection (e.g. an over-long sendText) with no
                        # response to carry the settle: the slot-owning utterance can never
                        # finish — end it. With a response open its own done still arrives,
                        # an abandoned slot waits for its cancel's answer, and when idle a
                        # sentinel would pop the next turn's meta_info, so these just log.
                        self._kill_open_turn()
                        if self._settle_slot(emit=True):
                            yield b"\x00"
                elif event == "sessionCreated":
                    # Normally consumed inside establish_connection; tolerated here in case
                    # the server ever pushes an unsolicited session update.
                    self.native_sample_rate = int(data.get("sample_rate") or self.native_sample_rate)
                elif event == "responseCreated":
                    # With segmentation this can arrive well before the flush. Whether the
                    # response serves the slot-owning utterance is decided HERE, once: the
                    # server serializes responses, so a response created while our slot is
                    # claimed can only be ours (and one created while the slot is idle is
                    # an orphan whose frames must not play). A barge-in after this point
                    # flips _slot_abandoned rather than rewriting this record.
                    self._wire_rid = rid
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
                # A bad key, model, or voice, or a blocked account, won't fix itself on retry;
                # an inference-side outage (e.g. voice seed temporarily unavailable) might.
                if err.get("type") in PERMANENT_INIT_ERRORS:
                    self.connection_error = message
                await websocket_conn.close()
                return None

            self.native_sample_rate = int(reply.get("sample_rate") or self.native_sample_rate)
            # The redial can finish before the receiver reaches the old socket's closure (which
            # it then skips as replaced): settle the old turn here, while its state still says
            # whether it was heard, so a heard turn is ended rather than replayed.
            lost_seq = self._slot_seq
            if self._settle_lost_utterance():
                self._lost_eos.append(lost_seq)
            # A flushed reply the old socket lost unheard is resent first, in the slot it kept.
            resend = self._slot_text if self._slot_carry_due and self._carry_wanted() else None
            stale = False
            if resend is not None:
                try:
                    await websocket_conn.send(json.dumps({"type": "sendText", "text": resend, "flush": True}))
                    # A barge-in landing while that send was in flight could not cancel on the
                    # old socket; cancel here, so the resend cannot hold the slot unwanted.
                    stale = not self._carry_wanted()
                    if stale:
                        await websocket_conn.send(json.dumps({"type": "cancelResponse"}))
                except Exception:
                    await websocket_conn.close()
                    raise
                logger.info(f"Resent Kalpa seq={self._slot_seq} lost with the previous socket")
            # A fresh connection has nothing else in flight; an open turn's text is NOT reset —
            # its sender sees the new generation and replays it here.
            self._conn_gen += 1
            if resend is None:
                self._free_slot()
            self._slot_carry_due = False
            self._slot_abandoned = self._slot_abandoned or stale
            self._wire_rid = None
            self._wire_serves_slot = False
            self._acks_pending = 1 if stale else 0
            self._finished_rids.clear()
            self._last_frame_at = time.perf_counter()  # sessionCreated: parked turns' idle clocks restart
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
        """POST /v1/tts/{voice_id}; the response JSON carries base64 16-bit PCM WAV.
        429, 502 and 503 responses are retried briefly."""
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
            for attempt in range(HTTP_RETRIES + 1):
                async with session.post(url, headers=headers, json=payload) as response:
                    request_id = response.headers.get("X-Request-ID")
                    if response.status == 200:
                        try:
                            data = await response.json()
                        except Exception as e:
                            logger.error(f"Kalpa TTS HTTP response was not JSON (request_id={request_id}): {e}")
                            return None
                        break
                    logger.error(
                        f"Kalpa TTS HTTP error: {response.status} request_id={request_id} - {await response.text()}"
                    )
                    if response.status not in HTTP_RETRY_STATUSES or attempt == HTTP_RETRIES:
                        return None
                    delay = _retry_delay(response.headers.get("Retry-After"), attempt)
                logger.info(f"Retrying Kalpa TTS in {delay:g}s")
                await asyncio.sleep(delay)
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
