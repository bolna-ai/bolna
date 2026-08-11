"""
Asterisk WebSocket (chan_websocket) output handler for sip-trunk provider.

Sends audio to Asterisk in bursts as it arrives from TTS (like Plivo/Twilio).
Each response is bracketed START_MEDIA_BUFFERING … STOP_MEDIA_BUFFERING: START
lets Asterisk reframe/retime for smooth RTP, STOP flushes the buffer to RTP so
even a short isolated response (e.g. the welcome message) plays immediately
instead of waiting for later audio. FLUSH_MEDIA on interrupt. Generation counter
drops stale audio after interruption.

Playback completion is mark-driven, same as Twilio/Plivo: each chunk is followed by
MARK_MEDIA <mark_id>, which Asterisk queues in-band behind that audio and echoes as
MEDIA_MARK_PROCESSED once it reaches the front of the playout queue (Asterisk 22.6+).
The duration estimate (first_send + total_audio_duration + settle) is kept only as a
fallback for when the echo doesn't arrive, and logs a warning when it has to act — it
runs early, because it assumes playout starts the instant the first frame is sent.

The synthesizer emits the final audio chunk and a trailing null-byte sentinel both
with is_final=True; both reach _schedule_finish, which is harmless because
_response_audio_duration is reset only on a new sequence_id or on interruption
(never on is_final) — so the second call recomputes an equivalent finish delay.

Ref: https://docs.asterisk.org/Configuration/Channel-Drivers/WebSocket/
"""

import asyncio
import os
import time
import uuid
import traceback
import audioop
from collections import deque
from bolna.output_handlers.telephony import TelephonyOutputHandler
from bolna.helpers.logger_config import configure_logger
from dotenv import load_dotenv

logger = configure_logger(__name__)
load_dotenv()

# One frame must arrive within the 10 short reads ws_safe_read() allows or Asterisk drops
# the call; a proxy relaying in ~4 KB chunks costs one read each.
MAX_WS_FRAME_BYTES = int(os.environ.get("SIP_MAX_WS_FRAME_BYTES", "8000"))  # 1 s of ulaw

# Extra buffer after estimated playback end before clearing is_audio_being_played.
# Accounts for Asterisk's internal retiming and RTP jitter buffer.
PLAYBACK_SETTLE_S = float(os.environ.get("SIP_PLAYBACK_SETTLE_S", "0.1"))

# ulaw 8 kHz = 8,000 bytes per second.
ULAW_BYTES_PER_SECOND = 8000

# Maximum send rate as a multiple of real-time playback speed.
# Asterisk's chan_websocket frame queue holds ~1186 frames (~23.7 s of ulaw audio).
# When the queue is full Asterisk silently drops every incoming binary frame
# ("WebSocket queue is full. Ignoring incoming binary message.") instead of
# sending a second MEDIA_XOFF.
# At 1.5× real-time the queue fills at 0.5× rate; for responses under ~47 s the
# queue never reaches capacity.  Longer responses trigger XOFF/XON cycles — the
# drain is also rate-limited (and re-anchored to exclude the XOFF pause time) so
# it never re-overflows the queue.  Set SIP_MAX_SEND_RATE_FACTOR=0 to disable.
MAX_SEND_RATE_FACTOR = float(os.environ.get("SIP_MAX_SEND_RATE_FACTOR", "1.5"))

# Tags for _local_audio_queue entries. Audio and marks share one queue so an XOFF pause
# can't let a mark overtake the audio it belongs to.
AUDIO_ENTRY = "audio"
MARK_ENTRY = "mark"


class SipTrunkOutputHandler(TelephonyOutputHandler):
    """
    Sends ulaw audio to Asterisk in bursts — Asterisk buffers and retimes.
    Playback completion via MARK_MEDIA echo, with duration tracking as fallback.
    FLUSH_MEDIA on interrupt. XOFF/XON for backpressure safety.
    """

    def __init__(
        self,
        io_provider="sip-trunk",
        websocket=None,
        mark_event_meta_data=None,
        log_dir_name=None,
        asterisk_media_start=None,
        agent_config=None,
        input_handler=None,
    ):
        super().__init__(io_provider, websocket, mark_event_meta_data, log_dir_name)
        self.asterisk_media_start = asterisk_media_start or {}
        self.agent_config = agent_config or {}
        self.input_handler = input_handler
        self.queue_full = False
        if input_handler:
            input_handler.output_handler_ref = self

        # Playback duration tracking
        self._response_first_send: float = 0.0  # monotonic time first chunk was sent
        self._response_audio_duration: float = 0.0  # accumulated seconds of audio
        self._bytes_sent: int = 0  # bytes sent this response (rate limiting)
        self._settle_task: asyncio.Task | None = None
        self._pending_finish: bool = False
        self._current_sequence_id: int | None = None  # track sequence to detect new responses

        # Generation counter — incremented on interruption, stale audio is dropped
        self._flush_generation: int = 0

        # Asterisk buffering state
        self._start_buffering_sent: bool = False

        # True while _send_frames is writing a chunk — sends are rate-limited, so a
        # large chunk stays in flight for seconds and teardown must not hang up over it.
        self._send_in_flight: bool = False

        # XOFF/XON: local queue when Asterisk signals backpressure
        self._local_audio_queue: deque = deque()
        # Serialize drains so a second MEDIA_XON can't run a drain concurrently with an
        # in-flight one (which would interleave frames and double-count send pacing).
        self._drain_lock = asyncio.Lock()

        output_config = self._get_output_config()
        self._output_format = (output_config.get("audio_format") or output_config.get("format") or "ulaw").lower()
        if self._output_format not in ("ulaw", "mulaw"):
            self._output_format = "ulaw"

    def _get_output_config(self):
        try:
            tasks = self.agent_config.get("tasks") or []
            if tasks and isinstance(tasks[0], dict):
                return tasks[0].get("tools_config", {}).get("output") or {}
        except Exception:
            pass
        return {}

    # ------------------------------------------------------------------
    # Per-response state
    # ------------------------------------------------------------------

    def _reset_response_counters(self) -> None:
        """Reset the per-response send/duration counters and cancel any pending finish
        timer. Shared by the new-response path (sequence_id change) and handle_interruption
        so both reset the same set — keeping them in sync is what prevents the finish timer
        from being computed off stale duration."""
        self._response_first_send = 0.0
        self._response_audio_duration = 0.0
        self._bytes_sent = 0
        self._pending_finish = False
        if self._settle_task:
            self._settle_task.cancel()
            self._settle_task = None

    # ------------------------------------------------------------------
    # Playback completion — server-side duration tracking
    # ------------------------------------------------------------------

    def _schedule_finish(self, generation: int):
        """Schedule playback completion based on how much audio was sent and when.

        Since audio is sent in bursts (faster than real-time), Asterisk buffers
        it and plays at 1x. Playback ends at approximately:
            first_send_time + total_audio_duration + settle_buffer
        """
        if self._settle_task:
            self._settle_task.cancel()

        elapsed = time.monotonic() - self._response_first_send if self._response_first_send else 0
        remaining = self._response_audio_duration - elapsed
        delay = max(remaining, 0) + PLAYBACK_SETTLE_S

        self._settle_task = asyncio.create_task(self._settle_and_finish(generation, delay))
        logger.info(
            f"sip-trunk: playback finish in {delay:.2f}s "
            f"(audio={self._response_audio_duration:.2f}s, elapsed={elapsed:.2f}s)"
        )

    async def _settle_and_finish(self, generation: int, delay: float):
        """Wait for estimated playback to complete, then clear playback state."""
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            return
        if self._closed or generation != self._flush_generation:
            return
        self._settle_task = None
        self._finish_playback(reason="duration-complete")

    def _finish_playback(self, reason: str = "duration-complete") -> None:
        """Fallback completion: ack whatever Asterisk never echoed and clear playback state.

        Normally a no-op — MEDIA_MARK_PROCESSED completes playback first. It only does real
        work when the mark echo is missing or late, and says so loudly, because acking on an
        estimate is what lets HANGUP cut a goodbye short.
        """
        if not self.input_handler or not self.input_handler.mark_event_meta_data:
            return
        remaining = list(self.input_handler.mark_event_meta_data.mark_event_meta_data.keys())
        if not remaining and not self.input_handler.is_audio_being_played_to_user():
            logger.info(f"sip-trunk: playback completed by mark echo before the {reason} timer")
            return
        logger.warning(
            f"sip-trunk: completing playback from the duration estimate, not Asterisk's mark echo "
            f"(reason={reason}, {len(remaining)} unacked mark(s))"
        )
        for mid in remaining:
            self.input_handler.process_mark_message({"name": mid})
        if self.input_handler.is_audio_being_played_to_user():
            self.input_handler.update_is_audio_being_played(False)

    # ------------------------------------------------------------------
    # Audio framing / sending
    # ------------------------------------------------------------------

    @staticmethod
    def _to_ulaw(audio_chunk: bytes, audio_format: str) -> bytes:
        """Convert a PCM/WAV chunk to 8-bit ulaw, stripping a 44-byte RIFF header if
        present. ulaw input is returned unchanged."""
        if audio_format in ("pcm", "wav") or (len(audio_chunk) > 44 and audio_chunk[:4] == b"RIFF"):
            if audio_chunk[:4] == b"RIFF":
                audio_chunk = audio_chunk[44:]
            audio_chunk = audioop.lin2ulaw(audio_chunk, 2)
        return audio_chunk

    async def _rate_limit(self, frame_bytes: int) -> None:
        """Sleep if needed after sending frame_bytes to stay ≤ MAX_SEND_RATE_FACTOR × real-time.

        Asterisk's internal chan_websocket frame queue holds ~1186 frames (~23.7 s).
        When the queue is full it silently drops frames without sending a second
        MEDIA_XOFF.  Capping the send rate prevents the queue from ever filling.
        """
        if MAX_SEND_RATE_FACTOR <= 0 or self._response_first_send == 0.0:
            return
        self._bytes_sent += frame_bytes
        min_elapsed = self._bytes_sent / (MAX_SEND_RATE_FACTOR * ULAW_BYTES_PER_SECOND)
        elapsed = time.monotonic() - self._response_first_send
        if elapsed < min_elapsed:
            await asyncio.sleep(min_elapsed - elapsed)

    async def _send_frames(self, chunk: bytes, gen: int) -> str:
        """Send `chunk` to Asterisk split into frames ≤ MAX_WS_FRAME_BYTES, rate-limited.

        Shared by handle() and drain_local_queue(). Returns:
          "sent"       — fully sent
          "queue_full" — MEDIA_XOFF arrived mid-send; the unsent remainder was re-queued
                         at the front of _local_audio_queue (preserves ordering)
          "stale"      — a FLUSH_MEDIA/interruption bumped the generation; nothing more sent
        """
        self._send_in_flight = True
        try:
            offset = 0
            while offset < len(chunk):
                if gen != self._flush_generation:
                    return "stale"
                if self.queue_full:
                    self._local_audio_queue.appendleft((AUDIO_ENTRY, chunk[offset:]))
                    return "queue_full"
                end = min(offset + MAX_WS_FRAME_BYTES, len(chunk))
                if self._response_first_send == 0.0:
                    self._response_first_send = time.monotonic()
                await self.websocket.send_bytes(chunk[offset:end])
                await self._rate_limit(end - offset)
                offset = end
            return "sent"
        finally:
            self._send_in_flight = False

    def has_unsent_audio(self) -> bool:
        """Audio accepted but not yet written to Asterisk (mid-send or XOFF-parked).

        XOFF-parked audio is only counted while the handler is open: close() runs before
        stop_handler() and drain_local_queue() bails once _closed, so after close that
        audio can never be sent — waiting on it would stall teardown for the full cap.
        """
        return self._send_in_flight or (not self._closed and bool(self._local_audio_queue))

    def _register_mark(self, meta_info: dict, is_final: bool, audio_duration: float) -> str:
        """Record per-mark metadata (same contract as Plivo/Twilio) and return the mark id
        so the caller can hand it to Asterisk via MARK_MEDIA."""
        message_category = meta_info.get("message_category", "agent_response")
        mark_id = meta_info.get("mark_id") or str(uuid.uuid4())
        self.mark_event_meta_data.update_data(
            mark_id,
            {
                "text_synthesized": meta_info.get("text_synthesized", "") if meta_info.get("sequence_id") != -1 else "",
                "type": message_category,
                "is_first_chunk": meta_info.get("is_first_chunk", False),
                "is_final_chunk": is_final,
                "sequence_id": meta_info.get("sequence_id", 0),
                "duration": audio_duration,
                "sent_ts": time.time(),
            },
        )
        return mark_id

    async def _emit_mark(self, mark_id: str) -> None:
        """Ask Asterisk to echo this mark once the audio ahead of it has been played out.

        Queued behind parked audio during XOFF: MARK_MEDIA is a TEXT frame and would
        otherwise overtake the binary audio it marks, echoing back early.
        """
        if self.queue_full or self._local_audio_queue:
            self._local_audio_queue.append((MARK_ENTRY, mark_id))
        else:
            await self._send_control("MARK_MEDIA", {"id": mark_id})

    # ------------------------------------------------------------------
    # Interruption
    # ------------------------------------------------------------------

    async def handle_interruption(self):
        """FLUSH_MEDIA, clear local queue, increment generation to drop stale audio."""
        logger.info("sip-trunk: handling interruption (FLUSH_MEDIA)")
        try:
            self._reset_response_counters()
            self._flush_generation += 1
            self._current_sequence_id = None
            self._local_audio_queue.clear()

            await self._send_control("FLUSH_MEDIA")
            # FLUSH_MEDIA tells Asterisk to drop its buffered audio, relieving any XOFF
            # backpressure. Reset queue_full so the next response sends immediately instead
            # of queuing behind stale backpressure state, and re-clear anything that raced
            # into the queue during the FLUSH round-trip (all pre-interruption / stale).
            self.queue_full = False
            self._local_audio_queue.clear()
            # FLUSH_MEDIA resets Asterisk's buffering state; re-arm for next response
            self._start_buffering_sent = False

            if self.mark_event_meta_data:
                self.mark_event_meta_data.clear_data()
            if self.input_handler:
                self.input_handler.update_is_audio_being_played(False)
        except Exception as e:
            logger.error(f"sip-trunk handle_interruption: {e}")

    # ------------------------------------------------------------------
    # XOFF/XON — backpressure safety for very long responses
    # ------------------------------------------------------------------

    async def drain_local_queue(self):
        """Send XOFF-queued audio to Asterisk (called on MEDIA_XON).

        Serialized via _drain_lock: a second MEDIA_XON waits for the in-flight drain
        to finish (then re-scans the queue) rather than running concurrently.
        """
        async with self._drain_lock:
            # Re-anchor the rate-limit budget to exclude the XOFF pause.
            # During XOFF the wall clock kept running while _bytes_sent was frozen;
            # without this, elapsed >> min_elapsed at drain start and _rate_limit
            # returns immediately for every frame — the drain bursts at full TTS speed
            # and immediately re-overflows Asterisk's frame queue.
            if self._response_first_send > 0.0 and MAX_SEND_RATE_FACTOR > 0:
                expected_elapsed = self._bytes_sent / (MAX_SEND_RATE_FACTOR * ULAW_BYTES_PER_SECOND)
                self._response_first_send = time.monotonic() - expected_elapsed
            gen = self._flush_generation
            while self._local_audio_queue and not self.queue_full:
                # Drop stale audio from before an interruption
                if gen != self._flush_generation:
                    self._local_audio_queue.clear()
                    self._pending_finish = False
                    return
                kind, payload = self._local_audio_queue.popleft()
                if not payload:
                    continue
                if self._closed:
                    return
                if kind == MARK_ENTRY:
                    # In-band with the audio around it, so Asterisk echoes it at the right
                    # playout position rather than the moment XOFF lifted.
                    await self._send_control("MARK_MEDIA", {"id": payload})
                    continue
                try:
                    status = await self._send_frames(payload, gen)
                except Exception as e:
                    logger.debug(f"sip-trunk drain send_bytes stopped: {e}")
                    return
                if status == "stale":
                    self._local_audio_queue.clear()
                    self._pending_finish = False
                    return
                if status == "queue_full":
                    # XOFF arrived mid-drain — remainder re-queued; stop until next XON.
                    return

            if not self._local_audio_queue and self._pending_finish:
                self._pending_finish = False
                await self._flush_buffer(gen)
                self._schedule_finish(self._flush_generation)

    # ------------------------------------------------------------------
    # Control helpers
    # ------------------------------------------------------------------

    async def _send_control(self, command, params=None):
        """Send one control command as TEXT (plain text, Asterisk-compatible)."""
        try:
            if self._closed:
                return
            msg = command if not params else f"{command} {' '.join(str(v) for v in params.values())}"
            await self.websocket.send_text(msg)
            logger.debug(f"sip-trunk sent: {command}")
        except Exception as e:
            logger.error(f"sip-trunk send_control {command}: {e}")
            traceback.print_exc()

    async def flush_media(self):
        await self._send_control("FLUSH_MEDIA")

    async def _flush_buffer(self, generation: int):
        """Flush Asterisk's media buffer to RTP at the end of a response and re-arm START
        for the next one. No id (avoids the asterisk#1384 hangup crash); skipped on a stale
        generation so an interrupted response isn't replayed after FLUSH_MEDIA."""
        if generation == self._flush_generation and self._start_buffering_sent:
            await self._send_control("STOP_MEDIA_BUFFERING")
            self._start_buffering_sent = False

    # sip-trunk sends audio as raw binary frames and marks as MARK_MEDIA TEXT frames,
    # so the JSON media/mark framing used by Twilio/Plivo is intentionally unused here.
    async def form_media_message(self, audio_data, audio_format="wav"):
        return None

    async def form_mark_message(self, mark_id):
        return None

    async def set_stream_sid(self, stream_id):
        self.stream_sid = stream_id

    # ------------------------------------------------------------------
    # Main handle — send audio directly to Asterisk (burst, like Plivo)
    # ------------------------------------------------------------------

    async def handle(self, ws_data_packet):
        """Send audio to Asterisk in bursts; track playback completion by accumulated
        duration. Pipeline: convert → reset-on-new-sequence → buffer-gate → register
        mark → send (or XOFF-queue) → emit mark → schedule finish on the final chunk.
        """
        if self._closed:
            return
        try:
            audio_chunk = ws_data_packet.get("data")
            meta_info = ws_data_packet.get("meta_info") or {}
            if self.stream_sid is None:
                self.stream_sid = meta_info.get("stream_sid")

            is_final = bool(
                (meta_info.get("end_of_llm_stream") and meta_info.get("end_of_synthesizer_stream"))
                or meta_info.get("is_final_chunk_of_entire_response")
                or (meta_info.get("sequence_id") == -1 and meta_info.get("end_of_llm_stream"))
            )
            has_audio = audio_chunk and len(audio_chunk) > 1 and audio_chunk != b"\x00\x00"

            if not has_audio and not is_final:
                return

            audio_format = (meta_info.get("format") or "ulaw").lower()
            audio_duration = 0.0
            gen = self._flush_generation

            if has_audio:
                if len(audio_chunk) == 1:
                    audio_chunk += b"\x00"
                audio_chunk = self._to_ulaw(audio_chunk, audio_format)

                if meta_info.get("message_category") == "agent_welcome_message" and not self.welcome_message_sent_ts:
                    self.welcome_message_sent_ts = time.time() * 1000

                # New response — reset duration tracking only when sequence_id actually
                # changes.  ElevenLabs may spuriously set is_first_chunk on every chunk
                # after the LLM stream ends, which would repeatedly reset the audio-duration
                # accumulator and make the server think playback finishes far too early.
                seq_id = meta_info.get("sequence_id")
                if seq_id is not None and seq_id != self._current_sequence_id:
                    self._current_sequence_id = seq_id
                    self._reset_response_counters()

                # Send START_MEDIA_BUFFERING once per response (re-armed after each STOP / FLUSH_MEDIA)
                if not self._start_buffering_sent:
                    await self._send_control("START_MEDIA_BUFFERING")
                    self._start_buffering_sent = True
                    logger.info("sip-trunk: START_MEDIA_BUFFERING sent")

                # Track audio duration
                audio_duration = len(audio_chunk) / ULAW_BYTES_PER_SECOND
                self._response_audio_duration += audio_duration

                # Drop stale audio from a previous generation
                if gen != self._flush_generation:
                    return

            # Register the mark BEFORE sending: the send is rate-limited, so a large
            # chunk stays in flight for seconds, and the hangup gate reads "no pending
            # marks" as "everything heard" — registering after the send lets HANGUP
            # cut off a chunk that is still being written.
            mark_id = self._register_mark(meta_info, is_final, audio_duration) if self.mark_event_meta_data else None

            if has_audio:
                # If XOFF is active or a drain is in flight, queue locally to preserve
                # ordering; otherwise send now (re-queues the remainder on a mid-send XOFF).
                if self.queue_full or self._local_audio_queue:
                    self._local_audio_queue.append((AUDIO_ENTRY, audio_chunk))
                else:
                    if self._closed:
                        return
                    try:
                        if await self._send_frames(audio_chunk, gen) == "stale":
                            return
                    except Exception as e:
                        logger.debug(f"sip-trunk send_bytes stopped: {e}")
                        return

            # Ask Asterisk to echo the mark back once this chunk reaches the front of
            # its playout queue (same contract as Plivo/Twilio).
            if mark_id is not None:
                if gen == self._flush_generation:
                    await self._emit_mark(mark_id)

                if is_final:
                    if self._local_audio_queue:
                        # XOFF queued audio not yet sent — defer finish
                        self._pending_finish = True
                        logger.debug("sip-trunk: final chunk, XOFF audio pending; deferring finish")
                    else:
                        self._pending_finish = False
                        await self._flush_buffer(gen)
                        self._schedule_finish(gen)

        except Exception as e:
            logger.error(f"sip-trunk output error: {e}")
            traceback.print_exc()

    async def send_hangup(self):
        await self._send_control("HANGUP")

    def set_hangup_sent(self):
        super().set_hangup_sent()
        try:
            asyncio.create_task(self.send_hangup())
        except Exception as e:
            logger.error(f"sip-trunk send_hangup: {e}")

    def requires_custom_voicemail_detection(self):
        # SIP/BYOT (Asterisk) has no carrier-side AMD, so voicemail must be caught via the LLM path.
        return True
