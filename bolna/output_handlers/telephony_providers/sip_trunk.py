"""
Asterisk WebSocket (chan_websocket) output handler for sip-trunk provider.

Implements flow-controlled audio streaming with remote buffer tracking,
generation-based interruption cleanup, and QUEUE_DRAINED playback detection.

Design informed by Pipecat's Asterisk transport (pipecat-ai/pipecat#3229):
- Local jitter buffer to smooth TTS startup latency
- Remote buffer estimation with real-time drain tracking
- Generation counter to drop stale frames after FLUSH_MEDIA
- XOFF/XON flow control for Asterisk buffer backpressure
- QUEUE_DRAINED as primary playback completion signal

Ref: https://docs.asterisk.org/Configuration/Channel-Drivers/WebSocket/
"""
import asyncio
import audioop
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from bolna.enums import AsteriskCommand, AudioFormat
from bolna.output_handlers.telephony import TelephonyOutputHandler
from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

# Asterisk ulaw: 160 bytes per 20ms frame at 8 kHz mono
ASTERISK_ULAW_OPTIMAL_FRAME_SIZE = 160
ASTERISK_ULAW_SAMPLE_RATE = 8000
DEFAULT_PTIME_MS = 20

# Asterisk chan_websocket hard-codes a 1000-frame media buffer.
ASTERISK_MAX_BUFFER_FRAMES = 1000

# Safety margin: fill at most half the remote buffer to avoid XOFF cycling.
DEFAULT_MAX_REMOTE_BUFFER_FRAMES = 500

# Resume sending when the remote buffer drops below this fraction of max.
REMOTE_BUFFER_RESUME_THRESHOLD = 0.5

# Accumulate this much audio locally before the first send to the remote
# buffer.  Prevents glitches when TTS is slow to produce the first chunk.
# Skipped for subsequent turns (TTS connection is already warm).
INITIAL_JITTER_BUFFER_MS = int(os.environ.get("SIP_JITTER_BUFFER_MS", "40"))

# Fallback timer added on top of the calculated audio duration.  Fires only
# when QUEUE_DRAINED never arrives — QUEUE_DRAINED is the primary signal.
PLAYBACK_DONE_FALLBACK_BUFFER_S = float(
    os.environ.get("SIP_FALLBACK_BUFFER_S", "0.1")
)

# Number of silence frames injected before FLUSH_MEDIA on interruption to
# produce a smooth fade-out instead of an abrupt cut.
INTERRUPTION_FADE_FRAMES = int(os.environ.get("SIP_FADE_FRAMES", "2"))

# ulaw zero-amplitude byte (used for silence / fade-out frames).
ULAW_SILENCE_BYTE = 0xFF


# ── Remote buffer state tracker ──────────────────────────────────────────────

@dataclass
class RemoteBufferState:
    """Estimates the fill level of Asterisk's media buffer.

    Asterisk plays audio at real-time rate.  We track how many bytes we sent
    and subtract an estimate of what Asterisk consumed since the last check
    (``drain_elapsed``).  This avoids relying solely on XOFF/XON events which
    arrive asynchronously.
    """

    bytes_sent: int = 0
    max_bytes: int = 0
    resume_threshold_bytes: int = 0
    optimal_frame_size: int = ASTERISK_ULAW_OPTIMAL_FRAME_SIZE
    ptime_s: float = DEFAULT_PTIME_MS / 1000.0
    last_check_time: float = field(default_factory=time.monotonic)
    is_full: bool = False

    def configure(
        self,
        optimal_frame_size: int,
        ptime_ms: int,
        max_frames: int,
    ) -> None:
        self.optimal_frame_size = optimal_frame_size
        self.ptime_s = ptime_ms / 1000.0
        self.max_bytes = max_frames * optimal_frame_size
        self.resume_threshold_bytes = int(
            self.max_bytes * REMOTE_BUFFER_RESUME_THRESHOLD
        )

    def record_sent(self, num_bytes: int) -> None:
        self.bytes_sent += num_bytes
        self.is_full = self.max_bytes > 0 and self.bytes_sent >= self.max_bytes

    def drain_elapsed(self) -> None:
        """Subtract bytes Asterisk consumed since the last check."""
        now = time.monotonic()
        elapsed = now - self.last_check_time
        self.last_check_time = now
        bytes_consumed = round((elapsed / self.ptime_s) * self.optimal_frame_size)
        self.bytes_sent = max(0, self.bytes_sent - bytes_consumed)
        self.is_full = self.max_bytes > 0 and self.bytes_sent >= self.max_bytes

    def should_resume(self) -> bool:
        return self.bytes_sent < self.resume_threshold_bytes

    def reset(self) -> None:
        """Reset after FLUSH_MEDIA — Asterisk's buffer is now empty."""
        self.bytes_sent = 0
        self.is_full = False
        self.last_check_time = time.monotonic()


# ── Output handler ───────────────────────────────────────────────────────────

class SipTrunkOutputHandler(TelephonyOutputHandler):
    """Asterisk WebSocket output handler.

    Key mechanisms:
    * **Remote buffer tracking** – prevents XOFF/XON cycling by estimating
      how full Asterisk's 1000-frame buffer is and pausing locally.
    * **Generation counter** – incremented on every interruption so that
      stale frames queued before the FLUSH are silently dropped.
    * **QUEUE_DRAINED** – the primary signal for playback completion.
      A duration-based fallback fires only if the event never arrives.
    * **Initial jitter buffer** – smooths TTS startup; skipped after the
      first response because the TTS connection is already warm.
    """

    def __init__(
        self,
        io_provider: str = "sip-trunk",
        websocket=None,
        mark_event_meta_data=None,
        log_dir_name: Optional[str] = None,
        asterisk_media_start: Optional[dict] = None,
        agent_config: Optional[dict] = None,
        input_handler=None,
    ):
        super().__init__(io_provider, websocket, mark_event_meta_data, log_dir_name)
        self.asterisk_media_start = asterisk_media_start or {}
        self.agent_config = agent_config or {}
        self.input_handler = input_handler
        if input_handler:
            input_handler.output_handler_ref = self

        # ── Flow control ────────────────────────────────────────────────
        self.queue_full: bool = False               # Set by input handler on XOFF
        self._remote_buffer = RemoteBufferState()
        self._can_send = asyncio.Event()
        self._can_send.set()
        self._buffer_monitor_task: Optional[asyncio.Task] = None

        # ── Buffering state ─────────────────────────────────────────────
        self._buffering_active: bool = False
        self._response_audio_duration: float = 0.0

        # ── Generation counter (incremented on interruption) ────────────
        self._flush_generation: int = 0

        # ── Playback completion tracking ────────────────────────────────
        self._playback_done_task: Optional[asyncio.Task] = None
        self._awaiting_queue_drained: bool = False

        # ── Local XOFF queue ────────────────────────────────────────────
        self._local_audio_queue: asyncio.Queue = asyncio.Queue()
        self._pending_stop_after_drain: bool = False
        self._pending_stop_duration: float = 0.0
        self._pending_stop_category: str = "agent_response"

        # ── Jitter buffer (first response only) ────────────────────────
        self._jitter_buffer_bytes: int = 0
        self._jitter_buffer_filled: bool = False
        self._jitter_buffer_target: int = 0
        self._first_response_sent: bool = False

        # ── Audio format ────────────────────────────────────────────────
        self._output_format = AudioFormat.ULAW.value

        # Configure from MEDIA_START if available
        self._apply_media_start(self.asterisk_media_start)

    # ── helpers ──────────────────────────────────────────────────────────

    def _apply_media_start(self, media_start: dict) -> None:
        """Extract optimal_frame_size and ptime from MEDIA_START data."""
        opt = media_start.get("optimal_frame_size")
        if opt is None:
            return
        try:
            frame_size = int(opt)
            ptime = int(media_start.get("ptime", DEFAULT_PTIME_MS))
            self._remote_buffer.configure(
                frame_size, ptime, DEFAULT_MAX_REMOTE_BUFFER_FRAMES
            )
            self._jitter_buffer_target = int(
                (INITIAL_JITTER_BUFFER_MS / ptime) * frame_size
            )
        except (TypeError, ValueError):
            pass

    def _is_ws_open(self) -> bool:
        """Return True if the underlying WebSocket is still connected."""
        if self.websocket is None:
            return False
        if hasattr(self.websocket, "client_state"):
            return self.websocket.client_state.value == 1
        return True

    # ── Asterisk control commands ────────────────────────────────────────

    async def _send_command(self, command: str) -> None:
        """Send a TEXT control command to Asterisk."""
        if not self._is_ws_open():
            return
        try:
            await self.websocket.send_text(command)
            logger.debug("sip-trunk sent: %s", command)
        except Exception as e:
            # Suppress "after websocket.close" — expected during shutdown
            if "websocket.close" not in str(e):
                logger.error("sip-trunk _send_command %s: %s", command, e)

    async def _send_binary(self, data: bytes) -> None:
        """Send binary audio in optimal-frame-size-aligned chunks."""
        if not self._is_ws_open():
            return
        frame_size = self._remote_buffer.optimal_frame_size
        for offset in range(0, len(data), frame_size):
            chunk = data[offset : offset + frame_size]
            if chunk:
                try:
                    await self.websocket.send_bytes(chunk)
                except Exception:
                    return
                self._remote_buffer.record_sent(len(chunk))

    # ── Buffer monitoring ────────────────────────────────────────────────

    async def _start_buffer_monitor(self) -> None:
        if self._buffer_monitor_task and not self._buffer_monitor_task.done():
            return
        self._buffer_monitor_task = asyncio.create_task(
            self._monitor_remote_buffer()
        )

    async def _monitor_remote_buffer(self) -> None:
        """Background: periodically drain estimated remote buffer and
        manage the ``_can_send`` gate."""
        try:
            while True:
                await asyncio.sleep(self._remote_buffer.ptime_s)
                self._remote_buffer.drain_elapsed()

                if self._can_send.is_set():
                    if self._remote_buffer.is_full:
                        self._can_send.clear()
                        logger.debug("sip-trunk: remote buffer full, pausing")
                elif self._remote_buffer.should_resume():
                    self._can_send.set()
                    logger.debug("sip-trunk: remote buffer drained, resuming")
        except asyncio.CancelledError:
            return

    # ── Flow-controlled send ─────────────────────────────────────────────

    async def _send_audio_flow_controlled(
        self, audio_chunk: bytes, generation: int
    ) -> None:
        """Send audio respecting remote buffer limits.  Drops stale frames."""
        if generation != self._flush_generation:
            return

        if self.queue_full:
            await self._local_audio_queue.put((audio_chunk, generation))
            return

        # Refresh estimate before checking — avoids stale "full" state
        if not self._can_send.is_set():
            self._remote_buffer.drain_elapsed()
            if self._remote_buffer.should_resume():
                self._can_send.set()

        if not self._can_send.is_set():
            try:
                await asyncio.wait_for(self._can_send.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("sip-trunk: timeout waiting for remote buffer space")
                return

        if generation != self._flush_generation:
            return

        await self._send_binary(audio_chunk)

    async def drain_local_queue(self) -> None:
        """Flush audio queued during MEDIA_XOFF (called on MEDIA_XON)."""
        while not self._local_audio_queue.empty() and not self.queue_full:
            try:
                chunk, generation = self._local_audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if generation != self._flush_generation or not chunk:
                continue
            await self._send_binary(chunk)

        if self._local_audio_queue.empty() and self._pending_stop_after_drain:
            self._pending_stop_after_drain = False
            await self._finish_response(
                self._pending_stop_duration, self._pending_stop_category
            )

    # ── Response lifecycle ───────────────────────────────────────────────

    async def _start_buffering(self) -> None:
        """Begin a new response: send START_MEDIA_BUFFERING and reset state."""
        await self._send_command(AsteriskCommand.START_MEDIA_BUFFERING.value)
        self._buffering_active = True
        self._response_audio_duration = 0.0
        self._jitter_buffer_bytes = 0
        self._jitter_buffer_filled = self._jitter_buffer_target <= 0
        self._awaiting_queue_drained = False
        await self._start_buffer_monitor()

    async def _finish_response(
        self, total_duration: float, message_category: str
    ) -> None:
        """End a response: STOP buffering, request QUEUE_DRAINED, schedule
        a duration-based fallback in case the event never arrives."""
        if self._buffering_active:
            await self._send_command(AsteriskCommand.STOP_MEDIA_BUFFERING.value)
            self._buffering_active = False

        self._first_response_sent = True

        await self._send_command(AsteriskCommand.REPORT_QUEUE_DRAINED.value)
        self._awaiting_queue_drained = True

        if self._playback_done_task:
            self._playback_done_task.cancel()
        self._playback_done_task = asyncio.create_task(
            self._playback_done_fallback(total_duration, message_category)
        )
        logger.debug(
            "sip-trunk: response done, awaiting QUEUE_DRAINED "
            "(fallback in %.1fs)",
            total_duration + PLAYBACK_DONE_FALLBACK_BUFFER_S,
        )

    def handle_queue_drained(self) -> None:
        """Called by the input handler when QUEUE_DRAINED arrives.

        This is the authoritative signal that Asterisk finished playing all
        buffered audio.
        """
        if not self._awaiting_queue_drained:
            return
        self._awaiting_queue_drained = False

        if self._playback_done_task:
            self._playback_done_task.cancel()
            self._playback_done_task = None

        self._process_pending_marks("agent_response")
        logger.info("sip-trunk: QUEUE_DRAINED received, playback complete")

    def _process_pending_marks(self, default_category: str) -> None:
        """Simulate mark-echo for Twilio/Plivo parity."""
        if not self.input_handler or not self.input_handler.mark_event_meta_data:
            return
        remaining = list(
            self.input_handler.mark_event_meta_data.mark_event_meta_data.keys()
        )
        if not remaining:
            return
        logger.info("sip-trunk: processing %d pending mark(s)", len(remaining))
        self.input_handler.update_is_audio_being_played(False)
        for mid in remaining:
            md = self.input_handler.mark_event_meta_data.mark_event_meta_data.get(
                mid, {}
            )
            self.input_handler.process_mark_message(
                {"name": mid, "type": md.get("type", default_category)}
            )

    async def _playback_done_fallback(
        self, duration: float, message_category: str
    ) -> None:
        """Safety net: process marks after estimated playback time if
        QUEUE_DRAINED never arrives."""
        try:
            await asyncio.sleep(duration + PLAYBACK_DONE_FALLBACK_BUFFER_S)
        except asyncio.CancelledError:
            return
        self._playback_done_task = None
        self._awaiting_queue_drained = False
        logger.info("sip-trunk: playback-done fallback fired (%.2fs)", duration)
        self._process_pending_marks(message_category)

    # ── Interruption ─────────────────────────────────────────────────────

    async def handle_interruption(self) -> None:
        """Smooth fade-out → FLUSH_MEDIA → reset all state.

        The generation counter is incremented *first* so that any in-flight
        frames (already in ``_send_audio_flow_controlled`` or the local XOFF
        queue) are silently dropped.  The fade-out sends a few silence frames
        before the flush so the caller hears a brief taper rather than an
        abrupt cut.
        """
        logger.info("sip-trunk: handling interruption (fade-out + FLUSH_MEDIA)")
        try:
            self._flush_generation += 1

            if self._playback_done_task:
                self._playback_done_task.cancel()
                self._playback_done_task = None

            # Smooth fade-out before flushing
            if self._buffering_active and INTERRUPTION_FADE_FRAMES > 0:
                silence = (
                    bytes([ULAW_SILENCE_BYTE])
                    * self._remote_buffer.optimal_frame_size
                )
                for _ in range(INTERRUPTION_FADE_FRAMES):
                    if not self._is_ws_open():
                        break
                    try:
                        await self.websocket.send_bytes(silence)
                    except Exception:
                        break

            self._buffering_active = False
            self._response_audio_duration = 0.0
            self._awaiting_queue_drained = False
            self._pending_stop_after_drain = False
            self._jitter_buffer_bytes = 0
            self._jitter_buffer_filled = False

            # Drain local XOFF queue (all frames are now stale)
            while not self._local_audio_queue.empty():
                try:
                    self._local_audio_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

            # Flush Asterisk's remote buffer
            await self._send_command(AsteriskCommand.FLUSH_MEDIA.value)
            self._remote_buffer.reset()
            self._can_send.set()

            if self.mark_event_meta_data:
                self.mark_event_meta_data.clear_data()
            if self.input_handler:
                self.input_handler.update_is_audio_being_played(False)
        except Exception as e:
            logger.error("sip-trunk handle_interruption: %s", e)

    # ── Main audio handling ──────────────────────────────────────────────

    async def handle(self, ws_data_packet: dict) -> None:
        """Process an audio packet from the synthesizer pipeline.

        Flow: validate → convert format → start buffering → jitter buffer
        → send flow-controlled → update mark metadata → finish response.
        """
        try:
            audio_chunk = ws_data_packet.get("data")
            meta_info = ws_data_packet.get("meta_info") or {}
            if self.stream_sid is None:
                self.stream_sid = meta_info.get("stream_sid")

            generation = self._flush_generation

            is_final = bool(
                (
                    meta_info.get("end_of_llm_stream")
                    and meta_info.get("end_of_synthesizer_stream")
                )
                or meta_info.get("is_final_chunk_of_entire_response")
                or (
                    meta_info.get("sequence_id") == -1
                    and meta_info.get("end_of_llm_stream")
                )
            )
            has_audio = (
                audio_chunk
                and len(audio_chunk) > 1
                and audio_chunk != b"\x00\x00"
            )

            if not has_audio and not is_final:
                return

            audio_format = (
                meta_info.get("format") or AudioFormat.ULAW.value
            ).lower()
            audio_duration = 0.0

            if has_audio:
                # Pad odd-length PCM to even for audioop
                if len(audio_chunk) == 1:
                    audio_chunk += b"\x00"

                # Convert PCM/WAV → ulaw
                if audio_format in (
                    AudioFormat.PCM.value,
                    AudioFormat.WAV.value,
                ) or (len(audio_chunk) > 44 and audio_chunk[:4] == b"RIFF"):
                    if audio_chunk[:4] == b"RIFF":
                        audio_chunk = audio_chunk[44:]
                    audio_chunk = audioop.lin2ulaw(audio_chunk, 2)

                if (
                    meta_info.get("message_category") == "agent_welcome_message"
                    and not self.welcome_message_sent_ts
                ):
                    self.welcome_message_sent_ts = time.time() * 1000

                # Start buffering on first audio of a new response
                if (
                    not self._buffering_active
                    and len(audio_chunk) > self._remote_buffer.optimal_frame_size
                ):
                    await self._start_buffering()

                # Skip jitter buffer for subsequent turns (TTS is warm)
                if self._first_response_sent and not self._jitter_buffer_filled:
                    self._jitter_buffer_filled = True
                    logger.debug("sip-trunk: skipping jitter buffer (warm TTS)")

                if not self._jitter_buffer_filled:
                    self._jitter_buffer_bytes += len(audio_chunk)
                    await self._local_audio_queue.put((audio_chunk, generation))
                    if self._jitter_buffer_bytes >= self._jitter_buffer_target:
                        self._jitter_buffer_filled = True
                        logger.debug("sip-trunk: jitter buffer filled, sending")
                        while (
                            not self._local_audio_queue.empty()
                            and not self.queue_full
                        ):
                            try:
                                chunk, gen = self._local_audio_queue.get_nowait()
                            except asyncio.QueueEmpty:
                                break
                            if gen == self._flush_generation:
                                await self._send_binary(chunk)
                else:
                    await self._send_audio_flow_controlled(
                        audio_chunk, generation
                    )

                audio_duration = len(audio_chunk) / ASTERISK_ULAW_SAMPLE_RATE
                self._response_audio_duration += audio_duration

            # Update mark metadata for latency tracking
            if self.mark_event_meta_data:
                message_category = meta_info.get(
                    "message_category", "agent_response"
                )
                mark_id = meta_info.get("mark_id") or str(uuid.uuid4())
                self.mark_event_meta_data.update_data(
                    mark_id,
                    {
                        "text_synthesized": (
                            meta_info.get("text_synthesized", "")
                            if meta_info.get("sequence_id") != -1
                            else ""
                        ),
                        "type": message_category,
                        "is_first_chunk": meta_info.get("is_first_chunk", False),
                        "is_final_chunk": is_final,
                        "sequence_id": meta_info.get("sequence_id", 0),
                        "duration": audio_duration,
                        "sent_ts": time.time(),
                    },
                )

                if is_final:
                    total_duration = self._response_audio_duration
                    # DO NOT reset _response_audio_duration here.
                    # is_final fires twice per response (last audio chunk +
                    # null-byte sentinel).  Resetting here would cause the
                    # second fire to schedule a 0-duration fallback.
                    # Reset happens in _start_buffering() for the next
                    # response.

                    if not self._local_audio_queue.empty():
                        self._pending_stop_after_drain = True
                        self._pending_stop_duration = total_duration
                        self._pending_stop_category = message_category
                        logger.debug(
                            "sip-trunk: final chunk queued; finish after drain"
                        )
                    elif total_duration > 0 or not self._playback_done_task:
                        await self._finish_response(
                            total_duration, message_category
                        )

        except Exception as e:
            logger.error("sip-trunk output error: %s", e, exc_info=True)

    # ── Base-class stubs ─────────────────────────────────────────────────

    async def form_media_message(self, audio_data, audio_format="wav"):
        return None

    async def form_mark_message(self, mark_id):
        return None

    async def set_stream_sid(self, stream_id):
        self.stream_sid = stream_id

    async def flush_media(self):
        await self._send_command(AsteriskCommand.FLUSH_MEDIA.value)

    async def send_hangup(self):
        await self._send_command(AsteriskCommand.HANGUP.value)

    def set_hangup_sent(self):
        super().set_hangup_sent()
        try:
            asyncio.create_task(self.send_hangup())
        except Exception as e:
            logger.error("sip-trunk send_hangup: %s", e)

    def requires_custom_voicemail_detection(self):
        return False
