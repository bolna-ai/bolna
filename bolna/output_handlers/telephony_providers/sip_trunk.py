"""
Asterisk WebSocket (chan_websocket) output handler for sip-trunk provider.

Implements flow-controlled audio streaming with remote buffer tracking,
generation-based interruption cleanup, and QUEUE_DRAINED playback detection.

Ref: https://docs.asterisk.org/Configuration/Channel-Drivers/WebSocket/
"""
import asyncio
import audioop
import time
import traceback
import uuid
from dataclasses import dataclass, field

from bolna.enums import AsteriskCommand, AudioFormat
from bolna.output_handlers.telephony import TelephonyOutputHandler
from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)

# Asterisk ulaw: 160 bytes per 20ms frame at 8kHz
ASTERISK_ULAW_OPTIMAL_FRAME_SIZE = 160
ASTERISK_ULAW_SAMPLE_RATE = 8000
# Default ptime in ms
DEFAULT_PTIME_MS = 20
# Max frames Asterisk chan_websocket buffers (hardcoded in Asterisk source)
ASTERISK_MAX_BUFFER_FRAMES = 1000
# Safety margin: only fill half the remote buffer to avoid XOFF
DEFAULT_MAX_REMOTE_BUFFER_FRAMES = 500
# Resume sending when remote buffer drops below this fraction
REMOTE_BUFFER_RESUME_THRESHOLD = 0.5
# Accumulate this much audio locally before first send (prevents glitches if TTS is slow)
INITIAL_JITTER_BUFFER_MS = 80
# Fallback timeout multiplier if QUEUE_DRAINED never arrives
PLAYBACK_DONE_FALLBACK_BUFFER_S = 0.5


@dataclass
class RemoteBufferState:
    """Tracks estimated fill level of Asterisk's media buffer."""
    bytes_sent: int = 0
    max_bytes: int = 0
    resume_threshold_bytes: int = 0
    optimal_frame_size: int = ASTERISK_ULAW_OPTIMAL_FRAME_SIZE
    ptime_s: float = DEFAULT_PTIME_MS / 1000.0
    last_check_time: float = field(default_factory=time.monotonic)
    is_full: bool = False

    def configure(self, optimal_frame_size: int, ptime_ms: int, max_frames: int):
        self.optimal_frame_size = optimal_frame_size
        self.ptime_s = ptime_ms / 1000.0
        self.max_bytes = max_frames * optimal_frame_size
        self.resume_threshold_bytes = int(self.max_bytes * REMOTE_BUFFER_RESUME_THRESHOLD)

    def record_sent(self, num_bytes: int):
        self.bytes_sent += num_bytes
        self.is_full = self.max_bytes > 0 and self.bytes_sent >= self.max_bytes

    def drain_elapsed(self):
        """Subtract bytes Asterisk consumed since last check (plays at real-time rate)."""
        now = time.monotonic()
        elapsed = now - self.last_check_time
        self.last_check_time = now
        bytes_consumed = round((elapsed / self.ptime_s) * self.optimal_frame_size)
        self.bytes_sent = max(0, self.bytes_sent - bytes_consumed)
        self.is_full = self.max_bytes > 0 and self.bytes_sent >= self.max_bytes

    def should_resume(self) -> bool:
        return self.bytes_sent < self.resume_threshold_bytes

    def reset(self):
        self.bytes_sent = 0
        self.is_full = False
        self.last_check_time = time.monotonic()


class SipTrunkOutputHandler(TelephonyOutputHandler):
    """
    Asterisk WebSocket output handler with:
    - Remote buffer tracking to prevent XOFF/XON cycling
    - Generation counter for clean interruptions (drops stale frames)
    - QUEUE_DRAINED as primary playback completion signal
    - Duration-based fallback timer as safety net
    - Initial jitter buffer to prevent glitches at start of speech
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
        if input_handler:
            input_handler.output_handler_ref = self

        # Flow control
        self.queue_full = False
        self._remote_buffer = RemoteBufferState()
        self._can_send = asyncio.Event()
        self._can_send.set()
        self._buffer_monitor_task = None

        # Buffering state
        self._buffering_active = False
        self._response_audio_duration = 0.0

        # Generation counter: incremented on interruption, stale frames are dropped
        self._flush_generation = 0

        # Playback completion tracking
        self._playback_done_task = None
        self._awaiting_queue_drained = False

        # Local queue for audio during XOFF
        self._local_audio_queue = asyncio.Queue()
        self._pending_stop_after_drain = False
        self._pending_stop_duration = 0.0
        self._pending_stop_category = "agent_response"
        self._drain_task = None

        # Initial jitter buffer
        self._jitter_buffer_bytes = 0
        self._jitter_buffer_filled = False
        self._jitter_buffer_target = 0

        # Audio format
        self._output_format = AudioFormat.ULAW.value

        # Configure from MEDIA_START if available
        opt = self.asterisk_media_start.get("optimal_frame_size")
        if opt is not None:
            try:
                frame_size = int(opt)
                ptime = int(self.asterisk_media_start.get("ptime", DEFAULT_PTIME_MS))
                self._remote_buffer.configure(frame_size, ptime, DEFAULT_MAX_REMOTE_BUFFER_FRAMES)
                self._jitter_buffer_target = int(
                    (INITIAL_JITTER_BUFFER_MS / ptime) * frame_size
                )
            except (TypeError, ValueError):
                pass

    # -- Asterisk control commands --

    async def _send_command(self, command: str):
        """Send a TEXT control command to Asterisk."""
        try:
            await self.websocket.send_text(command)
            logger.debug(f"sip-trunk sent: {command}")
        except Exception as e:
            logger.error(f"sip-trunk _send_command {command}: {e}")
            traceback.print_exc()

    async def _send_binary(self, data: bytes):
        """Send binary audio to Asterisk in optimal_frame_size-aligned chunks."""
        frame_size = self._remote_buffer.optimal_frame_size
        offset = 0
        n = len(data)
        while offset < n:
            chunk_end = min(offset + frame_size, n)
            chunk = data[offset:chunk_end]
            if chunk:
                await self.websocket.send_bytes(chunk)
                self._remote_buffer.record_sent(len(chunk))
            offset = chunk_end

    # -- Buffer monitoring --

    async def _start_buffer_monitor(self):
        """Background task that tracks remote buffer drain rate."""
        if self._buffer_monitor_task and not self._buffer_monitor_task.done():
            return
        self._buffer_monitor_task = asyncio.create_task(self._monitor_remote_buffer())

    async def _monitor_remote_buffer(self):
        """Periodically drain estimated remote buffer and manage send gate."""
        try:
            while True:
                await asyncio.sleep(self._remote_buffer.ptime_s)
                self._remote_buffer.drain_elapsed()

                if self._can_send.is_set():
                    if self._remote_buffer.is_full:
                        self._can_send.clear()
                        logger.debug("sip-trunk: remote buffer full, pausing send")
                elif self._remote_buffer.should_resume():
                    self._can_send.set()
                    logger.debug("sip-trunk: remote buffer drained, resuming send")
        except asyncio.CancelledError:
            return

    # -- Flow-controlled send --

    async def _send_audio_flow_controlled(self, audio_chunk: bytes, generation: int):
        """Send audio respecting remote buffer limits. Drops stale frames."""
        if generation != self._flush_generation:
            return  # Stale frame from before interruption

        if self.queue_full:
            await self._local_audio_queue.put((audio_chunk, generation))
            return

        # Wait for remote buffer space
        if not self._can_send.is_set():
            try:
                await asyncio.wait_for(self._can_send.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("sip-trunk: timeout waiting for remote buffer space")
                return

        if generation != self._flush_generation:
            return  # Check again after wait

        await self._send_binary(audio_chunk)

    async def drain_local_queue(self):
        """Send audio queued during MEDIA_XOFF (called on MEDIA_XON)."""
        while not self._local_audio_queue.empty() and not self.queue_full:
            try:
                chunk, generation = self._local_audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if generation != self._flush_generation:
                continue  # Drop stale frames
            if not chunk:
                continue
            await self._send_binary(chunk)

        if self._local_audio_queue.empty() and self._pending_stop_after_drain:
            self._pending_stop_after_drain = False
            await self._finish_response(
                self._pending_stop_duration, self._pending_stop_category
            )

    # -- Response lifecycle --

    async def _start_buffering(self):
        """Begin a new response: START_MEDIA_BUFFERING and reset state."""
        await self._send_command(AsteriskCommand.START_MEDIA_BUFFERING.value)
        self._buffering_active = True
        self._response_audio_duration = 0.0
        self._jitter_buffer_bytes = 0
        self._jitter_buffer_filled = self._jitter_buffer_target <= 0
        self._awaiting_queue_drained = False
        await self._start_buffer_monitor()

    async def _finish_response(self, total_duration: float, message_category: str):
        """End a response: STOP buffering, request QUEUE_DRAINED, schedule fallback."""
        if self._buffering_active:
            await self._send_command(AsteriskCommand.STOP_MEDIA_BUFFERING.value)
            self._buffering_active = False

        await self._send_command(AsteriskCommand.REPORT_QUEUE_DRAINED.value)
        self._awaiting_queue_drained = True

        # Schedule duration-based fallback in case QUEUE_DRAINED never arrives
        if self._playback_done_task:
            self._playback_done_task.cancel()
        self._playback_done_task = asyncio.create_task(
            self._playback_done_fallback(total_duration, message_category)
        )
        logger.debug(
            f"sip-trunk: response done, awaiting QUEUE_DRAINED "
            f"(fallback in {total_duration + PLAYBACK_DONE_FALLBACK_BUFFER_S:.1f}s)"
        )

    def handle_queue_drained(self):
        """Called by input handler when QUEUE_DRAINED arrives from Asterisk.
        This means Asterisk has finished playing all buffered audio."""
        if not self._awaiting_queue_drained:
            return
        self._awaiting_queue_drained = False

        # Cancel fallback timer — QUEUE_DRAINED is the authoritative signal
        if self._playback_done_task:
            self._playback_done_task.cancel()
            self._playback_done_task = None

        self._process_pending_marks("agent_response")
        logger.info("sip-trunk: QUEUE_DRAINED received, playback complete")

    def _process_pending_marks(self, default_category: str):
        """Process all pending mark events (simulates mark echo for Twilio/Plivo parity)."""
        if not self.input_handler or not self.input_handler.mark_event_meta_data:
            return
        remaining = list(self.input_handler.mark_event_meta_data.mark_event_meta_data.keys())
        if not remaining:
            return
        logger.info(f"sip-trunk: processing {len(remaining)} pending mark(s)")
        self.input_handler.update_is_audio_being_played(False)
        for mid in remaining:
            md = self.input_handler.mark_event_meta_data.mark_event_meta_data.get(mid, {})
            self.input_handler.process_mark_message(
                {"name": mid, "type": md.get("type", default_category)}
            )

    async def _playback_done_fallback(self, duration: float, message_category: str):
        """Safety net: if QUEUE_DRAINED never arrives, process marks after estimated playback time."""
        try:
            await asyncio.sleep(duration + PLAYBACK_DONE_FALLBACK_BUFFER_S)
        except asyncio.CancelledError:
            return
        self._playback_done_task = None
        self._awaiting_queue_drained = False
        logger.info(f"sip-trunk: playback-done fallback fired ({duration:.2f}s)")
        self._process_pending_marks(message_category)

    # -- Interruption --

    async def handle_interruption(self):
        """FLUSH_MEDIA, increment generation to invalidate in-flight audio, clear all state."""
        logger.info("sip-trunk: handling interruption (FLUSH_MEDIA)")
        try:
            # Increment generation so in-flight frames are dropped
            self._flush_generation += 1

            if self._playback_done_task:
                self._playback_done_task.cancel()
                self._playback_done_task = None

            self._buffering_active = False
            self._response_audio_duration = 0.0
            self._awaiting_queue_drained = False
            self._pending_stop_after_drain = False
            self._jitter_buffer_bytes = 0
            self._jitter_buffer_filled = False

            # Clear local queue
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
            logger.error(f"sip-trunk handle_interruption: {e}")

    # -- Main audio handling --

    async def handle(self, ws_data_packet):
        """Process an audio packet from the synthesizer pipeline.

        Flow: validate → convert format → start buffering → jitter buffer → send flow-controlled
        """
        try:
            audio_chunk = ws_data_packet.get("data")
            meta_info = ws_data_packet.get("meta_info") or {}
            if self.stream_sid is None:
                self.stream_sid = meta_info.get("stream_sid")

            generation = self._flush_generation

            is_final = bool(
                (meta_info.get("end_of_llm_stream") and meta_info.get("end_of_synthesizer_stream"))
                or meta_info.get("is_final_chunk_of_entire_response")
                or (meta_info.get("sequence_id") == -1 and meta_info.get("end_of_llm_stream"))
            )
            has_audio = audio_chunk and len(audio_chunk) > 1 and audio_chunk != b"\x00\x00"

            if not has_audio and not is_final:
                return

            audio_format = (meta_info.get("format") or AudioFormat.ULAW.value).lower()
            audio_duration = 0.0

            if has_audio:
                # Pad odd-length PCM to even for audioop
                if len(audio_chunk) == 1:
                    audio_chunk += b"\x00"

                # Convert PCM/WAV to ulaw
                if audio_format in (AudioFormat.PCM.value, AudioFormat.WAV.value) or (
                    len(audio_chunk) > 44 and audio_chunk[:4] == b"RIFF"
                ):
                    if audio_chunk[:4] == b"RIFF":
                        audio_chunk = audio_chunk[44:]
                    audio_chunk = audioop.lin2ulaw(audio_chunk, 2)

                if meta_info.get("message_category") == "agent_welcome_message" and not self.welcome_message_sent_ts:
                    self.welcome_message_sent_ts = time.time() * 1000

                # Start buffering on first audio of a new response
                if not self._buffering_active and len(audio_chunk) > self._remote_buffer.optimal_frame_size:
                    await self._start_buffering()

                # Initial jitter buffer: accumulate before first send
                if not self._jitter_buffer_filled:
                    self._jitter_buffer_bytes += len(audio_chunk)
                    await self._local_audio_queue.put((audio_chunk, generation))
                    if self._jitter_buffer_bytes >= self._jitter_buffer_target:
                        self._jitter_buffer_filled = True
                        logger.debug("sip-trunk: jitter buffer filled, starting send")
                        # Drain accumulated jitter buffer
                        while not self._local_audio_queue.empty() and not self.queue_full:
                            try:
                                chunk, gen = self._local_audio_queue.get_nowait()
                            except asyncio.QueueEmpty:
                                break
                            if gen == self._flush_generation:
                                await self._send_binary(chunk)
                else:
                    await self._send_audio_flow_controlled(audio_chunk, generation)

                audio_duration = len(audio_chunk) / ASTERISK_ULAW_SAMPLE_RATE
                self._response_audio_duration += audio_duration

            # Update mark metadata for latency tracking
            if self.mark_event_meta_data:
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

                if is_final:
                    total_duration = self._response_audio_duration
                    # DO NOT reset _response_audio_duration here.
                    # is_final fires twice per response (last audio + null-byte sentinel).
                    # Resetting here would cause the second fire to schedule a 0-duration fallback.
                    # Reset happens in _start_buffering() for the next response.

                    if not self._local_audio_queue.empty():
                        self._pending_stop_after_drain = True
                        self._pending_stop_duration = total_duration
                        self._pending_stop_category = message_category
                        logger.debug("sip-trunk: final chunk queued; will finish after drain")
                    else:
                        if total_duration > 0 or not self._playback_done_task:
                            await self._finish_response(total_duration, message_category)

        except Exception as e:
            logger.error(f"sip-trunk output error: {e}")
            traceback.print_exc()

    # -- Stub methods for base class compatibility --

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
            logger.error(f"sip-trunk send_hangup: {e}")

    def requires_custom_voicemail_detection(self):
        return False
