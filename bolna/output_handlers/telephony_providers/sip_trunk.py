"""
Asterisk WebSocket (chan_websocket) output handler for sip-trunk provider.

Sends ulaw audio at 1x real-time pace (160 bytes / 20ms) via a background send loop.
START_MEDIA_BUFFERING once at call start; FLUSH_MEDIA on interrupt.
Generation counter drops stale frames after interruption.

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

# 160 bytes = 20ms of ulaw at 8kHz (one RTP frame)
FRAME_SIZE = 160
FRAME_DURATION_S = 0.02

# Pre-buffer: accumulate this many ms of audio before sending first frame of a response.
# Absorbs TTS jitter so the first frames don't have micro-gaps.
DEFAULT_JITTER_BUFFER_MS = int(os.environ.get("SIP_JITTER_BUFFER_MS", "40"))

# Small delay after QUEUE_DRAINED for Asterisk's remaining RTP jitter buffer (seconds).
QUEUE_DRAINED_BUFFER_S = float(os.environ.get("SIP_QUEUE_DRAINED_BUFFER_S", "0.1"))

# Safety timeout if QUEUE_DRAINED is never received (seconds).
QUEUE_DRAINED_SAFETY_TIMEOUT_S = float(os.environ.get("SIP_QUEUE_DRAINED_TIMEOUT_S", "2.0"))


class SipTrunkOutputHandler(TelephonyOutputHandler):
    """
    Sends ulaw audio to Asterisk at 1x real-time pace via a background send loop.
    START_MEDIA_BUFFERING sent once per call. FLUSH_MEDIA on interrupt.
    MEDIA_XOFF/XON flow control queues audio locally when Asterisk is full.
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
        self._optimal_frame_size = FRAME_SIZE
        self.input_handler = input_handler
        self.queue_full = False
        if input_handler:
            input_handler.output_handler_ref = self

        self._response_audio_duration = 0.0
        self._playback_done_task = None
        self._awaiting_queue_drained: bool = False

        # Send loop: frames are enqueued as (bytes, generation) tuples and sent at 1x real-time
        self._send_queue: asyncio.Queue = asyncio.Queue()
        self._send_loop_task = None
        self._flush_generation = 0
        self._start_buffering_sent = False

        # Local queue when Asterisk sends MEDIA_XOFF; drained on MEDIA_XON
        self._local_audio_queue: deque = deque()
        # If is_final arrived while send queue still has frames, defer fallback
        self._pending_stop_after_drain = False

        output_config = self._get_output_config()
        self._output_format = (
            (output_config.get("audio_format") or output_config.get("format") or "ulaw")
        ).lower()
        if self._output_format not in ("ulaw", "mulaw"):
            self._output_format = "ulaw"

        opt = self.asterisk_media_start.get("optimal_frame_size")
        if opt is not None:
            try:
                self._optimal_frame_size = int(opt)
            except (TypeError, ValueError):
                pass

    def _get_output_config(self):
        try:
            tasks = self.agent_config.get("tasks") or []
            if tasks and isinstance(tasks[0], dict):
                return tasks[0].get("tools_config", {}).get("output") or {}
        except Exception:
            pass
        return {}

    # ------------------------------------------------------------------
    # Send loop — drains _send_queue at 1x real-time (160 bytes / 20ms)
    # ------------------------------------------------------------------

    def _ensure_send_loop(self):
        """Start the background send loop if not already running."""
        if self._send_loop_task is None or self._send_loop_task.done():
            self._send_loop_task = asyncio.create_task(self._send_loop())

    async def _send_loop(self):
        """Dequeue frames and send at 1x real-time pace."""
        jitter_buffer_frames = max(1, DEFAULT_JITTER_BUFFER_MS // 20)
        next_send = 0.0

        while True:
            try:
                frame, generation = await self._send_queue.get()
            except asyncio.CancelledError:
                return

            # Discard stale frames from a previous generation (post-interruption)
            if generation != self._flush_generation:
                continue

            # Pre-buffer: if this is the first frame after the queue was empty,
            # wait briefly to accumulate more frames and absorb TTS jitter.
            if self._send_queue.qsize() < jitter_buffer_frames - 1:
                try:
                    await asyncio.sleep(DEFAULT_JITTER_BUFFER_MS / 1000.0)
                except asyncio.CancelledError:
                    return

            # Respect XOFF backpressure
            if self.queue_full:
                self._local_audio_queue.append(frame)
                continue

            # Pace: wait until next_send time
            now = time.monotonic()
            if next_send > now:
                try:
                    await asyncio.sleep(next_send - now)
                except asyncio.CancelledError:
                    return

            # Re-check generation after sleep (interruption may have happened)
            if generation != self._flush_generation:
                continue

            try:
                if self._closed:
                    return
                await self.websocket.send_bytes(frame)
            except Exception as e:
                logger.debug(f"sip-trunk send_bytes stopped: {e}")
                return

            next_send = time.monotonic() + FRAME_DURATION_S

            # If the send queue just drained and we have a pending fallback, fire it
            if self._send_queue.empty() and self._pending_stop_after_drain and not self._local_audio_queue:
                self._pending_stop_after_drain = False
                self._awaiting_queue_drained = True
                await self._send_control("REPORT_QUEUE_DRAINED")
                if self._playback_done_task:
                    self._playback_done_task.cancel()
                self._playback_done_task = asyncio.create_task(
                    self._playback_done_fallback()
                )

    def _enqueue_audio(self, audio_chunk: bytes):
        """Split audio_chunk into FRAME_SIZE frames and enqueue with current generation."""
        gen = self._flush_generation
        offset = 0
        n = len(audio_chunk)
        while offset < n:
            end = min(offset + FRAME_SIZE, n)
            frame = audio_chunk[offset:end]
            # Pad last frame to FRAME_SIZE if short (Asterisk expects uniform frames)
            if len(frame) < FRAME_SIZE:
                frame = frame + b"\xff" * (FRAME_SIZE - len(frame))
            self._send_queue.put_nowait((frame, gen))
            offset = end

    # ------------------------------------------------------------------
    # Interruption
    # ------------------------------------------------------------------

    async def handle_interruption(self):
        """FLUSH_MEDIA, clear queues, increment generation to drop stale frames."""
        logger.info("sip-trunk: handling interruption (FLUSH_MEDIA)")
        try:
            if self._playback_done_task:
                self._playback_done_task.cancel()
                self._playback_done_task = None

            # Increment generation so send loop discards any in-flight frames
            self._flush_generation += 1
            self._response_audio_duration = 0.0
            self._pending_stop_after_drain = False
            self._awaiting_queue_drained = False

            # Clear both queues
            while not self._send_queue.empty():
                try:
                    self._send_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            self._local_audio_queue.clear()

            await self._send_control("FLUSH_MEDIA")
            if self.mark_event_meta_data:
                self.mark_event_meta_data.clear_data()
            if self.input_handler:
                self.input_handler.update_is_audio_being_played(False)
        except Exception as e:
            logger.error(f"sip-trunk handle_interruption: {e}")

    # ------------------------------------------------------------------
    # XOFF/XON drain
    # ------------------------------------------------------------------

    async def drain_local_queue(self):
        """Re-enqueue XOFF-queued audio into the send queue (called on MEDIA_XON)."""
        gen = self._flush_generation
        while self._local_audio_queue and not self.queue_full:
            chunk = self._local_audio_queue.popleft()
            if not chunk:
                continue
            # Re-enqueue as individual frames so pacing is maintained
            offset = 0
            n = len(chunk)
            while offset < n:
                end = min(offset + FRAME_SIZE, n)
                frame = chunk[offset:end]
                if len(frame) < FRAME_SIZE:
                    frame = frame + b"\xff" * (FRAME_SIZE - len(frame))
                self._send_queue.put_nowait((frame, gen))
                offset = end

        if not self._local_audio_queue and self._pending_stop_after_drain:
            self._pending_stop_after_drain = False
            self._awaiting_queue_drained = True
            await self._send_control("REPORT_QUEUE_DRAINED")
            if self._playback_done_task:
                self._playback_done_task.cancel()
            self._playback_done_task = asyncio.create_task(
                self._playback_done_fallback()
            )

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

    async def form_media_message(self, audio_data, audio_format="wav"):
        return None

    async def form_mark_message(self, mark_id):
        return None

    async def set_stream_sid(self, stream_id):
        self.stream_sid = stream_id

    def _duration_ulaw(self, num_bytes):
        return num_bytes / 8000.0

    # ------------------------------------------------------------------
    # Main handle — enqueue audio into the paced send loop
    # ------------------------------------------------------------------

    async def handle(self, ws_data_packet):
        """
        Enqueue audio into the 1x real-time send loop.
        START_MEDIA_BUFFERING is sent once per call on first audio.
        """
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

            if has_audio:
                if len(audio_chunk) == 1:
                    audio_chunk += b"\x00"
                if audio_format in ("pcm", "wav") or (len(audio_chunk) > 44 and audio_chunk[:4] == b"RIFF"):
                    if audio_chunk[:4] == b"RIFF":
                        audio_chunk = audio_chunk[44:]
                    audio_chunk = audioop.lin2ulaw(audio_chunk, 2)
                    audio_format = "ulaw"

                if meta_info.get("message_category") == "agent_welcome_message" and not self.welcome_message_sent_ts:
                    self.welcome_message_sent_ts = time.time() * 1000

                # Send START_MEDIA_BUFFERING once per call on first audio
                if not self._start_buffering_sent:
                    await self._send_control("START_MEDIA_BUFFERING")
                    self._start_buffering_sent = True
                    logger.info("sip-trunk: START_MEDIA_BUFFERING sent (once per call)")

                # Reset duration tracking at the start of each new response
                if meta_info.get("is_first_chunk"):
                    self._response_audio_duration = 0.0

                # Ensure the background send loop is running
                self._ensure_send_loop()

                # Enqueue frames — the send loop handles pacing, XOFF, and generation
                self._enqueue_audio(audio_chunk)

                audio_duration = self._duration_ulaw(len(audio_chunk))
                self._response_audio_duration += audio_duration

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
                    # _response_audio_duration is reset on the next new response
                    # (when _start_buffering_sent logic resets it), not here.
                    # is_final fires twice per response (last audio chunk +
                    # null-byte sentinel) and resetting here would replace a
                    # valid fallback timer with a 0-duration one.

                    if self._send_queue.qsize() > 0 or self._local_audio_queue:
                        # Audio still being sent/queued — defer fallback
                        self._pending_stop_after_drain = True
                        logger.debug("sip-trunk: final chunk received, audio still in send queue; deferring fallback")
                    else:
                        self._awaiting_queue_drained = True
                        await self._send_control("REPORT_QUEUE_DRAINED")
                        if total_duration > 0 or not self._playback_done_task:
                            if self._playback_done_task:
                                self._playback_done_task.cancel()
                            self._playback_done_task = asyncio.create_task(
                                self._playback_done_fallback()
                            )
                        logger.debug(f"sip-trunk: response done ({total_duration:.2f}s audio), safety fallback in {QUEUE_DRAINED_SAFETY_TIMEOUT_S}s")

        except Exception as e:
            logger.error(f"sip-trunk output error: {e}")
            traceback.print_exc()

    def _finish_playback(self, reason: str = "fallback") -> None:
        """Process all pending marks and clear playback state.

        Delegates to input_handler.process_mark_message() for each mark so
        that welcome-message detection, hangup observables, latency tracking,
        and response_heard_by_user all fire — same contract as Plivo/Twilio
        mark echoes.
        """
        if not self.input_handler or not self.input_handler.mark_event_meta_data:
            return
        remaining = list(self.input_handler.mark_event_meta_data.mark_event_meta_data.keys())
        if remaining:
            logger.info(
                f"sip-trunk: playback done ({reason}), processing {len(remaining)} mark(s)"
            )
        for mid in remaining:
            self.input_handler.process_mark_message({"name": mid})
        # Safety: if process_mark_message didn't clear it (e.g., no final mark
        # registered), force-clear so the flag doesn't stay stuck True.
        if self.input_handler.is_audio_being_played_to_user():
            self.input_handler.update_is_audio_being_played(False)

    async def on_queue_drained(self) -> None:
        """Called when Asterisk sends QUEUE_DRAINED — audio consumed for RTP.

        Cancels the safety fallback and clears playback state after a short
        buffer for the remaining RTP jitter.
        """
        if not self._awaiting_queue_drained:
            return  # Spurious or duplicate QUEUE_DRAINED
        self._awaiting_queue_drained = False

        if self._playback_done_task:
            self._playback_done_task.cancel()
            self._playback_done_task = None

        # Small sleep for Asterisk's RTP jitter buffer, then clear
        try:
            await asyncio.sleep(QUEUE_DRAINED_BUFFER_S)
        except asyncio.CancelledError:
            return

        if self._closed:
            return
        self._finish_playback(reason="QUEUE_DRAINED")

    async def _playback_done_fallback(self):
        """Safety timeout: clear playback state if QUEUE_DRAINED never arrives."""
        try:
            await asyncio.sleep(QUEUE_DRAINED_SAFETY_TIMEOUT_S)
        except asyncio.CancelledError:
            return
        self._playback_done_task = None
        if self._closed:
            return
        if self._awaiting_queue_drained:
            logger.warning(
                f"sip-trunk: QUEUE_DRAINED not received within "
                f"{QUEUE_DRAINED_SAFETY_TIMEOUT_S}s (audio duration {self._response_audio_duration:.2f}s), "
                f"using safety fallback"
            )
            self._awaiting_queue_drained = False
        self._finish_playback(reason="safety-timeout")

    async def send_hangup(self):
        await self._send_control("HANGUP")

    def set_hangup_sent(self):
        super().set_hangup_sent()
        try:
            asyncio.create_task(self.send_hangup())
        except Exception as e:
            logger.error(f"sip-trunk send_hangup: {e}")

    def requires_custom_voicemail_detection(self):
        return False
