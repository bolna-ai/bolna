"""
Asterisk WebSocket (chan_websocket) output handler for sip-trunk provider.

Sends audio to Asterisk in bursts as it arrives from TTS (like Plivo/Twilio).
START_MEDIA_BUFFERING once per call lets Asterisk reframe and retime for smooth
RTP delivery. Server-side duration tracking knows when playback ends without
relying on QUEUE_DRAINED. FLUSH_MEDIA on interrupt. Generation counter drops
stale audio after interruption.

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

# Asterisk's max WebSocket frame size is 65,500 bytes. Chunks larger than this
# cause "Cannot fit huge websocket frame" and kill the connection.
MAX_WS_FRAME_BYTES = 60000  # leave headroom below 65,500

# Extra buffer after estimated playback end before clearing is_audio_being_played.
# Accounts for Asterisk's internal retiming and RTP jitter buffer.
PLAYBACK_SETTLE_S = float(os.environ.get("SIP_PLAYBACK_SETTLE_S", "0.1"))


class SipTrunkOutputHandler(TelephonyOutputHandler):
    """
    Sends ulaw audio to Asterisk in bursts — Asterisk buffers and retimes.
    Server-side duration tracking for playback completion.
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
        self._response_first_send: float = 0.0   # monotonic time first chunk was sent
        self._response_audio_duration: float = 0.0  # accumulated seconds of audio
        self._settle_task: asyncio.Task | None = None
        self._pending_finish: bool = False
        self._current_sequence_id: int | None = None  # track sequence to detect new responses
        self._response_sealed: bool = False  # True once is_final fires; blocks further audio for this sequence

        # Generation counter — incremented on interruption, stale audio is dropped
        self._flush_generation: int = 0

        # Asterisk buffering state
        self._start_buffering_sent: bool = False

        # XOFF/XON: local queue when Asterisk signals backpressure
        self._local_audio_queue: deque = deque()

        output_config = self._get_output_config()
        self._output_format = (
            (output_config.get("audio_format") or output_config.get("format") or "ulaw")
        ).lower()
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

        self._settle_task = asyncio.create_task(
            self._settle_and_finish(generation, delay)
        )
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
        """Process all pending marks and clear playback state."""
        if not self.input_handler or not self.input_handler.mark_event_meta_data:
            return
        remaining = list(self.input_handler.mark_event_meta_data.mark_event_meta_data.keys())
        logger.info(f"sip-trunk: _finish_playback reason={reason}, {len(remaining)} mark(s)")
        for mid in remaining:
            self.input_handler.process_mark_message({"name": mid})
        if self.input_handler.is_audio_being_played_to_user():
            self.input_handler.update_is_audio_being_played(False)

    # ------------------------------------------------------------------
    # Interruption
    # ------------------------------------------------------------------

    async def handle_interruption(self):
        """FLUSH_MEDIA, clear local queue, increment generation to drop stale audio."""
        logger.info("sip-trunk: handling interruption (FLUSH_MEDIA)")
        try:
            if self._settle_task:
                self._settle_task.cancel()
                self._settle_task = None

            self._flush_generation += 1
            self._pending_finish = False
            self._response_audio_duration = 0.0
            self._response_first_send = 0.0
            self._current_sequence_id = None
            self._response_sealed = False
            self._local_audio_queue.clear()

            await self._send_control("FLUSH_MEDIA")
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
        """Send XOFF-queued audio to Asterisk (called on MEDIA_XON)."""
        gen = self._flush_generation
        while self._local_audio_queue and not self.queue_full:
            # Drop stale audio from before an interruption
            if gen != self._flush_generation:
                self._local_audio_queue.clear()
                self._pending_finish = False
                return
            chunk = self._local_audio_queue.popleft()
            if not chunk:
                continue
            try:
                if self._closed:
                    return
                offset = 0
                while offset < len(chunk):
                    if gen != self._flush_generation:
                        self._local_audio_queue.clear()
                        self._pending_finish = False
                        return
                    if self.queue_full:
                        # XOFF arrived mid-drain — put remainder back and stop
                        self._local_audio_queue.appendleft(chunk[offset:])
                        return
                    end = min(offset + MAX_WS_FRAME_BYTES, len(chunk))
                    await self.websocket.send_bytes(chunk[offset:end])
                    if self._response_first_send == 0.0:
                        self._response_first_send = time.monotonic()
                    offset = end
            except Exception as e:
                logger.debug(f"sip-trunk drain send_bytes stopped: {e}")
                return

        if not self._local_audio_queue and self._pending_finish:
            self._pending_finish = False
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
        """
        Send audio directly to Asterisk in bursts. Asterisk's START_MEDIA_BUFFERING
        reframes and retimes for smooth RTP. Server-side duration tracking knows
        when playback ends.
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
            gen = self._flush_generation

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

                # New response — reset duration tracking only when sequence_id
                # actually changes.  ElevenLabs may spuriously set is_first_chunk
                # on every chunk after the LLM stream ends, which would repeatedly
                # reset the audio-duration accumulator and make the server think
                # playback finishes far earlier than it actually does.
                seq_id = meta_info.get("sequence_id")
                if seq_id is not None and seq_id != self._current_sequence_id:
                    self._current_sequence_id = seq_id
                    self._response_first_send = 0.0
                    self._response_audio_duration = 0.0
                    self._pending_finish = False
                    self._response_sealed = False
                    if self._settle_task:
                        self._settle_task.cancel()
                        self._settle_task = None

                # After the null-byte sentinel seals this response, drop any
                # further audio.  Stale chunks may arrive from the synthesizer
                # with end_of_synthesizer_stream still True on the shared
                # meta_info; Plivo/Twilio ignore these (dumb pipe + mark echo),
                # but Asterisk will buffer and play every byte we send.
                if self._response_sealed:
                    return

                # Send START_MEDIA_BUFFERING once per call (or after FLUSH_MEDIA reset)
                if not self._start_buffering_sent:
                    await self._send_control("START_MEDIA_BUFFERING")
                    self._start_buffering_sent = True
                    logger.info("sip-trunk: START_MEDIA_BUFFERING sent")

                # Track audio duration
                audio_duration = len(audio_chunk) / 8000.0
                self._response_audio_duration += audio_duration

                # Drop stale audio from a previous generation
                if gen != self._flush_generation:
                    return

                # Send to Asterisk in chunks ≤ MAX_WS_FRAME_BYTES (Asterisk limit: 65,500).
                # If XOFF arrives mid-send, queue the remainder locally.
                # Also queue if drain is still in-flight to preserve ordering.
                if self.queue_full or self._local_audio_queue:
                    self._local_audio_queue.append(audio_chunk)
                else:
                    if self._response_first_send == 0.0:
                        self._response_first_send = time.monotonic()
                    try:
                        if self._closed:
                            return
                        offset = 0
                        while offset < len(audio_chunk):
                            if gen != self._flush_generation:
                                return
                            if self.queue_full:
                                # XOFF arrived mid-send — queue remainder at front
                                self._local_audio_queue.appendleft(audio_chunk[offset:])
                                break
                            end = min(offset + MAX_WS_FRAME_BYTES, len(audio_chunk))
                            await self.websocket.send_bytes(audio_chunk[offset:end])
                            offset = end
                    except Exception as e:
                        logger.debug(f"sip-trunk send_bytes stopped: {e}")
                        return

            # Register mark metadata (same as Plivo/Twilio contract)
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

                # Only schedule playback finish on the null-byte sentinel
                # (has_audio=False, is_final=True) — the TRUE end of response.
                # ElevenLabs may set end_of_synthesizer_stream on many audio
                # chunks (stale meta_info), making is_final=True repeatedly.
                # Firing _schedule_finish on those causes premature timers that
                # clear is_audio_being_played mid-response, truncating audio.
                # Plivo/Twilio are unaffected because they ignore is_final
                # entirely (mark echo handles playback tracking).
                if is_final and not has_audio:
                    self._response_sealed = True
                    if self._local_audio_queue:
                        # XOFF queued audio not yet sent — defer finish
                        self._pending_finish = True
                        logger.debug("sip-trunk: final chunk, XOFF audio pending; deferring finish")
                    else:
                        self._pending_finish = False
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
        return False
