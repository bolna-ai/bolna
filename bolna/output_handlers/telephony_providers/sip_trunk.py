"""
Asterisk WebSocket (chan_websocket) output handler for sip-trunk provider.
Audio is always ulaw over BINARY frames; control commands over TEXT frames (JSON or plain text).
Ref: https://docs.asterisk.org/Configuration/Channel-Drivers/WebSocket/#control-commands-and-events
"""
import json
import time
import uuid
import traceback
import audioop
import asyncio
from bolna.output_handlers.telephony import TelephonyOutputHandler
from bolna.helpers.logger_config import configure_logger
from dotenv import load_dotenv

logger = configure_logger(__name__)
load_dotenv()

# Asterisk ulaw: 160 bytes per 20ms frame
ASTERISK_ULAW_OPTIMAL_FRAME_SIZE = 160


def _asterisk_command(cmd: str, **kwargs) -> str:
    """Format Asterisk command as JSON TEXT frame (preferred in 20.18+, 22.8+, 23.2+)."""
    obj = {"command": cmd, **kwargs}
    return json.dumps(obj)


class SipTrunkOutputHandler(TelephonyOutputHandler):
    """Sends ulaw audio as BINARY frames; control commands (HANGUP, FLUSH_MEDIA, etc.) as TEXT."""

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
        super().__init__(
            io_provider,
            websocket,
            mark_event_meta_data,
            log_dir_name,
        )
        self.asterisk_media_start = asterisk_media_start or {}
        self.agent_config = agent_config or {}
        self._optimal_frame_size = ASTERISK_ULAW_OPTIMAL_FRAME_SIZE
        self._media_paused = False
        self.input_handler = input_handler  # Reference to input handler to simulate mark events
        self.queue_full = False

        # Per-response buffering state: a single START/STOP_MEDIA_BUFFERING pair
        # wraps all audio chunks for one agent response so Asterisk can properly
        # re-frame/re-time the media and QUEUE_DRAINED fires only after the
        # entire response has been played out.
        self._buffering_active = False
        self._response_audio_duration = 0.0
        self._mark_simulation_task = None

        # Resolve format from agent_config (must be ulaw for sip-trunk)
        output_config = self._get_output_config()
        self._output_format = (
            (output_config.get("audio_format") or output_config.get("format") or "ulaw")
        ).lower()
        if self._output_format not in ("ulaw", "mulaw"):
            logger.warning(
                f"sip-trunk output expects ulaw from agent_config; got {self._output_format}, using ulaw"
            )
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

    async def handle_interruption(self):
        """Send FLUSH_MEDIA to Asterisk to discard queued but not sent frames."""
        logger.info("Handling interruption - flushing media queue")
        try:
            # Cancel any pending fallback mark simulation
            if self._mark_simulation_task:
                self._mark_simulation_task.cancel()
                self._mark_simulation_task = None

            # FLUSH_MEDIA automatically ends any bulk transfer in progress
            # (per Asterisk docs: no need to send STOP_MEDIA_BUFFERING)
            self._buffering_active = False
            self._response_audio_duration = 0.0

            # Clear pending queue-drain mark on input handler
            if self.input_handler:
                self.input_handler._pending_queue_drain_mark_id = None
                self.input_handler._pending_queue_drain_category = None

            await self.flush_media()
            self.mark_event_meta_data.clear_data()
        except Exception as e:
            logger.error(f"Error handling interruption: {e}")

    async def flush_media(self):
        """Flush queued media"""
        await self.send_control_command('FLUSH_MEDIA')

    async def send_control_command(self, command, params=None):
        """Send control command to Asterisk via TEXT frame (plain text format)."""
        try:
            if params:
                param_str = ' '.join(str(v) for v in params.values())
                message = f"{command} {param_str}"
            else:
                message = command
            await self.websocket.send_text(message)
            logger.info(f"Sent Asterisk control command: {command}")
        except Exception as e:
            logger.error(f"Error sending control command {command}: {e}")
            traceback.print_exc()

    async def form_media_message(self, audio_data, audio_format="wav"):
        """Not used for Asterisk; we send raw BINARY. Kept for interface compatibility."""
        return None

    async def form_mark_message(self, mark_id):
        """Not used for Asterisk; no mark events. Kept for interface compatibility."""
        return None

    async def set_stream_sid(self, stream_id):
        self.stream_sid = stream_id

    async def handle(self, ws_data_packet):
        """Send audio as BINARY (ulaw) with consolidated per-response buffering.

        A single START_MEDIA_BUFFERING / STOP_MEDIA_BUFFERING pair wraps every
        chunk of one agent response.  REPORT_QUEUE_DRAINED is sent after STOP so
        that Asterisk fires QUEUE_DRAINED only when the *entire* response has
        been played out — giving us an accurate ``is_audio_being_played`` signal
        equivalent to Twilio's mark events.
        """
        try:
            audio_chunk = ws_data_packet.get("data")
            meta_info = ws_data_packet.get("meta_info") or {}
            if self.stream_sid is None:
                self.stream_sid = meta_info.get("stream_sid")

            # Skip if queue is full to avoid overwhelming Asterisk
            if self.queue_full:
                logger.warning("[SIP-TRUNK OUTPUT] Skipping audio send - Asterisk queue is full (XOFF state)")
                return

            # Determine is_final early so we can finalize buffering even when
            # the final synthesizer message carries no audio payload.
            is_final = (meta_info.get("end_of_llm_stream", False) and meta_info.get("end_of_synthesizer_stream", False)) or \
                      meta_info.get("is_final_chunk_of_entire_response", False) or \
                      (meta_info.get("sequence_id") == -1 and meta_info.get("end_of_llm_stream", False))

            has_audio = audio_chunk and len(audio_chunk) > 1 and audio_chunk != b"\x00\x00"

            if not has_audio and not is_final:
                return

            if has_audio:
                # Handle edge cases with audio chunk size
                if len(audio_chunk) == 1:
                    audio_chunk += b'\x00'

                audio_format = (meta_info.get("format", "ulaw") or "ulaw").lower()
                # Convert to ulaw when PCM, or when data looks like WAV (cached welcome may be PCM)
                if audio_format in ("pcm", "wav") or (len(audio_chunk) > 44 and audio_chunk[:4] == b"RIFF"):
                    if len(audio_chunk) > 44 and audio_chunk[:4] == b"RIFF":
                        audio_chunk = audio_chunk[44:]
                    audio_chunk = audioop.lin2ulaw(audio_chunk, 2)

                # Track welcome message timing
                if meta_info.get('message_category', '') == 'agent_welcome_message' and not self.welcome_message_sent_ts:
                    self.welcome_message_sent_ts = time.time() * 1000

                # --- Consolidated buffering: START once per response ---
                if not self._buffering_active and len(audio_chunk) > self._optimal_frame_size:
                    await self.send_control_command('START_MEDIA_BUFFERING')
                    self._buffering_active = True
                    self._response_audio_duration = 0.0

                # Send audio as BINARY frame(s)
                original_size = len(audio_chunk)
                bytes_sent = await self._send_audio_data(audio_chunk)

                # Accumulate total response duration
                audio_duration = self._calculate_audio_duration(audio_chunk, audio_format)
                self._response_audio_duration += audio_duration

                logger.info(f"[SIP-TRUNK OUTPUT] Sent {bytes_sent} bytes of audio to Asterisk (chunked from {original_size} bytes)")
            else:
                audio_duration = 0.0

            # Track mark events (Asterisk has no native marks; we emulate them)
            if self.mark_event_meta_data:
                message_category = meta_info.get('message_category', 'agent_response')

                mark_event_meta_data = {
                    "text_synthesized": "" if meta_info.get("sequence_id") == -1 else meta_info.get("text_synthesized", ""),
                    "type": message_category,
                    "is_first_chunk": meta_info.get("is_first_chunk", False),
                    "is_final_chunk": is_final,
                    "sequence_id": meta_info.get("sequence_id", 0),
                    "duration": audio_duration,
                    "sent_ts": time.time(),
                }
                mark_id = meta_info.get("mark_id") or str(uuid.uuid4())
                self.mark_event_meta_data.update_data(mark_id, mark_event_meta_data)

                if is_final:
                    # --- Consolidated buffering: STOP once per response ---
                    if self._buffering_active:
                        await self.send_control_command('STOP_MEDIA_BUFFERING')
                        self._buffering_active = False

                    # Ask Asterisk to notify when playback truly finishes
                    await self.send_control_command('REPORT_QUEUE_DRAINED')

                    # Store pending mark for QUEUE_DRAINED processing (input handler)
                    if self.input_handler:
                        self.input_handler._pending_queue_drain_mark_id = mark_id
                        self.input_handler._pending_queue_drain_category = message_category

                    # Schedule a fallback mark simulation using the *total*
                    # accumulated duration (not just the final chunk) + buffer.
                    # QUEUE_DRAINED is authoritative; this fires only if it doesn't arrive.
                    if self.input_handler:
                        if self._mark_simulation_task:
                            self._mark_simulation_task.cancel()
                        total_duration = self._response_audio_duration
                        self._mark_simulation_task = asyncio.create_task(
                            self._simulate_mark_event_after_duration(mark_id, total_duration, message_category))
                        logger.info(
                            f"[SIP-TRUNK OUTPUT] Final chunk sent. Total response duration: {total_duration:.3f}s. "
                            f"Waiting for QUEUE_DRAINED (fallback after {total_duration + 1.0:.1f}s).")

                    self._response_audio_duration = 0.0

        except Exception as e:
            traceback.print_exc()
            logger.error(f"sip-trunk (Asterisk) output error: {e}")

    def _calculate_audio_duration(self, audio_data, audio_format):
        """Calculate audio duration based on format (ulaw: 8kHz, 1 byte/sample)."""
        if audio_format in ('ulaw', 'alaw', 'mulaw'):
            return len(audio_data) / 8000.0
        elif audio_format in ('slin', 'pcm'):
            return len(audio_data) / 16000.0
        elif audio_format == 'slin16':
            return len(audio_data) / 32000.0
        else:
            return len(audio_data) / 8000.0  # Default to ulaw

    def _get_chunk_size(self, audio_format):
        """Get optimal chunk size for sending audio to Asterisk (100ms chunks)."""
        if audio_format in ('ulaw', 'alaw', 'mulaw'):
            return 800  # 100ms at 8kHz = 800 bytes
        elif audio_format in ('slin', 'pcm'):
            return 1600  # 100ms at 8kHz = 1600 bytes
        elif audio_format == 'slin16':
            return 3200  # 100ms at 16kHz = 3200 bytes
        else:
            return 800  # Default to ulaw

    async def _send_audio_data(self, audio_data):
        """Send audio as BINARY frame(s), chunking large payloads for responsiveness.

        Buffering commands (START/STOP_MEDIA_BUFFERING) are managed at the
        per-response level by ``handle()``; this method only pushes raw bytes.
        """
        chunk_size = self._get_chunk_size(
            (self._output_format if hasattr(self, '_output_format') else "ulaw"))
        total_bytes = len(audio_data)

        if total_bytes <= chunk_size:
            await self.websocket.send_bytes(audio_data)
            return total_bytes

        bytes_sent = 0
        offset = 0
        while offset < total_bytes:
            chunk = audio_data[offset:offset + chunk_size]
            if chunk:
                await self.websocket.send_bytes(chunk)
                bytes_sent += len(chunk)
                if offset + chunk_size < total_bytes:
                    await asyncio.sleep(0.001)
            offset += chunk_size
        return bytes_sent

    async def _simulate_mark_event_after_duration(self, mark_id, duration, message_category):
        """Fallback: simulate a mark event if QUEUE_DRAINED doesn't arrive in time.

        QUEUE_DRAINED from Asterisk is the authoritative signal that playback has
        finished.  This task fires only as a safety net (total audio duration +
        1 s buffer).  If QUEUE_DRAINED already processed the mark, the pending
        id will have been cleared and this becomes a no-op.
        """
        try:
            await asyncio.sleep(duration + 1.0)
            if not self.input_handler:
                return
            # Check whether QUEUE_DRAINED already handled this mark
            if getattr(self.input_handler, '_pending_queue_drain_mark_id', None) == mark_id:
                logger.info(
                    f"[SIP-TRUNK OUTPUT] QUEUE_DRAINED fallback: simulating mark after "
                    f"{duration:.3f}s for {message_category}")
                self.input_handler._pending_queue_drain_mark_id = None
                self.input_handler._pending_queue_drain_category = None
                self.input_handler.update_is_audio_being_played(False)
                mark_packet = {"name": mark_id, "type": message_category}
                self.input_handler.process_mark_message(mark_packet)
        except asyncio.CancelledError:
            pass  # Cancelled by interruption — expected
        except Exception as e:
            logger.error(f"Error in mark event fallback: {e}")

    async def send_hangup(self):
        """Send HANGUP command to Asterisk (TEXT frame)."""
        await self.send_control_command('HANGUP')

    def set_hangup_sent(self):
        """Mark hangup as sent and send HANGUP to Asterisk so the call is cut."""
        super().set_hangup_sent()
        try:
            asyncio.create_task(self.send_hangup())
        except Exception as e:
            logger.error(f"Error scheduling HANGUP for Asterisk: {e}")

    def requires_custom_voicemail_detection(self):
        return False
