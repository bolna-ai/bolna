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
        """Send audio as BINARY (ulaw); no Twilio-style mark/media JSON."""
        try:
            audio_chunk = ws_data_packet.get("data")
            meta_info = ws_data_packet.get("meta_info") or {}
            if self.stream_sid is None:
                self.stream_sid = meta_info.get("stream_sid")

            # Skip if queue is full to avoid overwhelming Asterisk
            if self.queue_full:
                logger.warning("[SIP-TRUNK OUTPUT] Skipping audio send - Asterisk queue is full (XOFF state)")
                return

            if not audio_chunk or len(audio_chunk) <= 1:
                return
            if audio_chunk == b"\x00\x00":
                return

            # Handle edge cases with audio chunk size
            if len(audio_chunk) == 1:
                audio_chunk += b'\x00'

            audio_format = (meta_info.get("format", "ulaw") or "ulaw").lower()
            # Convert to ulaw when PCM, or when data looks like WAV (cached welcome may be PCM)
            if audio_format in ("pcm", "wav") or (len(audio_chunk) > 44 and audio_chunk[:4] == b"RIFF"):
                # Strip standard WAV header if present (44 bytes), then convert 16-bit PCM to ulaw
                if len(audio_chunk) > 44 and audio_chunk[:4] == b"RIFF":
                    audio_chunk = audio_chunk[44:]
                audio_chunk = audioop.lin2ulaw(audio_chunk, 2)

            # Track welcome message timing
            if meta_info.get('message_category', '') == 'agent_welcome_message' and not self.welcome_message_sent_ts:
                self.welcome_message_sent_ts = time.time() * 1000

            # Send audio as BINARY frame to Asterisk (chunked for large data)
            original_size = len(audio_chunk)
            bytes_sent = await self._send_audio_chunked(audio_chunk, audio_format)
            logger.info(f"[SIP-TRUNK OUTPUT] Sent {bytes_sent} bytes of audio to Asterisk (chunked from {original_size} bytes)")

            # Track mark events internally for our own timing/debugging
            # (Asterisk doesn't have mark events, but we track for compatibility)
            if self.mark_event_meta_data:
                audio_duration = self._calculate_audio_duration(audio_chunk, audio_format)
                
                # Check if this is the final chunk
                is_final = (meta_info.get("end_of_llm_stream", False) and meta_info.get("end_of_synthesizer_stream", False)) or \
                          meta_info.get("is_final_chunk_of_entire_response", False) or \
                          (meta_info.get("sequence_id") == -1 and meta_info.get("end_of_llm_stream", False))
                
                message_category = meta_info.get('message_category', 'agent_response')
                
                mark_event_meta_data = {
                    "text_synthesized": "" if meta_info["sequence_id"] == -1 else meta_info.get("text_synthesized", ""),
                    "type": message_category,
                    "is_first_chunk": meta_info.get("is_first_chunk", False),
                    "is_final_chunk": is_final,
                    "sequence_id": meta_info.get("sequence_id", 0),
                    "duration": audio_duration,
                    "sent_ts": time.time(),
                }
                mark_id = meta_info.get("mark_id") or str(uuid.uuid4())
                self.mark_event_meta_data.update_data(mark_id, mark_event_meta_data)
                
                # Schedule a task to simulate the mark event after audio finishes playing
                # Critical for hangup messages - must simulate mark event to trigger hangup observer
                if is_final and self.input_handler:
                    asyncio.create_task(self._simulate_mark_event_after_duration(mark_id, audio_duration, message_category))
                    logger.info(f"[SIP-TRUNK OUTPUT] Scheduled mark event simulation for {message_category} after {audio_duration:.3f}s")

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

    async def _send_audio_chunked(self, audio_data, audio_format):
        """Send audio data in smaller chunks to prevent Asterisk from timing out."""
        chunk_size = self._get_chunk_size(audio_format)
        total_bytes = len(audio_data)
        
        # Only chunk if the data is larger than chunk_size
        if total_bytes <= chunk_size:
            await self.websocket.send_bytes(audio_data)
            return total_bytes
        
        # Send in chunks with small delays to keep websocket responsive
        bytes_sent = 0
        offset = 0
        
        while offset < total_bytes:
            chunk = audio_data[offset:offset + chunk_size]
            if chunk:
                await self.websocket.send_bytes(chunk)
                bytes_sent += len(chunk)
                
                # Small delay between chunks to keep websocket responsive
                if offset + chunk_size < total_bytes:
                    await asyncio.sleep(0.001)  # 1ms delay between chunks
            
            offset += chunk_size
        
        return bytes_sent

    async def _simulate_mark_event_after_duration(self, mark_id, duration, message_category):
        """Simulate a mark event after the audio finishes playing (Asterisk doesn't send mark events back)."""
        try:
            await asyncio.sleep(duration)
            if self.input_handler and message_category:
                logger.info(f"[SIP-TRUNK OUTPUT] Simulating mark event after {duration:.3f}s for category: {message_category}")
                mark_packet = {
                    "name": mark_id,
                    "type": message_category
                }
                self.input_handler.process_mark_message(mark_packet)
        except Exception as e:
            logger.error(f"Error simulating mark event: {e}")

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
