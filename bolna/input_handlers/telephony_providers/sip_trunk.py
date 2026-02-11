"""
Asterisk WebSocket (chan_websocket) input handler for sip-trunk provider.
Audio is always ulaw over BINARY frames; control events over TEXT frames (JSON or plain text).
Ref: https://docs.asterisk.org/Configuration/Channel-Drivers/WebSocket/#control-commands-and-events
"""
import asyncio
import json
import traceback
from bolna.input_handlers.telephony import TelephonyInputHandler
from bolna.helpers.utils import create_ws_data_packet
from bolna.helpers.logger_config import configure_logger
from starlette.websockets import WebSocketDisconnect
from dotenv import load_dotenv

logger = configure_logger(__name__)
load_dotenv()

# Asterisk ulaw: 160 bytes per 20ms frame
ASTERISK_ULAW_OPTIMAL_FRAME_SIZE = 160


def _parse_asterisk_control_message(text: str) -> dict:
    """Parse Asterisk control: JSON or plain 'KEY value' / 'KEY:value' lines.

    Returns dict with normalized 'event' key (uppercase, spaces replaced with underscores).
    """
    text = (text or "").strip()
    if not text:
        return {}
    result = {}
    # Try JSON first (Asterisk 20.18+, 22.8+, 23.2+)
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            result = obj
    except (json.JSONDecodeError, TypeError):
        # Plain text: "EVENT key:value key2:value2" or "KEY value"
        for part in text.split():
            if ":" in part:
                k, v = part.split(":", 1)
                result[k.strip().lower()] = v.strip()
        if " " in text:
            first = text.split()[0]
            if first not in result:
                result["event"] = first

    # Normalize event key
    event = (result.get("event") or result.get("command") or "").upper().replace(" ", "_")
    if event:
        result["event"] = event
    return result


class SipTrunkInputHandler(TelephonyInputHandler):
    """Handles Asterisk WebSocket: BINARY = ulaw audio, TEXT = control events."""

    def __init__(
        self,
        queues,
        websocket=None,
        input_types=None,
        mark_event_meta_data=None,
        turn_based_conversation=False,
        is_welcome_message_played=False,
        observable_variables=None,
        asterisk_media_start=None,
        agent_config=None,
        ws_context_data=None,
    ):
        super().__init__(
            queues,
            websocket,
            input_types,
            mark_event_meta_data,
            turn_based_conversation,
            is_welcome_message_played=is_welcome_message_played,
            observable_variables=observable_variables or {},
        )
        self.io_provider = "sip-trunk"
        self.agent_config = agent_config or {}
        self._optimal_frame_size = ASTERISK_ULAW_OPTIMAL_FRAME_SIZE
        self._media_xoff = False
        self.media_started = False
        self.channel_id = None
        self.connection_id = None
        self.ptime = 20

        # Pending mark info set by the output handler so that QUEUE_DRAINED
        # (authoritative Asterisk signal) can trigger proper mark processing
        # — equivalent to Twilio's mark event for is_audio_being_played,
        # welcome-message completion, and hangup detection.
        self._pending_queue_drain_mark_id = None
        self._pending_queue_drain_category = None

        # Resolve format from agent_config (must be ulaw for sip-trunk)
        input_config = self._get_input_config()
        self._expected_format = (input_config.get("audio_format") or input_config.get("format") or "ulaw").lower()
        if self._expected_format not in ("ulaw", "mulaw"):
            logger.warning(
                f"sip-trunk input expects ulaw from agent_config; got {self._expected_format}, using ulaw"
            )
            self._expected_format = "ulaw"

        # Initialize from context data if provided (when server intercepts MEDIA_START)
        # Support both ws_context_data (new) and asterisk_media_start (backward compat)
        media_start_data = None
        if ws_context_data and 'media_start_data' in ws_context_data:
            media_start_data = ws_context_data['media_start_data']
        elif asterisk_media_start:
            media_start_data = asterisk_media_start
        
        if media_start_data:
            self._initialize_from_media_start(media_start_data)

    def _get_input_config(self):
        try:
            tasks = self.agent_config.get("tasks") or []
            if tasks and isinstance(tasks[0], dict):
                return tasks[0].get("tools_config", {}).get("input") or {}
        except Exception:
            pass
        return {}

    async def disconnect_stream(self):
        """Send HANGUP command to Asterisk when disconnecting.

        Per Asterisk WebSocket docs, HANGUP causes the channel to be hung up
        and the WebSocket to be closed immediately.
        """
        try:
            if self.websocket and self.channel_id:
                # Check if WebSocket is still open before attempting to send
                # WebSocket state: 1 = OPEN, 2 = CLOSING, 3 = CLOSED
                if hasattr(self.websocket, 'client_state') and self.websocket.client_state.value == 1:
                    # According to Asterisk WebSocket documentation, control commands are sent as plain text
                    await self.websocket.send_text("HANGUP")
                    logger.info(f"Sent HANGUP command for channel {self.channel_id}")
                else:
                    logger.info(f"WebSocket already closed/closing, skipping HANGUP command for channel {self.channel_id}")
            else:
                logger.info(f"Cannot send HANGUP - websocket or channel_id is None")
        except Exception as e:
            logger.error(f"Error sending HANGUP command: {e}")

    async def stop_handler(self):
        """Override base stop_handler for faster SIP trunk disconnect.

        Per Asterisk docs, the HANGUP command causes the WebSocket channel to
        be hung up and the WebSocket to be closed immediately.  The base class
        sleeps 2 s after firing disconnect_stream, but Asterisk processes
        HANGUP within milliseconds so we only need a brief pause.
        """
        logger.info(f"stopping handler for channel {self.channel_id}")
        self.running = False
        await self.disconnect_stream()  # Send HANGUP (awaited, not fire-and-forget)
        # Asterisk closes the WebSocket almost immediately after HANGUP;
        # a brief wait is sufficient for the close to propagate.
        await asyncio.sleep(0.5)
        try:
            await self.websocket.close()
            logger.info("WebSocket connection closed")
        except Exception as e:
            logger.info(f"Error closing WebSocket: {e}")

    def _initialize_from_media_start(self, media_start_data):
        """Initialize channel info from pre-parsed MEDIA_START data"""
        self.channel_id = media_start_data.get('channel_id')
        self.connection_id = media_start_data.get('connection_id', self.channel_id)
        self.format = media_start_data.get('format')
        self.ptime = int(media_start_data.get('ptime', 20))
        
        # Use connection_id as stream_sid for compatibility
        self.stream_sid = self.connection_id or self.channel_id
        if self.channel_id and "_" in self.channel_id:
            self.call_sid = self.channel_id.split("_")[0]
        else:
            self.call_sid = self.channel_id
        
        opt = media_start_data.get("optimal_frame_size")
        if opt is not None:
            try:
                self._optimal_frame_size = int(opt)
            except (TypeError, ValueError):
                pass
        
        self.media_started = True
        
        logger.info(f"Initialized from context - Channel: {self.channel_id}, "
                   f"Connection: {self.connection_id}, Format: {self.format}, Ptime: {self.ptime}ms")

    async def call_start(self, packet):
        """Handle MEDIA_START: store channel_id, optimal_frame_size, format."""
        self._initialize_from_media_start(packet)

    async def _listen(self):
        """Receive both TEXT (control) and BINARY (ulaw) frames. Asterisk: TEXT=control, BINARY=media.
        Audio from Asterisk is 160 bytes per 20ms (optimal_frame_size). We forward to the transcriber
        in small batches so the full phrase is delivered before endpointing commits (avoids truncated finals).
        """
        buffer = []
        while self.running:
            try:
                # Receive message from WebSocket
                # Asterisk sends BINARY for media, TEXT for control
                message = await self.websocket.receive()
                
                # Check for disconnect message
                if message.get('type') == 'websocket.disconnect':
                    logger.info(f"WebSocket disconnect message received for channel {self.channel_id}")
                    break
                
                # Handle BINARY frames (media data)
                if 'bytes' in message:
                    media_audio = message['bytes']
                    if not media_audio:
                        continue
                    
                    meta_info = {
                        "io": self.io_provider,
                        "call_sid": self.call_sid,
                        "stream_sid": self.stream_sid,
                        "sequence": (self.input_types or {}).get("audio", 0),
                        "format": self._expected_format,
                    }
                    buffer.append(media_audio)
                    self.message_count += 1
                    # Forward audio to transcriber frequently so the full phrase is received before
                    # Deepgram endpointing commits. Asterisk sends 160-byte frames every 20ms;
                    # batching too much (e.g. 100ms) can leave the tail of the utterance in our
                    # buffer when the final is emitted, causing truncated transcripts.
                    # Use 2 frames (~40ms) as a balance between latency and queue load.
                    chunks_to_accumulate = max(1, 40 // self.ptime) if self.ptime else 2
                    
                    if self.message_count >= chunks_to_accumulate:
                        merged_audio = b''.join(buffer)
                        buffer = []
                        await self.ingest_audio(merged_audio, meta_info)
                        self.message_count = 0
                
                # Handle TEXT frames (control events)
                elif 'text' in message:
                    message_text = message['text']
                    await self._handle_control_message(message_text)
                else:
                    logger.debug(f"Received unknown message type: {message}")
            except WebSocketDisconnect as e:
                if getattr(e, "code", None) in (1000, 1001, 1006):
                    logger.info(f"WebSocket disconnected normally: code={getattr(e, 'code', None)}")
                else:
                    logger.error(
                        f"WebSocket disconnected unexpectedly: code={getattr(e, 'code', None)}, "
                        f"reason={getattr(e, 'reason', None)}"
                    )
                break
            except RuntimeError as e:
                # Handle "Cannot call receive once a disconnect message has been received"
                if "disconnect message has been received" in str(e):
                    logger.info(f"WebSocket already disconnected for channel {self.channel_id}")
                    break
                else:
                    logger.error(f"Runtime error processing message: {e}")
                    traceback.print_exc()
                    break
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                traceback.print_exc()
                break

        # Flush remaining buffer as EOS
        if buffer:
            merged = b"".join(buffer)
            if merged:
                meta_info = {
                    "io": self.io_provider,
                    "call_sid": self.call_sid,
                    "stream_sid": self.stream_sid,
                    "sequence": (self.input_types or {}).get("audio", 0),
                    "format": self._expected_format,
                }
                await self.ingest_audio(merged, meta_info)
        # Send EOS to transcriber when connection ends
        ws_data_packet = create_ws_data_packet(
            data=None,
            meta_info={
                'io': self.io_provider, 
                'eos': True,
                'sequence': (self.input_types or {}).get("audio", 0)
            }
        )
        self.queues["transcriber"].put_nowait(ws_data_packet)
        logger.info(f"Asterisk WebSocket connection closed for channel {self.channel_id}")

    async def _handle_control_message(self, text: str):
        """Handle Asterisk TEXT control: MEDIA_START, DTMF_END, MEDIA_XOFF, MEDIA_XON, etc."""
        parsed = _parse_asterisk_control_message(text)
        event = parsed.get("event", "")

        if event == "MEDIA_START":
            await self.call_start(parsed)
            return
        if event == "DTMF_END":
            digit = parsed.get("digit", "")
            if digit:
                logger.info(f"Asterisk DTMF_END digit={digit}")
                if self.is_dtmf_active:
                    self.queues["dtmf"].put_nowait(digit)
            return
        if event == "MEDIA_XOFF":
            self._media_xoff = True
            logger.warning(f"MEDIA_XOFF received - Asterisk queue is full for channel {self.channel_id}")
            # Notify output handler if available
            if hasattr(self, 'output_handler_ref'):
                self.output_handler_ref.queue_full = True
            return
        if event == "MEDIA_XON":
            self._media_xoff = False
            logger.info(f"MEDIA_XON received - Asterisk queue has space for channel {self.channel_id}")
            # Notify output handler if available
            if hasattr(self, 'output_handler_ref'):
                self.output_handler_ref.queue_full = False
            return
        if event == "STATUS":
            queue_length = parsed.get('queue_length', 0)
            xon_level = parsed.get('xon_level', 0)
            xoff_level = parsed.get('xoff_level', 0)
            queue_full = parsed.get('queue_full', False)
            bulk_media = parsed.get('bulk_media', False)
            media_paused = parsed.get('media_paused', False)
            logger.info(f"STATUS: Queue={queue_length}, XON={xon_level}, XOFF={xoff_level}, "
                       f"Full={queue_full}, Bulk={bulk_media}, Paused={media_paused}")
            return
        if "MEDIA_BUFFERING_COMPLETED" in event:
            correlation_id = parsed.get('correlation_id')
            logger.info(f"MEDIA_BUFFERING_COMPLETED received - Correlation ID: {correlation_id}")
            return
        if "QUEUE_DRAINED" in event or event == "QUEUE_DRAINED":
            logger.info(f"QUEUE_DRAINED received for channel {self.channel_id} - Asterisk finished playing media")
            # Asterisk has finished playing the queue.  This is the authoritative
            # signal (equivalent to Twilio mark events).  Unlike Twilio/Plivo
            # which send per-mark callbacks, Asterisk only gives a single
            # "everything done" signal.  Process ALL remaining marks so that
            # is_audio_being_played, welcome-message completion, and hangup
            # detection all work correctly.
            self.update_is_audio_being_played(False)

            # Clear pending tracking state
            self._pending_queue_drain_mark_id = None
            self._pending_queue_drain_category = None

            # Process every mark still in mark_event_meta_data.
            # fetch_data() pops each entry, so snapshot keys first.
            all_mark_ids = list(self.mark_event_meta_data.mark_event_meta_data.keys())
            if all_mark_ids:
                logger.info(f"QUEUE_DRAINED: processing {len(all_mark_ids)} remaining mark(s)")
                for mid in all_mark_ids:
                    mark_data = self.mark_event_meta_data.mark_event_meta_data.get(mid, {})
                    cat = mark_data.get("type", "agent_response")
                    mark_packet = {"name": mid, "type": cat}
                    self.process_mark_message(mark_packet)
            return
        # Unknown or empty
        if event or parsed:
            logger.debug(f"Asterisk control: {text} -> event={event} parsed={parsed}")
