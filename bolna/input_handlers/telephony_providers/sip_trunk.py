"""
Asterisk WebSocket (chan_websocket) input handler for sip-trunk provider.
BINARY = ulaw audio; TEXT = control events (JSON or plain text).
Ref: https://docs.asterisk.org/Configuration/Channel-Drivers/WebSocket/
"""
import asyncio
import json
import traceback

from starlette.websockets import WebSocketDisconnect

from bolna.enums import AsteriskEvent, AudioFormat
from bolna.input_handlers.telephony import TelephonyInputHandler
from bolna.helpers.utils import create_ws_data_packet
from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)

# Asterisk ulaw: 160 bytes per 20ms frame
ASTERISK_ULAW_OPTIMAL_FRAME_SIZE = 160
DEFAULT_PTIME_MS = 20


def _parse_asterisk_control_message(text: str) -> dict:
    """Parse Asterisk control message: JSON or plain 'KEY value' / 'KEY:value'.
    Returns dict with normalized 'event' key (uppercase, spaces replaced with underscores).
    """
    text = (text or "").strip()
    if not text:
        return {}

    # Try JSON first
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            event = (obj.get("event") or obj.get("command") or "").upper().replace(" ", "_")
            if event:
                obj["event"] = event
            return obj
    except (json.JSONDecodeError, TypeError):
        pass

    # Plain text format: "EVENT key:value key:value"
    result = {}
    parts = text.split()
    if parts:
        result["event"] = parts[0].upper().replace(" ", "_")
        for part in parts[1:]:
            if ":" in part:
                k, v = part.split(":", 1)
                result[k.strip().lower()] = v.strip()
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
        self.media_started = False
        self.channel_id = None
        self.connection_id = None
        self.ptime = DEFAULT_PTIME_MS
        self._pending_stream_sid = None  # promoted to stream_sid on first audio frame
        self.output_handler_ref = None

        self._expected_format = AudioFormat.ULAW.value

        media_start_data = None
        if ws_context_data and "media_start_data" in ws_context_data:
            media_start_data = ws_context_data["media_start_data"]
        elif asterisk_media_start:
            media_start_data = asterisk_media_start

        if media_start_data:
            self._initialize_from_media_start(media_start_data)

    async def disconnect_stream(self):
        """Send HANGUP to Asterisk so the channel and WebSocket close."""
        try:
            if self.websocket and self.channel_id:
                if hasattr(self.websocket, "client_state") and self.websocket.client_state.value == 1:
                    await self.websocket.send_text("HANGUP")
                    logger.info(f"Sent HANGUP for channel {self.channel_id}")
        except Exception as e:
            logger.error(f"Error sending HANGUP: {e}")

    async def stop_handler(self):
        """Stop and disconnect; Asterisk closes quickly after HANGUP."""
        logger.info(f"Stopping sip-trunk handler for channel {self.channel_id}")
        self.running = False
        await self.disconnect_stream()
        await asyncio.sleep(0.5)
        try:
            await self.websocket.close()
        except Exception as e:
            logger.info(f"Error closing WebSocket: {e}")

    def _initialize_from_media_start(self, media_start_data):
        """Set channel info from MEDIA_START.

        NOTE: stream_sid is intentionally NOT set here. It is held in
        _pending_stream_sid and promoted to stream_sid only when the first
        binary audio frame arrives from Asterisk.

        Why: the channel is auto-answered by Asterisk when the WebSocket
        connects, but it is NOT in any bridge yet. The sip-server still needs
        to call add_channel_to_bridge() after this. __forced_first_message polls
        stream_sid != None before sending welcome audio, so deferring here
        guarantees welcome audio is only sent once the bridge is active (proven
        by Asterisk writing audio frames back to us).
        """
        self.channel_id = media_start_data.get("channel_id")
        self.connection_id = media_start_data.get("connection_id", self.channel_id)
        self.format = media_start_data.get("format")
        self.ptime = int(media_start_data.get("ptime", DEFAULT_PTIME_MS))
        self._pending_stream_sid = self.connection_id or self.channel_id
        self.call_sid = (
            self.channel_id.split("_")[0]
            if (self.channel_id and "_" in self.channel_id)
            else self.channel_id
        )

        opt = media_start_data.get("optimal_frame_size")
        if opt is not None:
            try:
                self._optimal_frame_size = int(opt)
            except (TypeError, ValueError):
                pass
        self.media_started = True
        logger.info(
            f"Initialized from MEDIA_START - channel_id={self.channel_id}, "
            f"format={self.format}, ptime={self.ptime}ms "
            f"(stream_sid deferred until first audio frame from bridge)"
        )

    async def call_start(self, packet):
        """Handle MEDIA_START."""
        self._initialize_from_media_start(packet)

    async def _listen(self):
        """Receive TEXT (control) and BINARY (ulaw audio).
        Forward audio in ~80ms batches for balance of latency and transcript accuracy.
        """
        buffer = []
        chunks_per_batch = max(2, 80 // self.ptime) if self.ptime else 4
        message_count = 0

        while self.running:
            try:
                message = await self.websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break

                if "bytes" in message:
                    media_audio = message["bytes"]
                    if not media_audio:
                        continue

                    # First audio frame proves bridge is active
                    if self.stream_sid is None and self._pending_stream_sid:
                        self.stream_sid = self._pending_stream_sid
                        logger.info(
                            f"First audio frame received — bridge active, stream_sid={self.stream_sid}"
                        )

                    buffer.append(media_audio)
                    message_count += 1
                    if message_count >= chunks_per_batch:
                        merged = b"".join(buffer)
                        buffer.clear()
                        message_count = 0
                        meta_info = {
                            "io": self.io_provider,
                            "call_sid": self.call_sid,
                            "stream_sid": self.stream_sid,
                            "sequence": (self.input_types or {}).get("audio", 0),
                            "format": self._expected_format,
                        }
                        await self.ingest_audio(merged, meta_info)

                elif "text" in message:
                    await self._handle_control_message(message["text"])

            except WebSocketDisconnect as e:
                if getattr(e, "code", None) in (1000, 1001, 1006):
                    logger.info("WebSocket disconnected normally")
                else:
                    logger.error(
                        f"WebSocket disconnected: code={getattr(e, 'code', None)}, "
                        f"reason={getattr(e, 'reason', None)}"
                    )
                break
            except RuntimeError as e:
                if "disconnect message has been received" in str(e):
                    break
                logger.error(f"Runtime error: {e}", exc_info=True)
                break
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in _listen: {e}")
                traceback.print_exc()
                break

        # Flush remaining buffer
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

        ws_data_packet = create_ws_data_packet(
            data=None,
            meta_info={
                "io": self.io_provider,
                "eos": True,
                "sequence": (self.input_types or {}).get("audio", 0),
            },
        )
        self.queues["transcriber"].put_nowait(ws_data_packet)
        logger.info(f"sip-trunk WebSocket closed for channel {self.channel_id}")

    async def _handle_control_message(self, text: str):
        """Handle Asterisk TEXT control events."""
        parsed = _parse_asterisk_control_message(text)
        event = parsed.get("event", "")

        if event == AsteriskEvent.MEDIA_START.value:
            await self.call_start(parsed)
            return

        if event == AsteriskEvent.DTMF_END.value:
            digit = parsed.get("digit", "")
            if digit and self.is_dtmf_active:
                self.queues["dtmf"].put_nowait(digit)
            return

        if event == AsteriskEvent.MEDIA_XOFF.value:
            if self.output_handler_ref:
                self.output_handler_ref.queue_full = True
            logger.debug(f"MEDIA_XOFF for channel {self.channel_id}")
            return

        if event == AsteriskEvent.MEDIA_XON.value:
            if self.output_handler_ref:
                self.output_handler_ref.queue_full = False
                task = asyncio.create_task(self.output_handler_ref.drain_local_queue())
                task.add_done_callback(self._on_drain_done)
            logger.debug(f"MEDIA_XON for channel {self.channel_id}")
            return

        if event == AsteriskEvent.QUEUE_DRAINED.value:
            # QUEUE_DRAINED means Asterisk finished playing all buffered audio.
            # Forward to output handler for precise playback completion tracking.
            if self.output_handler_ref:
                self.output_handler_ref.handle_queue_drained()
            logger.debug(f"QUEUE_DRAINED for channel {self.channel_id}")
            return

        if event in (AsteriskEvent.STATUS.value, AsteriskEvent.MEDIA_BUFFERING_COMPLETED.value):
            logger.debug(f"Asterisk control: {event}")
            return

        if event or parsed:
            logger.debug(f"Asterisk control: {text} -> event={event}")

    @staticmethod
    def _on_drain_done(task):
        exc = task.exception() if not task.cancelled() else None
        if exc is not None:
            logger.exception("sip-trunk drain_local_queue failed: %s", exc)
