"""
Asterisk WebSocket (chan_websocket) output handler for sip-trunk provider.
Send all media between START_MEDIA_BUFFERING and STOP_MEDIA_BUFFERING; Asterisk re-frames and re-times.
On MEDIA_XOFF queue locally; on MEDIA_XON drain to Asterisk. FLUSH_MEDIA on interrupt.
Ref: https://docs.asterisk.org/Configuration/Channel-Drivers/WebSocket/
"""
import asyncio
import time
import uuid
import traceback
import audioop
from bolna.output_handlers.telephony import TelephonyOutputHandler
from bolna.helpers.logger_config import configure_logger
from dotenv import load_dotenv

logger = configure_logger(__name__)
load_dotenv()

ASTERISK_ULAW_OPTIMAL_FRAME_SIZE = 160
# Asterisk max WebSocket message size (docs: sending > 65500 closes the connection)
MAX_WS_MESSAGE_BYTES = 65500
# Preferred chunk size when sending binary to Asterisk. res_http_websocket.c allows only 10
# short-read retries per frame; large frames (e.g. 65KB) can cause "Websocket seems unresponsive,
# disconnecting" (ws_safe_read). Use smaller frames so Asterisk receives each in one or few reads.
PREFERRED_WS_SEND_CHUNK_BYTES = 8192
PLAYBACK_DONE_BUFFER_S = 0.5


class SipTrunkOutputHandler(TelephonyOutputHandler):
    """
    Sends ulaw as BINARY between START_MEDIA_BUFFERING and STOP_MEDIA_BUFFERING.
    When Asterisk sends MEDIA_XOFF (queue full), we queue audio locally; on MEDIA_XON we drain to Asterisk.
    On user interrupt we send FLUSH_MEDIA and clear the local queue.
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
        self._optimal_frame_size = ASTERISK_ULAW_OPTIMAL_FRAME_SIZE
        self.input_handler = input_handler
        self.queue_full = False
        if input_handler:
            input_handler.output_handler_ref = self

        self._buffering_active = False
        self._response_audio_duration = 0.0
        self._playback_done_task = None
        # Local queue when Asterisk sends MEDIA_XOFF; drained on MEDIA_XON
        self._local_audio_queue = []
        # If is_final arrived while we had queued audio, send STOP after drain
        self._pending_stop_after_drain = False
        self._pending_stop_duration = 0.0
        self._pending_stop_category = "agent_response"

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

    async def handle_interruption(self):
        """FLUSH_MEDIA, clear local queue and state, mark playback not active."""
        logger.info("sip-trunk: handling interruption (FLUSH_MEDIA)")
        try:
            if self._playback_done_task:
                self._playback_done_task.cancel()
                self._playback_done_task = None
            self._buffering_active = False
            self._response_audio_duration = 0.0
            self._local_audio_queue.clear()
            self._pending_stop_after_drain = False
            await self._send_control("FLUSH_MEDIA")
            if self.mark_event_meta_data:
                self.mark_event_meta_data.clear_data()
            if self.input_handler:
                self.input_handler.update_is_audio_being_played(False)
        except Exception as e:
            logger.error(f"sip-trunk handle_interruption: {e}")

    async def drain_local_queue(self):
        """Send any audio queued during MEDIA_XOFF to Asterisk (called when MEDIA_XON is received)."""
        while self._local_audio_queue and not self.queue_full:
            chunk = self._local_audio_queue.pop(0)
            if not chunk:
                continue
            await self._send_binary(chunk)
        if not self._local_audio_queue and self._pending_stop_after_drain:
            self._pending_stop_after_drain = False
            if self._buffering_active:
                await self._send_control("STOP_MEDIA_BUFFERING")
                self._buffering_active = False
            await self._send_control("REPORT_QUEUE_DRAINED")
            duration = self._pending_stop_duration
            category = self._pending_stop_category
            if self._playback_done_task:
                self._playback_done_task.cancel()
            self._playback_done_task = asyncio.create_task(
                self._playback_done_fallback(duration, category)
            )

    async def _send_control(self, command, params=None):
        """Send one control command as TEXT (plain text, Asterisk-compatible)."""
        try:
            msg = command if not params else f"{command} {' '.join(str(v) for v in params.values())}"
            await self.websocket.send_text(msg)
            logger.debug(f"sip-trunk sent: {command}")
        except Exception as e:
            logger.error(f"sip-trunk send_control {command}: {e}")
            traceback.print_exc()

    async def _send_binary(self, data: bytes):
        """Send raw ulaw to Asterisk in small chunks to avoid Asterisk ws_safe_read 'unresponsive' disconnect."""
        n = len(data)
        if n <= PREFERRED_WS_SEND_CHUNK_BYTES:
            await self.websocket.send_bytes(data)
            return
        offset = 0
        while offset < n:
            chunk_size = min(PREFERRED_WS_SEND_CHUNK_BYTES, n - offset, MAX_WS_MESSAGE_BYTES)
            chunk = data[offset : offset + chunk_size]
            if chunk:
                await self.websocket.send_bytes(chunk)
            offset += chunk_size

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

    async def handle(self, ws_data_packet):
        """
        One START_MEDIA_BUFFERING per response, then send all audio (or queue if MEDIA_XOFF), then STOP_MEDIA_BUFFERING.
        Asterisk re-frames and re-times; we do not pace packets.
        """
        try:
            audio_chunk = ws_data_packet.get("data")
            meta_info = ws_data_packet.get("meta_info") or {}
            if self.stream_sid is None:
                self.stream_sid = meta_info.get("stream_sid")

            is_final = (
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

                # Per Asterisk docs: START_MEDIA_BUFFERING before media so driver can buffer and re-frame
                if not self._buffering_active and len(audio_chunk) > self._optimal_frame_size:
                    await self._send_control("START_MEDIA_BUFFERING")
                    self._buffering_active = True
                    self._response_audio_duration = 0.0

                if self.queue_full:
                    self._local_audio_queue.append(audio_chunk)
                else:
                    await self._send_binary(audio_chunk)

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
                    self._response_audio_duration = 0.0

                    if self._local_audio_queue:
                        self._pending_stop_after_drain = True
                        self._pending_stop_duration = total_duration
                        self._pending_stop_category = message_category
                        logger.debug("sip-trunk: final chunk queued (XOFF); will send STOP after drain")
                    else:
                        if self._buffering_active:
                            await self._send_control("STOP_MEDIA_BUFFERING")
                            self._buffering_active = False
                        await self._send_control("REPORT_QUEUE_DRAINED")
                        if self._playback_done_task:
                            self._playback_done_task.cancel()
                        self._playback_done_task = asyncio.create_task(
                            self._playback_done_fallback(total_duration, message_category)
                        )
                        logger.debug(f"sip-trunk: response done, fallback in {total_duration + PLAYBACK_DONE_BUFFER_S:.1f}s")

        except Exception as e:
            logger.error(f"sip-trunk output error: {e}")
            traceback.print_exc()

    async def _playback_done_fallback(self, duration: float, message_category: str):
        """If QUEUE_DRAINED never arrives, mark playback done after duration + buffer and process marks."""
        try:
            await asyncio.sleep(duration + PLAYBACK_DONE_BUFFER_S)
        except asyncio.CancelledError:
            return
        self._playback_done_task = None
        if not self.input_handler or not self.input_handler.mark_event_meta_data:
            return
        remaining = list(self.input_handler.mark_event_meta_data.mark_event_meta_data.keys())
        if not remaining:
            return
        logger.info(f"sip-trunk: playback-done fallback ({duration:.2f}s), processing {len(remaining)} mark(s)")
        self.input_handler.update_is_audio_being_played(False)
        for mid in remaining:
            md = self.input_handler.mark_event_meta_data.mark_event_meta_data.get(mid, {})
            self.input_handler.process_mark_message({"name": mid, "type": md.get("type", message_category)})

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
