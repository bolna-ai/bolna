import asyncio
import base64
import json
import os
import audioop
import time
import uuid
import traceback
from dotenv import load_dotenv
from .default import DefaultOutputHandler
from bolna.helpers.logger_config import configure_logger
logger = configure_logger(__name__)
load_dotenv()

TELEPHONY_MAX_SEND_RATE_FACTOR = float(os.environ.get("TELEPHONY_MAX_SEND_RATE_FACTOR", "1.2"))


class TelephonyOutputHandler(DefaultOutputHandler):
    def __init__(self, io_provider, websocket=None, mark_event_meta_data=None, log_dir_name=None):
        super().__init__(io_provider, websocket, log_dir_name, mark_event_meta_data=mark_event_meta_data)
        self.mark_event_meta_data = mark_event_meta_data

        self.stream_sid = None
        self.current_request_id = None
        self.rejected_request_ids = set()

        # Rate-limit state (activated per-provider in handle())
        self._rl_first_send = 0.0
        self._rl_bytes_sent = 0
        self._rl_sequence_id = None

    async def handle_interruption(self):
        self._rl_first_send = 0.0
        self._rl_bytes_sent = 0
        self._rl_sequence_id = None

    async def _rate_limit(self, audio_bytes, audio_format):
        """Sleep if needed to stay ≤ TELEPHONY_MAX_SEND_RATE_FACTOR × real-time."""
        if TELEPHONY_MAX_SEND_RATE_FACTOR <= 0 or self._rl_first_send == 0.0:
            return
        bytes_per_second = 8000 if audio_format in ('mulaw', 'ulaw') else 16000
        self._rl_bytes_sent += audio_bytes
        min_elapsed = self._rl_bytes_sent / (TELEPHONY_MAX_SEND_RATE_FACTOR * bytes_per_second)
        elapsed = time.monotonic() - self._rl_first_send
        if elapsed < min_elapsed:
            await asyncio.sleep(min_elapsed - elapsed)

    async def form_media_message(self, audio_data, audio_format):
        pass

    async def form_mark_message(self, mark_id):
        pass

    async def set_stream_sid(self, stream_id):
        self.stream_sid = stream_id

    async def handle(self, ws_data_packet):
        if self._closed:
            return
        try:
            audio_chunk = ws_data_packet.get('data')
            meta_info = ws_data_packet.get('meta_info')
            if self.stream_sid is None:
                self.stream_sid = meta_info.get('stream_sid', None)

            if audio_chunk is None:
                logger.info("No audio data in packet, skipping send")
                return

            try:
                if len(audio_chunk) == 1:
                    audio_chunk += b'\x00'

                if audio_chunk and self.stream_sid and len(audio_chunk) != 1:
                    if audio_chunk != b'\x00\x00':
                        audio_format = meta_info.get("format", "wav")

                        # Rate-limit: detect new sequence → reset counters
                        seq_id = meta_info.get("sequence_id")
                        if seq_id is not None and seq_id != self._rl_sequence_id:
                            self._rl_sequence_id = seq_id
                            self._rl_first_send = 0.0
                            self._rl_bytes_sent = 0

                        # sending of pre-mark message
                        pre_mark_event_meta_data = {
                            "type": "pre_mark_message",
                        }
                        mark_id = str(uuid.uuid4())
                        self.mark_event_meta_data.update_data(mark_id, pre_mark_event_meta_data)
                        mark_message = await self.form_mark_message(mark_id)
                        await self.websocket.send_text(json.dumps(mark_message))

                        # sending of audio chunk
                        if audio_format == 'pcm' and meta_info.get('message_category', '') == 'agent_welcome_message' and self.io_provider in ('plivo', 'vobiz') and meta_info['cached'] is True:
                            audio_format = 'wav'
                        media_message = await self.form_media_message(audio_chunk, audio_format)
                        await self.websocket.send_text(json.dumps(media_message))
                        if self._rl_first_send == 0.0:
                            self._rl_first_send = time.monotonic()
                        if meta_info.get('message_category', '') == 'agent_welcome_message' and not self.welcome_message_sent_ts:
                            self.welcome_message_sent_ts = time.time() * 1000
                        logger.info(f"Sending media event - {meta_info.get('mark_id')}")

                    # sending of post-mark message
                    mark_event_meta_data = {
                        "text_synthesized": "" if meta_info["sequence_id"] == -1 else meta_info.get("text_synthesized", ""),
                        "type": meta_info.get('message_category', ''),
                        "is_first_chunk": meta_info.get("is_first_chunk", False),
                        "is_final_chunk": meta_info.get("end_of_llm_stream", False) and meta_info.get("end_of_synthesizer_stream", False),
                        "sequence_id": meta_info["sequence_id"],
                        "duration": len(audio_chunk) / 8000 if meta_info.get('format', 'mulaw') == 'mulaw' else len(audio_chunk) / 16000,
                        "sent_ts": time.time()  # Track when audio was actually sent to telephony provider
                    }
                    mark_id = meta_info.get("mark_id") if (meta_info.get("mark_id") and meta_info.get("mark_id") != "") else str(uuid.uuid4())

                    self.mark_event_meta_data.update_data(mark_id, mark_event_meta_data)
                    mark_message = await self.form_mark_message(mark_id)
                    await self.websocket.send_text(json.dumps(mark_message))

                    # Rate-limit: throttle audio send rate for Plivo
                    if self.io_provider == 'plivo' and audio_chunk != b'\x00\x00':
                        await self._rate_limit(len(audio_chunk), meta_info.get('format', 'mulaw'))
                else:
                    logger.info("Not sending")
            except Exception as e:
                self._closed = True  # Prevent further send attempts
                logger.debug(f'WebSocket send failed (client disconnected): {e}')

        except Exception as e:
            self._closed = True
            logger.debug(f'WebSocket handling failed (client disconnected): {e}')
