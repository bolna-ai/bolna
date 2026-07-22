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


class TelephonyOutputHandler(DefaultOutputHandler):
    def __init__(self, io_provider, websocket=None, mark_event_meta_data=None, log_dir_name=None):
        super().__init__(io_provider, websocket, log_dir_name, mark_event_meta_data=mark_event_meta_data)
        self.mark_event_meta_data = mark_event_meta_data

        self.stream_sid = None
        self.current_request_id = None
        self.rejected_request_ids = set()

        # Pace media sends to <= this multiple of real-time so the provider buffer can't fill
        # far ahead; otherwise a barge-in's clearAudio discards seconds of already-sent-but-
        # unplayed audio (see 87da790e). Disabled by default; subclasses opt in (VoBiz does).
        self.max_send_rate_factor = 0.0
        self._rl_first_send_ts = 0.0
        self._rl_bytes_sent = 0
        self._rl_last_send_ts = 0.0

    async def _rate_limit_send(self, audio_bytes, audio_format):
        """Sleep so cumulative send rate stays <= max_send_rate_factor x real-time.
        Re-anchors after a >1s gap so each response burst is paced from its own start."""
        if self.max_send_rate_factor <= 0 or not audio_bytes:
            return
        bytes_per_sec = 8000 if audio_format == "mulaw" else 16000
        now = time.monotonic()
        if self._rl_first_send_ts == 0.0 or (now - self._rl_last_send_ts) > 1.0:
            self._rl_first_send_ts = now
            self._rl_bytes_sent = 0
        self._rl_last_send_ts = now
        self._rl_bytes_sent += audio_bytes
        min_elapsed = self._rl_bytes_sent / (self.max_send_rate_factor * bytes_per_sec)
        elapsed = now - self._rl_first_send_ts
        if elapsed < min_elapsed:
            await asyncio.sleep(min_elapsed - elapsed)

    async def handle_interruption(self):
        pass

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
            audio_chunk = ws_data_packet.get("data")
            meta_info = ws_data_packet.get("meta_info")
            if self.stream_sid is None:
                self.stream_sid = meta_info.get("stream_sid", None)

            if audio_chunk is None:
                logger.info("No audio data in packet, skipping send")
                return

            try:
                if len(audio_chunk) == 1:
                    audio_chunk += b"\x00"

                if audio_chunk and self.stream_sid and len(audio_chunk) != 1:
                    if audio_chunk != b"\x00\x00":
                        audio_format = meta_info.get("format", "wav")

                        # sending of pre-mark message
                        pre_mark_event_meta_data = {
                            "type": "pre_mark_message",
                            "sequence_id": meta_info.get("sequence_id"),
                            "turn_id": meta_info.get("turn_id"),
                            "response_uid": meta_info.get("response_uid"),
                            "response_group_uid": meta_info.get("response_group_uid"),
                        }
                        mark_id = str(uuid.uuid4())
                        self.mark_event_meta_data.update_data(mark_id, pre_mark_event_meta_data)
                        if (
                            meta_info.get("message_category") == "agent_welcome_message"
                            and self.mark_event_meta_data.welcome_pre_mark_id is None
                        ):
                            self.mark_event_meta_data.welcome_pre_mark_id = mark_id
                        logger.info(
                            "BOLNA_TRACE_TEL send_pre_mark mark_id=%s seq=%s turn=%s response_uid=%s group_uid=%s category=%s",
                            mark_id,
                            meta_info.get("sequence_id"),
                            meta_info.get("turn_id"),
                            meta_info.get("response_uid"),
                            meta_info.get("response_group_uid"),
                            meta_info.get("message_category", ""),
                        )
                        mark_message = await self.form_mark_message(mark_id)
                        await self.websocket.send_text(json.dumps(mark_message))

                        # sending of audio chunk
                        if (
                            audio_format == "pcm"
                            and meta_info.get("message_category", "") == "agent_welcome_message"
                            and self.io_provider in ("plivo", "vobiz")
                            and meta_info["cached"] is True
                        ):
                            audio_format = "wav"
                        media_message = await self.form_media_message(audio_chunk, audio_format)
                        await self.websocket.send_text(json.dumps(media_message))
                        if (
                            meta_info.get("message_category", "") == "agent_welcome_message"
                            and not self.welcome_message_sent_ts
                        ):
                            self.welcome_message_sent_ts = time.time() * 1000
                        logger.info(f"Sending media event - {meta_info.get('mark_id')}")
                        await self._rate_limit_send(len(audio_chunk), meta_info.get("format", "mulaw"))

                    mark_event_meta_data = {
                        "text_synthesized": ""
                        if meta_info["sequence_id"] == -1
                        else meta_info.get("text_synthesized", ""),
                        "type": meta_info.get("message_category", ""),
                        "is_first_chunk": meta_info.get("is_first_chunk", False),
                        "is_final_chunk": meta_info.get("end_of_llm_stream", False)
                        and meta_info.get("end_of_synthesizer_stream", False),
                        "sequence_id": meta_info["sequence_id"],
                        "turn_id": meta_info.get("turn_id"),
                        "response_uid": meta_info.get("response_uid"),
                        "response_group_uid": meta_info.get("response_group_uid"),
                        "duration": len(audio_chunk) / 8000
                        if meta_info.get("format", "mulaw") == "mulaw"
                        else len(audio_chunk) / 16000,
                        "sent_ts": time.time(),  # Track when audio was actually sent to telephony provider
                    }
                    mark_id = (
                        meta_info.get("mark_id")
                        if (meta_info.get("mark_id") and meta_info.get("mark_id") != "")
                        else str(uuid.uuid4())
                    )
                    # sending of post-mark message
                    self.mark_event_meta_data.update_data(mark_id, mark_event_meta_data)
                    logger.info(
                        "BOLNA_TRACE_TEL send_post_mark mark_id=%s seq=%s turn=%s response_uid=%s group_uid=%s final=%s category=%s text_len=%s",
                        mark_id,
                        meta_info.get("sequence_id"),
                        meta_info.get("turn_id"),
                        meta_info.get("response_uid"),
                        meta_info.get("response_group_uid"),
                        mark_event_meta_data.get("is_final_chunk"),
                        meta_info.get("message_category", ""),
                        len(mark_event_meta_data.get("text_synthesized", "") or ""),
                    )
                    mark_message = await self.form_mark_message(mark_id)
                    await self.websocket.send_text(json.dumps(mark_message))
                else:
                    logger.info("Not sending")
            except Exception as e:
                self._closed = True  # Prevent further send attempts
                logger.debug(f"WebSocket send failed (client disconnected): {e}")

        except Exception as e:
            self._closed = True
            logger.debug(f"WebSocket handling failed (client disconnected): {e}")
