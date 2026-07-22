import base64
import json
import os
from dotenv import load_dotenv
from bolna.output_handlers.telephony import TelephonyOutputHandler
from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)
load_dotenv()


class VobizOutputHandler(TelephonyOutputHandler):
    def __init__(self, websocket=None, mark_event_meta_data=None, log_dir_name=None):
        io_provider = "vobiz"

        super().__init__(io_provider, websocket, mark_event_meta_data, log_dir_name)
        self.is_chunking_supported = True

        # VoBiz accepts far-ahead audio into its buffer that a barge-in then discards via
        # clearAudio (see 87da790e), so pace sends to <= this x real-time. 0 disables.
        self.max_send_rate_factor = float(os.getenv("VOBIZ_MAX_SEND_RATE_FACTOR", "1.5") or 0)

    async def handle_interruption(self):
        if self._closed:
            return
        try:
            logger.info("interrupting because user spoke in between")
            message_clear = {
                "event": "clearAudio",
                "streamId": self.stream_sid,
            }
            await self.websocket.send_text(json.dumps(message_clear))
            self.mark_event_meta_data.clear_data()
        except Exception as e:
            logger.info(f"WebSocket closed during interruption: {e}")
            self._closed = True

    async def form_media_message(self, audio_data, audio_format="audio/x-mulaw"):
        base64_audio = base64.b64encode(audio_data).decode("utf-8")
        message = {
            "event": "playAudio",
            "media": {
                "payload": base64_audio,
                "sampleRate": "8000",
                "contentType": "wav" if audio_format == "wav" else "audio/x-mulaw",
            },
        }

        return message

    async def form_mark_message(self, mark_id):
        mark_message = {"event": "checkpoint", "streamId": self.stream_sid, "name": mark_id}

        return mark_message
