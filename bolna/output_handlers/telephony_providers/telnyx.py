import base64
import json
import audioop
from dotenv import load_dotenv
from bolna.helpers.logger_config import configure_logger
from bolna.output_handlers.telephony import TelephonyOutputHandler

logger = configure_logger(__name__)
load_dotenv()


class TelnyxOutputHandler(TelephonyOutputHandler):
    """Unlike Twilio/Plivo/Vobiz, Telnyx's outbound media/mark/clear frames carry no
    session id (https://developers.telnyx.com/api-reference/websockets/stream-call-media-over-websocket)
    - the event name and payload alone are enough, the stream is identified by the
    websocket connection itself."""

    def __init__(self, websocket=None, mark_event_meta_data=None, log_dir_name=None):
        io_provider = "telnyx"

        super().__init__(io_provider, websocket, mark_event_meta_data, log_dir_name)
        self.is_chunking_supported = True

    async def handle_interruption(self):
        if self._closed:
            return
        try:
            logger.info("interrupting because user spoke in between")
            await self.websocket.send_text(json.dumps({"event": "clear"}))
            self.mark_event_meta_data.clear_data()
        except Exception as e:
            logger.info(f"WebSocket closed during interruption: {e}")
            self._closed = True

    async def form_media_message(self, audio_data, audio_format="wav"):
        if audio_format != "mulaw":
            audio_data = audioop.lin2ulaw(audio_data, 2)
        base64_audio = base64.b64encode(audio_data).decode("utf-8")
        message = {"event": "media", "media": {"payload": base64_audio}}

        return message

    async def form_mark_message(self, mark_id):
        mark_message = {"event": "mark", "mark": {"name": mark_id}}

        return mark_message
