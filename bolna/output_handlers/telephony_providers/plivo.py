import base64
import json
from dotenv import load_dotenv
from bolna.output_handlers.telephony import TelephonyOutputHandler
from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)
load_dotenv()


class PlivoOutputHandler(TelephonyOutputHandler):
    def __init__(self, websocket=None, mark_event_meta_data=None, log_dir_name=None):
        io_provider = 'plivo'

        super().__init__(io_provider, websocket, mark_event_meta_data, log_dir_name)
        self.is_chunking_supported = True

    async def handle_interruption(self):
        logger.info("interrupting because user spoke in between")
        message_clear = {
            "event": "clearAudio",
            "streamId": self.stream_sid,
        }
        await self.websocket.send_text(json.dumps(message_clear))
        self.mark_event_meta_data.clear_data()

    async def form_media_message(self, audio_data, audio_format='audio/x-mulaw'):
        base64_audio = base64.b64encode(audio_data).decode("utf-8")
        message = {
            'event': 'playAudio',
            'media': {
                'payload': base64_audio,
                'sampleRate': '8000',
                'contentType': 'wav' if audio_format == 'wav' else 'audio/x-mulaw'
            }
        }

        return message

    async def form_mark_message(self, mark_id):
        mark_message = {
            "event": "checkpoint",
            "streamId": self.stream_sid,
            "name": mark_id
        }

        return mark_message
