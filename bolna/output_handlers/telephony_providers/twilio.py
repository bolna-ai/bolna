import base64
import json
import os
import audioop
from dotenv import load_dotenv
from bolna.helpers.logger_config import configure_logger
from bolna.output_handlers.telephony import TelephonyOutputHandler

logger = configure_logger(__name__)
load_dotenv()


class TwilioOutputHandler(TelephonyOutputHandler):
    def __init__(self, websocket=None, mark_event_meta_data=None, log_dir_name=None):
        io_provider = 'twilio'

        super().__init__(io_provider, websocket, mark_event_meta_data, log_dir_name)
        self.is_chunking_supported = True

    async def handle_interruption(self):
        logger.info("interrupting because user spoke in between")
        message_clear = {
            "event": "clear",
            "streamSid": self.stream_sid,
        }
        await self.websocket.send_text(json.dumps(message_clear))
        self.mark_event_meta_data.clear_data()

    async def form_media_message(self, audio_data, audio_format="wav"):
        if audio_format != "mulaw":
            logger.info(f"Converting to mulaw")
            audio_data = audioop.lin2ulaw(audio_data, 2)
        base64_audio = base64.b64encode(audio_data).decode("utf-8")
        message = {
            'event': 'media',
            'streamSid': self.stream_sid,
            'media': {
                'payload': base64_audio
            }
        }

        return message

    async def form_mark_message(self, mark_id):
        mark_message = {
            "event": "mark",
            "streamSid": self.stream_sid,
            "mark": {
                "name": mark_id
            }
        }

        return mark_message
