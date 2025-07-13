from bolna.input_handlers.telephony import TelephonyInputHandler
from dotenv import load_dotenv
from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)
load_dotenv()


class TwilioInputHandler(TelephonyInputHandler):
    def __init__(self, queues, websocket=None, input_types=None, mark_event_meta_data=None, turn_based_conversation=False,
                 is_welcome_message_played=False, observable_variables=None, websocket_ready_event=None):
        super().__init__(queues, websocket, input_types, mark_event_meta_data, turn_based_conversation,
                         is_welcome_message_played=is_welcome_message_played, observable_variables=observable_variables,
                         websocket_ready_event=websocket_ready_event)
        self.io_provider = 'twilio'

    async def call_start(self, packet):
        start = packet['start']
        self.call_sid = start['callSid']
        self.stream_sid = start['streamSid']
        if self.websocket_ready_event:
            self.websocket_ready_event.set()
            logger.info(f"Websocket ready event set for Twilio with stream_sid: {self.stream_sid}")

    def get_mark_event_meta_data_obj(self, packet):
        mark_id = packet["mark"]["name"]
        return self.mark_event_meta_data.fetch_data(mark_id)
