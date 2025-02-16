from bolna.input_handlers.telephony import TelephonyInputHandler
from dotenv import load_dotenv
from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)
load_dotenv()


class ExotelInputHandler(TelephonyInputHandler):
    def __init__(self, queues, websocket=None, input_types=None, mark_event_meta_data=None, turn_based_conversation=False,
                 is_welcome_message_played=False, observable_variables=None):
        super().__init__(queues, websocket, input_types, mark_event_meta_data, turn_based_conversation,
                         is_welcome_message_played=is_welcome_message_played, observable_variables=observable_variables)
        self.io_provider = 'exotel'

    async def call_start(self, packet):
        start = packet['start']
        self.call_sid = start['call_sid']
        self.stream_sid = start['stream_sid']

    def get_mark_event_meta_data_obj(self, packet):
        pass