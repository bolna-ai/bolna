import os
import plivo as plivosdk
from dotenv import load_dotenv
from bolna.input_handlers.telephony import TelephonyInputHandler
from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)
load_dotenv()


class PlivoInputHandler(TelephonyInputHandler):
    def __init__(self, queues, websocket=None, input_types=None, mark_event_meta_data=None, turn_based_conversation=False,
                 is_welcome_message_played=False, observable_variables=None):
        super().__init__(queues, websocket, input_types, mark_event_meta_data, turn_based_conversation,
                         is_welcome_message_played=is_welcome_message_played, observable_variables=observable_variables)
        self.io_provider = 'plivo'
        self.client = plivosdk.RestClient(os.getenv('PLIVO_AUTH_ID'), os.getenv('PLIVO_AUTH_TOKEN'))

    async def call_start(self, packet):
        start = packet['start']
        self.call_sid = start['callId']
        self.stream_sid = start['streamId']

    async def disconnect_stream(self):
        try:
            self.client.calls.delete_all_streams(self.call_sid)
        except Exception as e:
            logger.info('Error deleting plivo stream: {}'.format(str(e)))

    def get_mark_event_meta_data_obj(self, packet):
        mark_id = packet["name"]
        return self.mark_event_meta_data.fetch_data(mark_id)
