import os
import plivo as plivosdk
from dotenv import load_dotenv
from bolna.input_handlers.telephony import TelephonyInputHandler
from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)
load_dotenv()


class PlivoInputHandler(TelephonyInputHandler):
    def __init__(self, queues, websocket=None, input_types=None, mark_set=None, turn_based_conversation=False):
        super().__init__(queues, websocket, input_types, mark_set, turn_based_conversation)
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

    async def process_mark_message(self, packet, mark_set):
        mark_event_name = packet["name"]
        if mark_event_name in mark_set:
            mark_set.remove(mark_event_name)
        if mark_event_name == "agent_welcome_message":
            logger.info("Received mark event for agent_welcome_message")
            self.is_welcome_message_played = True
