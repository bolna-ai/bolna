from bolna.input_handlers.telephony import TelephonyInputHandler
from dotenv import load_dotenv
from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)
load_dotenv()


class TwilioInputHandler(TelephonyInputHandler):
    def __init__(self, queues, websocket=None, input_types=None, mark_set=None, turn_based_conversation=False):
        super().__init__(queues, websocket, input_types, mark_set, turn_based_conversation)
        self.io_provider = 'twilio'

    async def call_start(self, packet):
        start = packet['start']
        self.call_sid = start['callSid']
        self.stream_sid = start['streamSid']

    async def process_mark_message(self, packet, mark_set):
        mark_event_name = packet["mark"]["name"]
        if mark_event_name in mark_set:
            mark_set.remove(mark_event_name)
        if mark_event_name == "agent_welcome_message":
            logger.info("Received mark event for agent_welcome_message")
            self.is_welcome_message_played = True
