import os
import requests
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv
from bolna.input_handlers.telephony import TelephonyInputHandler
from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)
load_dotenv()


class VobizInputHandler(TelephonyInputHandler):
    def __init__(self, queues, websocket=None, input_types=None, mark_event_meta_data=None, turn_based_conversation=False,
                 is_welcome_message_played=False, observable_variables=None):
        super().__init__(queues, websocket, input_types, mark_event_meta_data, turn_based_conversation,
                         is_welcome_message_played=is_welcome_message_played, observable_variables=observable_variables)
        self.io_provider = 'vobiz'

    async def call_start(self, packet):
        logger.info('Vobiz call started: {}'.format(packet))
        start = packet['start']
        self.call_sid = start['callId']
        self.stream_sid = start['streamId']

    async def disconnect_stream(self):
        try:
            logger.info('Disconnecting vobiz stream for call: {}'.format(self.call_sid))
            api_key = os.getenv('VOBIZ_API_KEY')
            api_secret = os.getenv('VOBIZ_API_SECRET')
            call_uuid = self.call_sid
            
            if api_key and call_uuid:
                url = f"https://api.vobiz.ai/api/v1/Account/{api_key}/Call/{call_uuid}/"
                auth = None
                if api_key and api_secret:
                    auth = HTTPBasicAuth(api_key, api_secret)
                response = requests.delete(url, auth=auth)
                if response.status_code == 200:
                    logger.info(f"Successfully disconnected Vobiz call: {call_uuid}")
                else:
                    logger.warning(f"Failed to disconnect Vobiz call {call_uuid}: Status {response.status_code}, Response: {response.text}")
            else:
                logger.warning("Cannot disconnect Vobiz call: VOBIZ_AUTH_ID or call_sid missing")
            logger.info('Disconnecting vobiz stream for call: {}'.format(self.call_sid))
        except Exception as e:
            logger.info('Error deleting vobiz stream: {}'.format(str(e)))

    def get_mark_event_meta_data_obj(self, packet):
        mark_id = packet["name"]
        return self.mark_event_meta_data.fetch_data(mark_id)
