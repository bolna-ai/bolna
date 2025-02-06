import base64
from dotenv import load_dotenv
from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)
load_dotenv()


class DefaultOutputHandler:
    def __init__(self, io_provider='default', websocket=None, queue=None, is_web_based_call=False):
        self.websocket = websocket
        self.is_interruption_task_on = False
        self.queue = queue
        self.io_provider = io_provider
        self.is_chunking_supported = True
        self.is_last_hangup_chunk_sent = False
        self.is_welcome_message_sent = False
        self.is_web_based_call = is_web_based_call

    # @TODO Figure out the best way to handle this
    async def handle_interruption(self):
        logger.info("#######   Sending interruption message ####################")
        response = {"data": None, "type": "clear"}
        await self.websocket.send_json(response)

    def process_in_chunks(self, yield_chunks=False):
        return self.is_chunking_supported and yield_chunks

    def get_provider(self):
        return self.io_provider

    def hangup_sent(self):
        return self.is_last_hangup_chunk_sent

    def welcome_message_sent(self):
        return self.is_welcome_message_sent

    async def handle(self, packet):
        try:
            logger.info(f"Packet received:")
            if (self.is_web_based_call and packet["meta_info"].get("message_category", "") == "agent_welcome_message" and
                    packet["meta_info"].get("is_final_chunk_of_entire_response", True)):
                self.is_welcome_message_sent = True

            data = None
            if packet["meta_info"]['type'] in ('audio', 'text'):
                if packet["meta_info"]['type'] == 'audio':
                    logger.info(f"Sending audio")
                    data = base64.b64encode(packet['data']).decode("utf-8")
                elif packet["meta_info"]['type'] == 'text':
                    logger.info(f"Sending text response {packet['data']}")
                    data = packet['data']

                logger.info(f"Sending to the frontend {len(data)}")
                response = {"data": data, "type": packet["meta_info"]['type']}
                await self.websocket.send_json(response)

            else:
                logger.error("Other modalities are not implemented yet")
        except Exception as e:
            logger.error(f"something went wrong in speaking {e}")
