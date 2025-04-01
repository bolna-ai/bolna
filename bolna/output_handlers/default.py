import json
import uuid
import base64
from dotenv import load_dotenv
from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)
load_dotenv()


class DefaultOutputHandler:
    def __init__(self, io_provider='default', websocket=None, queue=None, is_web_based_call=False, mark_event_meta_data=None):
        self.websocket = websocket
        self.is_interruption_task_on = False
        self.queue = queue
        self.io_provider = io_provider
        self.is_chunking_supported = True
        self.is_last_hangup_chunk_sent = False
        # self.is_welcome_message_sent = False
        self.is_web_based_call = is_web_based_call
        self.mark_event_meta_data = mark_event_meta_data

    # @TODO Figure out the best way to handle this
    async def handle_interruption(self):
        logger.info("#######   Sending interruption message ####################")
        response = {"data": None, "type": "clear"}
        await self.websocket.send_json(response)

    def process_in_chunks(self, yield_chunks=False):
        return self.is_chunking_supported and yield_chunks

    def get_provider(self):
        return self.io_provider

    def set_hangup_sent(self):
        self.is_last_hangup_chunk_sent = True

    def hangup_sent(self):
        return self.is_last_hangup_chunk_sent

    # def welcome_message_sent(self):
    #     return self.is_welcome_message_sent

    async def send_init_acknowledgement(self):
        data = {
            "type": "ack"
        }
        logger.info(f"Sending ack event")
        await self.websocket.send_text(json.dumps(data))

    async def handle(self, packet):
        try:
            logger.info(f"Packet received:")
            # if (self.is_web_based_call and packet["meta_info"].get("message_category", "") == "agent_welcome_message" and
            #         packet["meta_info"].get("is_final_chunk_of_entire_response", True)):
            #     self.is_welcome_message_sent = True

            data = None
            if packet["meta_info"]['type'] in ('audio', 'text'):
                if packet["meta_info"]['type'] == 'audio':
                    logger.info(f"Sending audio")
                    data = base64.b64encode(packet['data']).decode("utf-8")
                elif packet["meta_info"]['type'] == 'text':
                    logger.info(f"Sending text response {packet['data']}")
                    data = packet['data']

                # sending of pre-mark message
                if packet["meta_info"]['type'] == 'audio':
                    pre_mark_event_meta_data = {
                        "type": "pre_mark_message",
                    }
                    mark_id = str(uuid.uuid4())
                    self.mark_event_meta_data.update_data(mark_id, pre_mark_event_meta_data)
                    mark_message = {
                        "type": "mark",
                        "name": mark_id
                    }
                    logger.info(f"Sending pre-mark event - {mark_message}")
                    await self.websocket.send_text(json.dumps(mark_message))

                logger.info(f"Sending to the frontend {len(data)}")
                response = {"data": data, "type": packet["meta_info"]['type']}
                await self.websocket.send_json(response)

                # sending of post-mark message
                if packet["meta_info"]['type'] == 'audio':
                    meta_info = packet["meta_info"]
                    mark_event_meta_data = {
                        "text_synthesized": "" if meta_info["sequence_id"] == -1 else meta_info.get("text_synthesized",
                                                                                                    ""),
                        "type": meta_info.get('message_category', ''),
                        "is_first_chunk": meta_info.get("is_first_chunk", False),
                        "is_final_chunk": meta_info.get("end_of_llm_stream", False) and meta_info.get("end_of_synthesizer_stream", False),
                        "sequence_id": meta_info["sequence_id"]
                    }
                    mark_id = meta_info.get("mark_id") if (
                                meta_info.get("mark_id") and meta_info.get("mark_id") != "") else str(uuid.uuid4())
                    logger.info(f"Mark meta data being saved for mark id - {mark_id} is - {mark_event_meta_data}")
                    self.mark_event_meta_data.update_data(mark_id, mark_event_meta_data)
                    mark_message = {
                        "type": "mark",
                        "name": mark_id
                    }
                    logger.info(f"Sending mark event - {mark_message}")
                    await self.websocket.send_text(json.dumps(mark_message))
            else:
                logger.error("Other modalities are not implemented yet")
        except Exception as e:
            logger.error(f"something went wrong in speaking {e}")
