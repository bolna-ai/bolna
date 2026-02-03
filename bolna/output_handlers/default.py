import json
import uuid
import time
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
        self.welcome_message_sent_ts = None
        self._closed = False

    def close(self):
        """Mark the output handler as closed to prevent sends after websocket close."""
        self._closed = True

    def is_closed(self):
        return self._closed

    async def handle_interruption(self):
        if self._closed:
            return
        try:
            response = {"data": None, "type": "clear"}
            await self.websocket.send_json(response)
            self.mark_event_meta_data.clear_data()
        except Exception as e:
            logger.info(f"WebSocket closed during interruption: {e}")
            self._closed = True

    def process_in_chunks(self, yield_chunks=False):
        return self.is_chunking_supported and yield_chunks

    def get_provider(self):
        return self.io_provider

    def set_hangup_sent(self):
        self.is_last_hangup_chunk_sent = True

    def hangup_sent(self):
        return self.is_last_hangup_chunk_sent

    def get_welcome_message_sent_ts(self):
        return self.welcome_message_sent_ts
    
    def requires_custom_voicemail_detection(self):
        return True

    # def welcome_message_sent(self):
    #     return self.is_welcome_message_sent

    async def send_init_acknowledgement(self):
        if self._closed:
            return
        try:
            data = {
                "type": "ack"
            }
            logger.info(f"Sending ack event")
            await self.websocket.send_text(json.dumps(data))
        except Exception as e:
            logger.info(f"WebSocket closed during init ack: {e}")
            self._closed = True

    async def handle(self, packet):
        if self._closed:
            return
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
                    await self.websocket.send_text(json.dumps(mark_message))

                logger.info(f"Sending to the frontend {len(data)}")
                if packet['meta_info'].get('message_category') == 'agent_welcome_message' and not self.welcome_message_sent_ts:
                    self.welcome_message_sent_ts = time.time() * 1000

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

                    self.mark_event_meta_data.update_data(mark_id, mark_event_meta_data)
                    mark_message = {
                        "type": "mark",
                        "name": mark_id
                    }
                    await self.websocket.send_text(json.dumps(mark_message))
            else:
                logger.error("Other modalities are not implemented yet")
        except Exception as e:
            self._closed = True  # Prevent further send attempts
            logger.debug(f"WebSocket send failed (client disconnected): {e}")
