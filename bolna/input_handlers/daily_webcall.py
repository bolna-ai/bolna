import asyncio
import base64
import time
from urllib.parse import urlparse
from dotenv import load_dotenv
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.utils import create_ws_data_packet

logger = configure_logger(__name__)
load_dotenv()


class DailyInputHandler:
    def __init__(self, queues=None, websocket=None, input_types=None, mark_event_meta_data=None, queue=None,
                conversation_recording=None, room_url=None):
        self.queues = queues
        self.websocket = websocket
        self.input_types = input_types
        self.websocket_listen_task = None
        self.stream_sid = None
        self.room_url = room_url
        self.running = True
        self.queue = queue
        self.conversation_recording = conversation_recording

    def get_stream_sid(self):
        parsed_url = urlparse(self.room_url)
        path = parsed_url.path
        self.stream_sid = path.split('/')[-1]
        return self.stream_sid

    async def stop_handler(self):
        self.running = False
        try:
            if not self.queue:
                await self.websocket.close()
        except Exception as e:
            logger.error(f"Error closing WebSocket: {e}")

    def __process_audio(self, audio):
        data = base64.b64decode(audio)
        ws_data_packet = create_ws_data_packet(
            data=data,
            meta_info={
                'io': 'daily',
                'type': 'audio',
                'sequence': self.input_types['audio']
            })
        if self.conversation_recording:
            if self.conversation_recording["metadata"]["started"] == 0:
                self.conversation_recording["metadata"]["started"] = time.time()
            self.conversation_recording['input']['data'] += data

        self.queues['transcriber'].put_nowait(ws_data_packet)

    async def _listen(self):
        try:
            while self.running:
                if self.queue is not None:
                    logger.info(f"self.queue is not None and hence listening to the queue")
                    request = await self.queue.get()
                else:
                    request = await self.websocket.receive_json()
                await self.process_message(request)
        except Exception as e:
            # Send EOS message to transcriber to shut the connection
            ws_data_packet = create_ws_data_packet(
                data=None,
                meta_info={
                    'io': 'daily',
                    'eos': True
                })
            import traceback
            traceback.print_exc()
            self.queues['transcriber'].put_nowait(ws_data_packet)
            logger.info(f"Error while handling websocket message: {e}")
            return

    async def process_message(self, message):
        if message['type'] not in self.input_types.keys():
            logger.info(f"straight away returning")
            return {"message": "invalid input type"}

        if message['type'] == 'audio':
            self.__process_audio(message['data'])

        # elif message["type"] == "text":
        #     logger.info(f"Received text: {message['data']}")
        #     self.__process_text(message['data'])
        else:
            return {"message": "Other modalities not implemented yet"}

    async def handle(self):
        self.websocket_listen_task = asyncio.create_task(self._listen())
