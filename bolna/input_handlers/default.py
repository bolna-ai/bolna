import asyncio
import base64
import time
import uuid
from dotenv import load_dotenv
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.utils import create_ws_data_packet

logger = configure_logger(__name__)
load_dotenv()


class DefaultInputHandler:
    def __init__(self, queues=None, websocket=None, input_types=None, mark_event_meta_data=None, queue=None,
                 turn_based_conversation=False, conversation_recording=None, is_welcome_message_played=False,
                 observable_variables=None):
        self.queues = queues
        self.websocket = websocket
        self.input_types = input_types
        self.websocket_listen_task = None
        self.running = True
        self.turn_based_conversation = turn_based_conversation
        self.queue = queue
        self.conversation_recording = conversation_recording
        self.is_welcome_message_played = is_welcome_message_played
        # This variable stores the response which has been heard by the user
        self.response_heard_by_user = ""
        self._is_audio_being_played_to_user = False
        self.observable_variables = observable_variables
        self.mark_event_meta_data = mark_event_meta_data
        self.audio_chunks_received = 0

    def get_audio_chunks_received(self):
        audio_chunks_received = self.audio_chunks_received
        self.audio_chunks_received = 0
        return audio_chunks_received
        
    def update_is_audio_being_played(self, value):
        logger.info(f"Audio is being updated - {value}")
        self._is_audio_being_played_to_user = value

    def is_audio_being_played_to_user(self):
        return self._is_audio_being_played_to_user

    def get_response_heard_by_user(self):
        response = self.response_heard_by_user
        self.response_heard_by_user = ""
        return response.strip()

    async def stop_handler(self):
        self.running = False
        try:
            if not self.queue:
                await self.websocket.close()
        except Exception as e:
            logger.error(f"Error closing WebSocket: {e}")

    def get_stream_sid(self):
        return str(uuid.uuid4())

    def welcome_message_played(self):
        return self.is_welcome_message_played

    def get_mark_event_meta_data_obj(self, packet):
        mark_id = packet["name"]
        return self.mark_event_meta_data.fetch_data(mark_id)

    def process_mark_message(self, packet):
        mark_event_meta_data_obj = self.get_mark_event_meta_data_obj(packet)
        if not mark_event_meta_data_obj:
            logger.info(f"No object retrieved from global dict of mark_event_meta_data for received mark event - {packet}")
            return

        logger.info(f"Mark event meta data object retrieved = {mark_event_meta_data_obj}")
        message_type = mark_event_meta_data_obj.get("type")

        if message_type == "pre_mark_message":
            self.update_is_audio_being_played(True)
            return

        self.audio_chunks_received += 1
        self.response_heard_by_user += mark_event_meta_data_obj.get("text_synthesized")

        if mark_event_meta_data_obj.get("is_final_chunk"):
            if message_type != "is_user_online_message":
                self.observable_variables["final_chunk_played_observable"].value = not self.observable_variables["final_chunk_played_observable"].value

            self.update_is_audio_being_played(False)

            if message_type == "agent_welcome_message":
                logger.info("Received mark event for agent_welcome_message")
                self.audio_chunks_received = 0
                self.is_welcome_message_played = True

            elif message_type == "agent_hangup":
                logger.info(f"Agent hangup has been triggered")
                self.observable_variables["agent_hangup_observable"].value = True

    def __process_mark_event(self, packet):
        self.process_mark_message(packet)

    def __process_audio(self, audio):
        data = base64.b64decode(audio)
        ws_data_packet = create_ws_data_packet(
            data=data,
            meta_info={
                'io': 'default',
                'type': 'audio',
                'sequence': self.input_types['audio']
            })
        if self.conversation_recording:
            if self.conversation_recording["metadata"]["started"] ==0:
                self.conversation_recording["metadata"]["started"] = time.time()
            self.conversation_recording['input']['data'] += data

        self.queues['transcriber'].put_nowait(ws_data_packet)
    
    def __process_text(self, text):
        logger.info(f"Sequences {self.input_types}")
        ws_data_packet = create_ws_data_packet(
            data=text,
            meta_info={
                'io': 'default',
                'type': 'text',
                'sequence': self.input_types['audio']
            })

        if self.turn_based_conversation:
            ws_data_packet["meta_info"]["bypass_synth"] = True
        self.queues['llm'].put_nowait(ws_data_packet)

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
                    'io': 'default',
                    'eos': True
                })
            import traceback
            traceback.print_exc()
            self.queues['transcriber'].put_nowait(ws_data_packet)
            logger.info(f"Error while handling websocket message: {e}")
            return

    async def process_message(self, message):
        # TODO check what condition needs to be added over here
        # if message['type'] not in self.input_types.keys() and not self.turn_based_conversation:
        #     logger.info(f"straight away returning")
        #     return {"message": "invalid input type"}

        if message['type'] == 'audio':
            self.__process_audio(message['data'])

        elif message["type"] == "text":
            logger.info(f"Received text: {message['data']}")
            self.__process_text(message['data'])

        elif message["type"] == "mark":
            logger.info(f"Received mark event")
            self.__process_mark_event(message)

        elif message["type"] == "init":
            logger.info(f"Received init event")
            if self.observable_variables.get("init_event_observable") is not None:
                self.observable_variables.get("init_event_observable").value = message.get("meta_data", None)

        else:
            return {"message": "Other modalities not implemented yet"}
            
    async def handle(self):
        self.websocket_listen_task = asyncio.create_task(self._listen())
