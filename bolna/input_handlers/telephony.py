import traceback
from .default import DefaultInputHandler
import asyncio
import base64
import json
from dotenv import load_dotenv
from bolna.helpers.utils import create_ws_data_packet
from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)
load_dotenv()


class TelephonyInputHandler(DefaultInputHandler):
    def __init__(self, queues, websocket=None, input_types=None, mark_event_meta_data=None, turn_based_conversation=False,
                 is_welcome_message_played=False, observable_variables=None):
        super().__init__(queues, websocket, input_types, turn_based_conversation, is_welcome_message_played=is_welcome_message_played)
        self.stream_sid = None
        self.call_sid = None
        self.buffer = []
        self.message_count = 0
        self.mark_event_meta_data = mark_event_meta_data
        self.last_media_received = 0
        self.io_provider = None
        # This variable stores the response which has been heard by the user
        self.response_heard_by_user = ""
        self.is_audio_being_played_to_user = False
        # self.is_clear_event_sent = False
        self.observable_variables = observable_variables

    def get_stream_sid(self):
        return self.stream_sid

    def get_call_sid(self):
        return self.call_sid

    async def call_start(self, packet):
        pass

    async def disconnect_stream(self):
        pass

    def get_mark_event_meta_data_obj(self, packet):
        pass

    def update_is_audio_being_played(self, value):
        self.is_audio_being_played_to_user = value
        # self.is_clear_event_sent = value

    def _is_audio_being_played_to_user(self):
        return self.is_audio_being_played_to_user

    def get_response_heard_by_user(self):
        response = self.response_heard_by_user
        self.response_heard_by_user = ""
        return response.strip()

    async def process_mark_message(self, packet):
        mark_event_meta_data_obj = self.get_mark_event_meta_data_obj(packet)
        if not mark_event_meta_data_obj:
            logger.info("No object retrieved from global dict of mark_event_meta_data")
            return

        # if self.is_clear_event_sent and not mark_event_meta_data_obj.get("is_first_chunk"):
        #     return
        #
        # self.is_clear_event_sent = False
        # mark_event_meta_data = {
        #     "text_synthesized": meta_info.get("text_synthesized", ""),
        #     "type": meta_info.get('message_category', ''),
        #     "is_first_chunk": meta_info.get("is_first_chunk", False),
        #     "is_final_chunk": meta_info.get("end_of_synthesizer_stream", False)
        # }

        logger.info(f"Mark event meta data object retrieved = {mark_event_meta_data_obj}")
        message_type = mark_event_meta_data_obj.get("type")
        self.response_heard_by_user += mark_event_meta_data_obj.get("text_synthesized")

        # if mark_event_meta_data_obj.get("is_first_chunk"):
        #     self.is_audio_being_played_to_user = True

        if mark_event_meta_data_obj.get("is_final_chunk"):
            self.is_audio_being_played_to_user = False

        if message_type == "agent_welcome_message" and mark_event_meta_data_obj.get("is_final_chunk"):
            logger.info("Received mark event for agent_welcome_message")
            self.is_welcome_message_played = True
        elif message_type == "agent_hangup" and mark_event_meta_data_obj.get("is_final_chunk"):
            logger.info(f"Agent hangup has been triggered")
            self.observable_variables["agent_hangup_observable"].value = True


    async def stop_handler(self):
        asyncio.create_task(self.disconnect_stream())
        logger.info("stopping handler")
        self.running = False
        logger.info("sleeping for 2 seconds so that whatever needs to pass is passed")
        await asyncio.sleep(2)
        try:
            await self.websocket.close()
            logger.info("WebSocket connection closed")
        except Exception as e:
            logger.info(f"Error closing WebSocket: {e}")

    async def ingest_audio(self, audio_data, meta_info):
        ws_data_packet = create_ws_data_packet(data=audio_data, meta_info=meta_info)
        self.queues['transcriber'].put_nowait(ws_data_packet)

    async def _listen(self):
        buffer = []
        while True:
            try:
                message = await self.websocket.receive_text()

                packet = json.loads(message)
                if packet['event'] == 'start':
                    await self.call_start(packet)
                elif packet['event'] == 'media':
                    media_data = packet['media']
                    media_audio = base64.b64decode(media_data['payload'])
                    media_ts = int(media_data["timestamp"])

                    if 'chunk' in packet['media'] or ('track' in packet['media'] and packet['media']['track'] == 'inbound'):
                        meta_info = {
                            'io': self.io_provider,
                            'call_sid': self.call_sid,
                            'stream_sid': self.stream_sid,
                            'sequence': self.input_types['audio']
                        }
                        '''
                        if self.last_media_received + 20 < media_ts:
                            bytes_to_fill = 8 * (media_ts - (self.last_media_received + 20))
                            logger.info(f"Filling {bytes_to_fill} bytes of silence")
                            #await self.ingest_audio(b"\xff" * bytes_to_fill, meta_info)
                        '''
                        self.last_media_received = media_ts
                        buffer.append(media_audio)
                        self.message_count += 1

                        # Send 100 ms of audio to deepgram
                        if self.message_count == 10:
                            merged_audio = b''.join(buffer)
                            buffer = []
                            await self.ingest_audio(merged_audio, meta_info)
                            self.message_count = 0
                    else:
                        logger.info("Getting media elements but not inbound media")

                elif packet['event'] == 'mark' or packet['event'] == 'playedStream':
                    await self.process_mark_message(packet)

                elif packet['event'] == 'stop':
                    logger.info('call stopping')
                    ws_data_packet = create_ws_data_packet(data=None, meta_info={'io': 'default', 'eos': True})
                    self.queues['transcriber'].put_nowait(ws_data_packet)
                    break

            except Exception as e:
                traceback.print_exc()
                ws_data_packet = create_ws_data_packet(
                    data=None,
                    meta_info={
                        'io': 'default',
                        'eos': True
                    })
                self.queues['transcriber'].put_nowait(ws_data_packet)
                logger.info('Exception in twilio_receiver reading events: {}'.format(e))
                break

    async def handle(self):
        self.websocket_listen_task = asyncio.create_task(self._listen())
