import base64
from dotenv import load_dotenv
from bolna.helpers.logger_config import configure_logger
from daily import Daily, CallClient

logger = configure_logger(__name__)
load_dotenv()


class DailyOutputHandler:
    def __init__(self, io_provider='daily', websocket=None, queue=None, room_url=None, mark_event_meta_data=None):
        self.websocket = websocket
        self.is_interruption_task_on = False
        self.queue = queue
        self.io_provider = io_provider
        self.is_chunking_supported = True
        self.room_url = room_url

        # initiating daily client
        Daily.init()
        self.daily_call = CallClient()

        # Create a virtual microphone device
        self.mic_device = Daily.create_microphone_device(
            "bolna-ai-mic",
            sample_rate=8000,
            channels=1
        )

        # join the call with the virtual microphone
        self.daily_call.join(meeting_url=room_url, client_settings={
            "inputs": {
                "camera": False,
                "microphone": {
                    "isEnabled": True,
                    "settings": {
                        "deviceId": "bolna-ai-mic"
                    }
                }
            }
        })

        # set username
        self.daily_call.set_user_name(user_name="bolna-ai")

        # Start recording
        self.daily_call.start_recording()

    # @TODO Figure out the best way to handle this
    async def handle_interruption(self):
        logger.info("#######   Sending interruption message ####################")
        response = {"connection": True, "type": "clear"}
        await self.websocket.send_json(response)

    def process_in_chunks(self, yield_chunks=False):
        return self.is_chunking_supported and yield_chunks

    async def release_call(self):
        logger.info("#######   Closing daily recording #####################")
        self.daily_call.stop_recording()
        logger.info("#######   leaving daily call #####################")
        self.daily_call.leave()

    def get_provider(self):
        return self.io_provider

    async def handle(self, packet):
        try:
            if packet["meta_info"]['type'] == 'audio':
                logger.info(f"audio Packet received in audio Web-call")
                if packet["meta_info"]['type'] == 'audio':
                    logger.info(f"Sending audio from daily Web-call")
                    self.mic_device.write_frames(packet['data'])
                logger.info(f"Sending to the frontend from daily Web-call")
                response = {"connection": True, "type": packet["meta_info"]['type']}
                await self.websocket.send_json(response)
            else:
                logger.error("Other modalities are not implemented yet")
        except Exception as e:
            logger.error(f"something went wrong in speaking {e}")