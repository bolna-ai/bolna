import aiohttp
import os
import traceback

from .base_synthesizer import BaseSynthesizer
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.utils import create_ws_data_packet

logger = configure_logger(__name__)


class SmallestSynthesizer(BaseSynthesizer):
    def __init__(self, voice, voice_id, model="lightning", audio_format="mp3", sampling_rate="8000",
                 stream=False, buffer_size=400, synthesizer_key=None, **kwargs):
        super().__init__(stream)
        self.api_key = os.environ["SMALLEST_API_KEY"] if synthesizer_key is None else synthesizer_key
        self.voice_id = voice_id
        self.model = model
        self.stream = False
        self.sampling_rate = int(sampling_rate)
        self.api_url = f"https://waves-api.smallest.ai/api/v1/{self.model}/get_speech"
        self.first_chunk_generated = False
        self.last_text_sent = False
        self.meta_info = None
        self.synthesized_characters = 0
        self.previous_request_ids = []

    def get_engine(self):
        return self.model

    async def __send_payload(self, payload):
        headers = {
            'Authorization': 'Bearer {}'.format(self.api_key),
            'Content-Type': 'application/json'
        }

        async with aiohttp.ClientSession() as session:
            if payload is not None:
                async with session.post(self.api_url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        data = await response.read()
                        return data
                    else:
                        logger.error(f"Error: {response.status} - {await response.text()}")
            else:
                logger.info("Payload was null")

    async def synthesize(self, text):
        audio = await self.__generate_http(text)
        return audio

    def supports_websocket(self):
        return False

    async def __generate_http(self, text):
        payload = None
        logger.info(f"text {text}")

        payload = {
            "text": text,
            "voice_id": self.voice_id,
            "sample_rate": self.sampling_rate,
            "add_wav_header": False
        }
        response = await self.__send_payload(payload)
        return response

    def get_synthesized_characters(self):
        return self.synthesized_characters

    async def generate(self):
        try:
            while True:
                message = await self.internal_queue.get()
                logger.info(f"Generating TTS response for message: {message}")
                meta_info, text = message.get("meta_info"), message.get("data")
                meta_info['is_cached'] = False
                self.synthesized_characters += len(text)
                audio = await self.__generate_http(text)
                if not audio:
                    audio = b'\x00'

                meta_info['text'] = text
                if not self.first_chunk_generated:
                    meta_info["is_first_chunk"] = True
                    self.first_chunk_generated = True

                if "end_of_llm_stream" in meta_info and meta_info["end_of_llm_stream"]:
                    meta_info["end_of_synthesizer_stream"] = True
                    self.first_chunk_generated = False

                meta_info['format'] = "wav"
                yield create_ws_data_packet(audio, meta_info)

        except Exception as e:
            traceback.print_exc()
            logger.info(f"Error in smallest generate {e}")

    async def push(self, message):
        logger.info(f"Pushed message to internal queue {message}")
        self.internal_queue.put_nowait(message)
