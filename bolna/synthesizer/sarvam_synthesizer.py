import aiohttp
import os
import uuid
import traceback
import base64
from .base_synthesizer import BaseSynthesizer
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.utils import create_ws_data_packet

logger = configure_logger(__name__)


class SarvamSynthesizer(BaseSynthesizer):
    def __init__(self, voice_id, model, language, sampling_rate="8000", stream=False, buffer_size=400, synthesizer_key=None, **kwargs):
        super().__init__(kwargs.get("task_manager_instance", None), stream)
        self.api_key = os.environ["SARVAM_API_KEY"] if synthesizer_key is None else synthesizer_key
        self.voice_id = voice_id
        self.model = model
        self.stream = False
        self.sampling_rate = int(sampling_rate)
        self.api_url = f"https://api.sarvam.ai/text-to-speech"

        self.language = language
        self.loudness = 1.0
        self.pitch = 0.0
        self.pace = 1.0
        self.enable_preprocessing = True

        self.previous_request_ids = []

    def get_engine(self):
        return self.model

    async def __send_payload(self, payload):
        headers = {
            'api-subscription-key': self.api_key,
            'Content-Type': 'application/json'
        }

        async with aiohttp.ClientSession() as session:
            if payload is not None:
                async with session.post(self.api_url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data and data.get('audios', []) and isinstance(data.get('audios', []), list):
                            return data.get('audios')[0]
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
        logger.info(f"text {text}")

        payload = {
            "target_language_code": self.language,
            "text": text,
            "speaker": self.voice_id,
            "pitch": self.pitch,
            "loudness": self.loudness,
            "speech_sample_rate": self.sampling_rate,
            "enable_preprocessing": self.enable_preprocessing,
            "model": self.model
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

                if not self.should_synthesize_response(meta_info.get('sequence_id')):
                    logger.info(
                        f"Not synthesizing text as the sequence_id ({meta_info.get('sequence_id')}) of it is not in the list of sequence_ids present in the task manager.")
                    return

                meta_info['is_cached'] = False
                self.synthesized_characters += len(text)
                audio = await self.__generate_http(text)
                if not audio:
                    audio = b'\x00'
                else:
                    audio = base64.b64decode(audio)

                meta_info['text'] = text
                self.set_first_chunk_metadata(meta_info)
                self.set_end_of_stream_metadata(meta_info)

                yield self.create_audio_packet(audio, meta_info, f"{text} ")

        except Exception as e:
            traceback.print_exc()
            logger.info(f"Error in sarvam generate {e}")
