import copy
import aiohttp
import os
import uuid
from dotenv import load_dotenv
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.utils import convert_audio_to_wav, create_ws_data_packet
from bolna.memory.cache.inmemory_scalar_cache import InmemoryScalarCache
from .base_synthesizer import BaseSynthesizer

logger = configure_logger(__name__)
load_dotenv()
DEEPGRAM_HOST = os.getenv('DEEPGRAM_HOST', 'api.deepgram.com')
DEEPGRAM_TTS_URL = f"https://{DEEPGRAM_HOST}/v1/speak"


class DeepgramSynthesizer(BaseSynthesizer):
    def __init__(self, voice_id="luna", voice="female", audio_format="pcm", sampling_rate="8000", stream=False, buffer_size=400, caching=True,
                 model="aura-2-thalia-en", **kwargs):
        super().__init__(kwargs.get("task_manager_instance", None), stream, buffer_size)

        # Deepgram supports PCM, mulaw, wav, mp3, ogg
        self.format = "mulaw" if audio_format in ["pcm", "wav"] else audio_format
        self.sample_rate = str(sampling_rate)
        self.model = model  # e.g., "aura-2-thalia-en"
        self.first_chunk_generated = False
        self.api_key = kwargs.get("transcriber_key", os.getenv('DEEPGRAM_AUTH_TOKEN'))

        self.synthesized_characters = 0
        self.caching = caching
        if caching:
            self.cache = InmemoryScalarCache()

    def get_synthesized_characters(self):
        return self.synthesized_characters

    def get_engine(self):
        return self.model

    async def __generate_http(self, text):
        headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "application/json"
        }
        url = (
            f"{DEEPGRAM_TTS_URL}"
            f"?encoding={self.format}&sample_rate={self.sample_rate}&model={self.model}"
        )

        logger.info(f"Sending Deepgram TTS request {url}")

        payload = {"text": text}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    audio_bytes = await response.read()
                    if response.status == 200:
                        logger.info(f"Deepgram TTS success: {len(audio_bytes)} bytes")
                        return audio_bytes
                    else:
                        logger.error(f"Deepgram TTS failed: {response.status}, {audio_bytes}")
                        return b"\x00"
        except Exception as e:
            logger.error(f"Deepgram TTS request error: {e}")
            return b"\x00"

    def supports_websocket(self):
        return False

    async def open_connection(self):
        pass

    async def synthesize(self, text):
        """For one-off synthesis (IVR, voice lab, etc.)"""
        try:
            audio = await self.__generate_http(text)
            if self.format == "mp3":
                audio = convert_audio_to_wav(audio, source_format="mp3")
            return audio
        except Exception as e:
            logger.error(f"Could not synthesize: {e}")
            return b"\x00"

    async def generate(self):
        while True:
            message = await self.internal_queue.get()
            logger.info(f"Generating TTS response for: {message}")
            meta_info, text = message.get("meta_info"), message.get("data")

            if not self.should_synthesize_response(meta_info.get("sequence_id")):
                logger.info(f"Skipping synthesis (seq_id {meta_info.get('sequence_id')} not in task manager)")
                return

            if self.caching and self.cache.get(text):
                logger.info(f"Cache hit for text: {text}")
                audio_bytes = self.cache.get(text)
            else:
                logger.info("Cache miss → synthesizing")
                self.synthesized_characters += len(text)
                audio_bytes = await self.__generate_http(text)
                if self.caching:
                    self.cache.set(text, audio_bytes)

            if self.format == "mp3":
                audio_bytes = convert_audio_to_wav(audio_bytes, source_format="mp3")

            if not self.first_chunk_generated:
                meta_info["is_first_chunk"] = True
                self.first_chunk_generated = True
            else:
                meta_info["is_first_chunk"] = False

            if meta_info.get("end_of_llm_stream"):
                meta_info["end_of_synthesizer_stream"] = True
                self.first_chunk_generated = False

            meta_info.update({
                "text": text,
                "format": self.format,
                "text_synthesized": f"{text} ",
                "mark_id": str(uuid.uuid4())
            })

            yield create_ws_data_packet(audio_bytes, meta_info)

    async def push(self, message):
        logger.info(f"Pushed message to queue {message}")
        self.internal_queue.put_nowait(copy.deepcopy(message))
