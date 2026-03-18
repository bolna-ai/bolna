import io
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

from .base_synthesizer import BaseSynthesizer
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.utils import convert_audio_to_wav, create_ws_data_packet, resample

logger = configure_logger(__name__)
load_dotenv()


class OPENAISynthesizer(BaseSynthesizer):
    def __init__(self, voice, audio_format="mp3", model="tts-1", stream=False, sampling_rate=8000,
                 buffer_size=400, **kwargs):
        super().__init__(kwargs.get("task_manager_instance"), stream, buffer_size)
        self.voice = voice
        self.model = model
        self.sample_rate = int(sampling_rate) if isinstance(sampling_rate, str) else sampling_rate
        self.stream = False  # OpenAI TTS doesn't support true streaming
        api_key = kwargs.get("synthesizer_key", os.getenv("OPENAI_API_KEY"))
        self.async_client = AsyncOpenAI(api_key=api_key)

    def supports_websocket(self):
        return True

    # ------------------------------------------------------------------
    # HTTP synthesis
    # ------------------------------------------------------------------

    async def _generate_http(self, text):
        spoken_response = await self.async_client.audio.speech.create(
            model=self.model, voice=self.voice, response_format="mp3", input=text,
        )
        buffer = io.BytesIO()
        for chunk in spoken_response.iter_bytes(chunk_size=4096):
            buffer.write(chunk)
        buffer.seek(0)
        return buffer.getvalue()

    async def synthesize(self, text):
        return await self._generate_http(text)

    # ------------------------------------------------------------------
    # generate / push
    # ------------------------------------------------------------------

    async def generate(self):
        try:
            while True:
                message = await self.internal_queue.get()
                logger.info(f"Generating TTS response for message: {message}")
                meta_info, text = message.get("meta_info"), message.get("data")
                meta_info["text"] = text

                if not self.should_synthesize_response(meta_info.get("sequence_id")):
                    logger.info(f"Not synthesizing: sequence_id {meta_info.get('sequence_id')} not current")
                    return

                logger.info("Generating without a stream")
                audio = await self._generate_http(text)

                self._stamp_first_chunk(meta_info)
                self._stamp_end_of_stream(meta_info)

                audio = resample(convert_audio_to_wav(audio, "mp3"), self.sample_rate, format="wav")
                yield create_ws_data_packet(audio, meta_info)

        except Exception as e:
            logger.error(f"Error in openai generate {e}")
            raise

    async def push(self, message):
        logger.info(f"Pushed message to internal queue {message}")
        self.internal_queue.put_nowait(message)
