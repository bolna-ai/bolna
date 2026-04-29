import io
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

from .base_synthesizer import BaseSynthesizer
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.utils import convert_audio_to_wav, resample

logger = configure_logger(__name__)
load_dotenv()


class OPENAISynthesizer(BaseSynthesizer):
    def __init__(
        self, voice, audio_format="mp3", model="tts-1", stream=False, sampling_rate=8000, buffer_size=400, **kwargs
    ):
        super().__init__(kwargs.get("task_manager_instance"), stream, buffer_size)
        self.voice = voice
        self.model = model
        self.sample_rate = int(sampling_rate) if isinstance(sampling_rate, str) else sampling_rate
        self.stream = False
        api_key = kwargs.get("synthesizer_key", os.getenv("OPENAI_API_KEY"))
        self.async_client = AsyncOpenAI(api_key=api_key)

    def supports_websocket(self):
        return True

    # ------------------------------------------------------------------
    # BaseSynthesizer hooks
    # ------------------------------------------------------------------

    def _process_http_audio(self, audio):
        # OpenAI always returns mp3 — convert + resample to target rate
        return resample(convert_audio_to_wav(audio, "mp3"), self.sample_rate, format="wav")

    async def _generate_http(self, text):
        spoken_response = await self.async_client.audio.speech.create(
            model=self.model,
            voice=self.voice,
            response_format="mp3",
            input=text,
        )
        buffer = io.BytesIO()
        for chunk in spoken_response.iter_bytes(chunk_size=4096):
            buffer.write(chunk)
        buffer.seek(0)
        return buffer.getvalue()

    async def synthesize(self, text):
        return await self._generate_http(text)

    # ------------------------------------------------------------------
    # generate / push — use base _generate_http_loop
    # ------------------------------------------------------------------

    async def generate(self):
        async for packet in self._generate_http_loop():
            yield packet
