import io
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

from .base_synthesizer import BaseSynthesizer
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.utils import convert_audio_to_wav, resample

logger = configure_logger(__name__)
load_dotenv()

GANDR_TTS_BASE_URL = os.getenv("GANDR_TTS_BASE_URL", "https://tts.gandr.ai/v1")


class GandrSynthesizer(BaseSynthesizer):
    """Gandr TTS (https://gandr.ai). The API is OpenAI compatible (POST /v1/audio/speech,
    Bearer key), so this mirrors OPENAISynthesizer with the Gandr endpoint, key and
    voices (gandr-mia, gandr-ava, gandr-jenny, gandr-dane, gandr-leo, gandr-lewis)."""

    def __init__(
        self, voice="gandr-mia", audio_format="mp3", model="tts-1", stream=False, sampling_rate=8000, buffer_size=400, **kwargs
    ):
        super().__init__(kwargs.get("task_manager_instance"), stream, buffer_size)
        self.voice = voice
        self.model = model
        self.sample_rate = int(sampling_rate) if isinstance(sampling_rate, str) else sampling_rate
        self.stream = False
        api_key = kwargs.get("synthesizer_key", os.getenv("GANDR_API_KEY"))
        self.async_client = AsyncOpenAI(api_key=api_key, base_url=GANDR_TTS_BASE_URL)

    def supports_websocket(self):
        return True

    # ------------------------------------------------------------------
    # BaseSynthesizer hooks
    # ------------------------------------------------------------------

    def _process_http_audio(self, audio):
        # Gandr returns mp3 by default; convert and resample to the target rate
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
    # generate / push use the base _generate_http_loop
    # ------------------------------------------------------------------

    async def generate(self):
        async for packet in self._generate_http_loop():
            yield packet
