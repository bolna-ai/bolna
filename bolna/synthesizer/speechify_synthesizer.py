import os

import aiohttp

from .base_synthesizer import BaseSynthesizer
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.utils import resample
from bolna.memory.cache.inmemory_scalar_cache import InmemoryScalarCache

logger = configure_logger(__name__)

# Attribution header so Speechify usage is tracked as bolna traffic (required on every
# outbound call per Speechify's integration guidelines).
CALLER_HEADER_NAME = "Speechify-Caller"
CALLER_HEADER_VALUE = "bolna"

# output_format sample rates POST /v1/audio/stream accepts for pcm_* (wav_* is not
# supported on the streaming endpoint).
SUPPORTED_PCM_RATES = (8000, 16000, 22050, 24000, 44100)


class SpeechifySynthesizer(BaseSynthesizer):
    def __init__(
        self,
        voice_id,
        model="simba-3.2",
        language=None,
        audio_format="pcm",
        sampling_rate="16000",
        stream=False,
        buffer_size=400,
        synthesizer_key=None,
        caching=True,
        **kwargs,
    ):
        super().__init__(kwargs.get("task_manager_instance"), stream, buffer_size)
        self.api_key = os.environ["SPEECHIFY_API_KEY"] if synthesizer_key is None else synthesizer_key
        self.voice_id = voice_id
        self.model = model
        self.language = language
        self.sampling_rate = sampling_rate
        self.caching = caching
        if self.caching:
            self.cache = InmemoryScalarCache()

        # Telephony wants mu-law 8k with no transcode step; anything else is raw PCM
        # at the nearest rate the API supports, resampled to the requested rate.
        self.use_mulaw = kwargs.get("use_mulaw", audio_format == "mulaw")
        if self.use_mulaw:
            self.wire_output_format = "ulaw_8000"
        else:
            rate = int(sampling_rate)
            self.pcm_wire_rate = rate if rate in SUPPORTED_PCM_RATES else 24000
            self.wire_output_format = f"pcm_{self.pcm_wire_rate}"

        self.speechify_host = os.getenv("SPEECHIFY_API_HOST", "api.speechify.ai")
        self.api_url = f"https://{self.speechify_host}/v1/audio/stream"

    def supports_websocket(self):
        return False

    # ------------------------------------------------------------------
    # BaseSynthesizer hooks
    # ------------------------------------------------------------------

    def _get_http_audio_format(self):
        return "mulaw" if self.use_mulaw else "pcm"

    def _process_http_audio(self, audio):
        if self.use_mulaw or audio is None:
            return audio
        return resample(audio, int(self.sampling_rate), format="pcm", original_sample_rate=self.pcm_wire_rate)

    async def _generate_http(self, text):
        payload = {
            "input": text,
            "voice_id": self.voice_id,
            "model": self.model,
            "output_format": self.wire_output_format,
        }
        if self.language:
            payload["language"] = self.language

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            CALLER_HEADER_NAME: CALLER_HEADER_VALUE,
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(self.api_url, headers=headers, json=payload) as response:
                if response.status == 200:
                    return await response.read()
                logger.error(f"Speechify TTS error: {response.status} - {await response.text()}")
                return None

    async def synthesize(self, text):
        return await self._generate_http(text)

    async def synthesize_telephony_clip(self, text):
        """One-shot render in the telephony wire format (mu-law 8000) - no
        decode/transcode step, unlike the resampled PCM synthesize() returns for
        non-mulaw configs. None on non-mulaw configs so callers fall back to
        synthesize() (mirrors ElevenlabsSynthesizer)."""
        if not self.use_mulaw:
            return None
        return await self._generate_http(text)

    # ------------------------------------------------------------------
    # generate / push — HTTP-only, no WebSocket transport for this provider
    # ------------------------------------------------------------------

    async def generate(self):
        async for packet in self._generate_http_loop():
            yield packet
