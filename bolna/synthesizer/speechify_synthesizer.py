import asyncio
import os
from importlib.metadata import PackageNotFoundError, version as _package_version

import httpx
from speechify import AsyncSpeechify
from speechify.core.api_error import ApiError

from .base_synthesizer import BaseSynthesizer
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.utils import resample
from bolna.memory.cache.inmemory_scalar_cache import InmemoryScalarCache

logger = configure_logger(__name__)

# Speechify attribution: the slug names this integration, the version rides its own
# header so usage reports group by integration rather than per release. Both are set
# on the SDK client so every request carries them (the SDK adds no caller of its own).
CALLER = "bolna"
try:
    CALLER_VERSION = _package_version("bolna")
except PackageNotFoundError:
    CALLER_VERSION = "0.0.0"

# output_format sample rates the streaming endpoint accepts for pcm_* (wav_* is not).
SUPPORTED_PCM_RATES = (8000, 16000, 22050, 24000, 44100, 48000)


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

        host = os.getenv("SPEECHIFY_API_HOST")
        # Own the httpx client so its connection pool is closed deterministically
        # in cleanup() rather than leaked when the synthesizer is torn down.
        self._httpx_client = httpx.AsyncClient(timeout=30)
        self.client = AsyncSpeechify(
            token=self.api_key,
            base_url=f"https://{host}" if host else None,
            headers={"Speechify-Caller": CALLER, "Speechify-Caller-Version": CALLER_VERSION},
            httpx_client=self._httpx_client,
        )

    def supports_websocket(self):
        return False

    async def cleanup(self):
        await self._httpx_client.aclose()

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
        params = {
            "input": text,
            "voice_id": self.voice_id,
            "model": self.model,
            "output_format": self.wire_output_format,
        }
        if self.language:
            params["language"] = self.language

        try:
            chunks = bytearray()
            async for chunk in self.client.audio.stream(**params):
                chunks.extend(chunk)
            return bytes(chunks)
        except asyncio.CancelledError:
            # An interruption/shutdown — propagate, never mask as a failed render.
            raise
        except ApiError as e:
            # Classify so failures are diagnosable rather than silently degrading
            # to a click of silence: auth errors are terminal (a retry can't help),
            # every other status may be transient.
            kind = "auth" if e.status_code in (401, 403) else "api"
            logger.error(f"Speechify TTS {kind} error (status={e.status_code}): {e.body}")
            return None
        except Exception as e:
            logger.error(f"Speechify TTS error: {e}")
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
