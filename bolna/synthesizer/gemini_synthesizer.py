"""Google Gemini TTS over the Vertex generateContent endpoint.

Non-streaming HTTP, rendered one-shot like the OpenAI path. Style goes in structured
`speech_metadata`; inline `<tags>` stay in the text. Audio returns as base64 PCM in inlineData.
"""

import asyncio
import base64
import os
import re

import aiohttp

from .base_synthesizer import BaseSynthesizer
from bolna.constants import GEMINI_TTS_DEFAULT_MODEL, GEMINI_TTS_LOCATION, GEMINI_TTS_NATIVE_SAMPLE_RATE
from bolna.helpers.gcp_auth import get_gcp_credentials
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.utils import pcm_to_ulaw, pcm_to_wav_bytes, resample
from bolna.memory.cache.inmemory_scalar_cache import InmemoryScalarCache

logger = configure_logger(__name__)

MULAW_SAMPLE_RATE = 8000
_VERTEX_HOST = "https://aiplatform.googleapis.com"
_MIME_RATE = re.compile(r"rate=(\d+)")
_REQUEST_TIMEOUT_SECONDS = 20


class GeminiSynthesizer(BaseSynthesizer):
    def __init__(
        self,
        voice="Puck",
        voice_id=None,
        model=GEMINI_TTS_DEFAULT_MODEL,
        language="en",
        style=None,
        sampling_rate="24000",
        stream=False,
        buffer_size=400,
        caching=True,
        **kwargs,
    ):
        super().__init__(kwargs.get("task_manager_instance"), stream, buffer_size)
        # voice_id is the Gemini voice_name; voice is only the label. Never None.
        self.voice = voice_id or voice
        self.model = model
        self.language = language
        self.style = style or None
        self.stream = False

        # Auth is ADC, not a synthesizer_key; the project is stamped in x-goog-user-project.
        self.project = kwargs.get("project") or os.getenv("GEMINI_TTS_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")

        self.use_mulaw = kwargs.get("use_mulaw", False)
        self.target_sample_rate = MULAW_SAMPLE_RATE if self.use_mulaw else int(sampling_rate)
        self.sampling_rate = self.target_sample_rate

        self.caching = caching
        if caching:
            self.cache = InmemoryScalarCache()

    def _endpoint(self, project):
        return (
            f"{_VERTEX_HOST}/v1beta1/projects/{project}/locations/{GEMINI_TTS_LOCATION}"
            f"/publishers/google/models/{self.model}:generateContent"
        )

    def _build_payload(self, text):
        part = {"text": text}
        if self.style:
            part["speech_metadata"] = {"style": self.style}
        return {
            # role is required; the endpoint 400s ("valid role: user, model") without it.
            "contents": [{"role": "user", "parts": [part]}],
            "generation_config": {
                "response_modalities": ["AUDIO"],
                "speech_config": {"voice_config": {"prebuilt_voice_config": {"voice_name": self.voice}}},
            },
        }

    @staticmethod
    def _extract_audio(response_json):
        """First inlineData part as (pcm_bytes, sample_rate), or (None, None)."""
        for candidate in response_json.get("candidates", []):
            for part in (candidate.get("content") or {}).get("parts", []):
                inline = part.get("inlineData") or part.get("inline_data")
                if inline and inline.get("data"):
                    pcm = base64.b64decode(inline["data"])
                    mime = inline.get("mimeType") or inline.get("mime_type") or ""
                    match = _MIME_RATE.search(mime)
                    rate = int(match.group(1)) if match else GEMINI_TTS_NATIVE_SAMPLE_RATE
                    return pcm, rate
        return None, None

    async def _fetch_pcm(self, text):
        """One generateContent call. Returns raw PCM at Gemini's native rate, or (None, None)."""
        # Off the event loop, and bounded like the audio POST so a stalled mint ends the turn.
        try:
            token, adc_project = await asyncio.wait_for(get_gcp_credentials(), timeout=_REQUEST_TIMEOUT_SECONDS)
        except Exception as e:
            logger.error(f"Gemini TTS: could not obtain GCP credentials: {e}")
            return None, None

        project = self.project or adc_project
        if not project:
            logger.error("Gemini TTS: no GCP project configured (set GEMINI_TTS_PROJECT)")
            return None, None

        headers = {
            "Authorization": f"Bearer {token}",
            "x-goog-user-project": project,
            "Content-Type": "application/json",
        }
        timeout = aiohttp.ClientTimeout(total=_REQUEST_TIMEOUT_SECONDS)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    self._endpoint(project), headers=headers, json=self._build_payload(text)
                ) as response:
                    if response.status != 200:
                        logger.error(f"Gemini TTS HTTP error: {response.status} - {await response.text()}")
                        return None, None
                    data = await response.json()
        except Exception as e:
            logger.error(f"Gemini TTS request failed: {e}")
            return None, None
        return self._extract_audio(data)

    def _to_target(self, pcm, native_rate):
        """Resample to the target rate; mu-law encode for telephony."""
        if not pcm:
            return None
        try:
            audio = resample(pcm, self.target_sample_rate, format="pcm", original_sample_rate=native_rate)
        except Exception as e:
            logger.error(f"Error resampling Gemini audio: {e}")
            return None
        return pcm_to_ulaw(audio) if self.use_mulaw else audio

    def _get_http_audio_format(self):
        return "mulaw" if self.use_mulaw else "pcm"

    async def _generate_http(self, text):
        pcm, native_rate = await self._fetch_pcm(text)
        return self._to_target(pcm, native_rate)

    async def generate(self):
        async for packet in self._generate_http_loop():
            yield packet

    async def synthesize(self, text):
        """WAV-wrapped at the native rate so audio_to_mulaw8k()'s guessed rate_hint can't misread it."""
        pcm, native_rate = await self._fetch_pcm(text)
        if not pcm:
            return None
        return pcm_to_wav_bytes(pcm, sample_rate=native_rate)
