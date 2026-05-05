"""
lid_provider.py — Language Identification (LID) via Sarvam saaras:v3.

Opens a dedicated WebSocket to Sarvam with language-code=unknown so the
server auto-detects the spoken language and returns language_code in each
data payload alongside the transcript. Audio is forwarded in real-time
from the TranscriberPool audio router — zero added latency to the ASR path.

Usage (in TranscriberPool):
    lid = SarvamLID(on_language=callback, config={...})
    await lid.start()
    lid.feed(audio_chunk_bytes)   # called for every incoming audio packet
    await lid.stop()
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import wave
from typing import Awaitable, Callable, Optional

from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)

# Signature: async def on_language(lang: str, confidence: float) -> None
OnLanguageCallback = Callable[[str, float], Awaitable[None]]


class SarvamLID:
    """
    LID via Sarvam saaras:v3 with language_code=unknown.

    Config keys (all optional, fall back to env vars):
        sarvam_api_key     — SARVAM_API_KEY env var
        sarvam_host        — api.sarvam.ai
        telephony_provider — "twilio" | "plivo" | other
        sampling_rate      — 16000
    """

    _WS_BASE = "wss://{host}/speech-to-text/ws"

    def __init__(self, on_language: OnLanguageCallback, config: dict):
        self.on_language = on_language
        self.config = config
        self._api_key = config.get("sarvam_api_key") or os.getenv("SARVAM_API_KEY", "")
        self._host = config.get("sarvam_host") or os.getenv("SARVAM_HOST", "api.sarvam.ai")
        self._telephony = config.get("telephony_provider", "")
        self._sr = int(config.get("sampling_rate", 16000))
        self._input_sr = 8000 if self._telephony in ("twilio", "plivo") else self._sr
        self._encoding = "mulaw" if self._telephony == "twilio" else "linear16"

        # Bounded queue: LID is best-effort. If the Sarvam WS stalls, we drop
        # chunks rather than buffering unboundedly for the entire call duration.
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=200)
        self._ws = None
        self._sender_task: Optional[asyncio.Task] = None
        self._receiver_task: Optional[asyncio.Task] = None
        # Set to True if the receiver loop exits abnormally (WS drop / error).
        # feed() will log a warning when dead so silent stat bias is visible.
        self._dead: bool = False

    def _build_url(self) -> str:
        params = {
            "model": "saaras:v3",
            "mode": "transcribe",
            "language-code": "unknown",
            "high_vad_sensitivity": "true",
        }
        qs = "&".join(f"{k}={v}" for k, v in params.items())
        return f"{self._WS_BASE.format(host=self._host)}?{qs}"

    def _convert_to_wav_b64(self, raw: bytes) -> Optional[str]:
        """Convert telephony audio to 16kHz WAV base64 for Sarvam."""
        import audioop

        try:
            if self._encoding == "mulaw":
                raw = audioop.ulaw2lin(raw, 2)
            if self._input_sr != self._sr:
                raw, _ = audioop.ratecv(raw, 2, 1, self._input_sr, self._sr, None)
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self._sr)
                wf.writeframes(raw)
            return base64.b64encode(buf.getvalue()).decode()
        except Exception as e:
            logger.warning(f"SarvamLID audio convert error: {e}")
            return None

    async def start(self) -> None:
        import websockets as ws_lib

        url = self._build_url()
        headers = {"api-subscription-key": self._api_key}
        logger.info(f"SarvamLID: connecting to {url}")
        self._ws = await ws_lib.connect(url, additional_headers=headers)
        self._sender_task = asyncio.create_task(self._sender_loop())
        self._receiver_task = asyncio.create_task(self._receiver_loop())
        logger.info("SarvamLID: connected")

    def feed(self, audio_bytes: bytes) -> None:
        if self._dead:
            logger.warning("SarvamLID: feed() called but WS is dead — chunk dropped (LID inactive)")
            return
        try:
            self._queue.put_nowait(audio_bytes)
        except asyncio.QueueFull:
            logger.debug("SarvamLID: audio queue full — chunk dropped (backpressure)")

    async def _sender_loop(self) -> None:
        try:
            while True:
                chunk = await self._queue.get()
                if chunk is None:
                    break
                b64 = self._convert_to_wav_b64(chunk)
                if b64:
                    msg = {"audio": {"data": b64, "encoding": "audio/wav", "sample_rate": self._sr}}
                    await self._ws.send(json.dumps(msg))
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"SarvamLID sender error: {e}")
            self._dead = True
            logger.warning("SarvamLID: sender loop exited abnormally — LID inactive for remainder of call")

    async def _receiver_loop(self) -> None:
        try:
            async for raw in self._ws:
                try:
                    data = json.loads(raw) if isinstance(raw, str) else {}
                    if data.get("type") == "data":
                        payload = data.get("data", {})
                        lang = payload.get("language_code", "")
                        # Sarvam returns language_probability=None when operating in
                        # unknown-language mode — the language_code is the signal.
                        # conf is passed through for API compatibility but the pool's
                        # confidence gate is skipped for Sarvam (see _handle_lid_signal).
                        conf = float(payload.get("language_probability") or 0.0)
                        if lang and lang != "unknown":
                            short = lang.split("-")[0].lower()
                            logger.info(f"SarvamLID: detected {lang!r} (short={short!r}, conf={conf:.2f})")
                            await self.on_language(short, conf)
                except Exception as e:
                    logger.error(f"SarvamLID receiver parse error: {e}")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"SarvamLID receiver error: {e}")
            self._dead = True
            logger.warning("SarvamLID: receiver loop exited abnormally — LID inactive for remainder of call")

    async def stop(self) -> None:
        self._queue.put_nowait(None)
        for task in (self._sender_task, self._receiver_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
        logger.info("SarvamLID: stopped")


# ── AzureLID ──────────────────────────────────────────────────────────────────


class AzureLID:
    """
    LID via Azure Cognitive Services continuous language identification.

    Natively accepts 8kHz mulaw via PushAudioInputStream — no upsampling needed.
    Supports up to 10 candidate languages. Uses existing AZURE_SPEECH_KEY and
    AZURE_SPEECH_REGION env vars (same as azure_transcriber / azure_synthesizer).

    Since Azure SDK does not expose LID confidence scores, utterance duration
    is used as a proxy:
        < 500ms  → 0.60  (needs 2 debounce hits to switch)
        500–1000ms → 0.80
        > 1000ms → 1.00

    Config keys:
        azure_speech_key    — AZURE_SPEECH_KEY env var
        azure_speech_region — AZURE_SPEECH_REGION env var (default: centralindia)
        languages           — list of BCP-47 locales to detect
                              (default: hi-IN, en-IN, ta-IN, te-IN, kn-IN, gu-IN, bn-IN, mr-IN)
        telephony_provider  — "twilio" | "plivo" | other
        sampling_rate       — 8000 (telephony default)
    """

    _DEFAULT_LANGUAGES = ["hi-IN", "en-IN", "ta-IN", "te-IN", "kn-IN", "gu-IN", "bn-IN", "mr-IN"]

    def __init__(self, on_language: OnLanguageCallback, config: dict):
        self.on_language = on_language
        self.config = config
        self._key = config.get("azure_speech_key") or os.getenv("AZURE_SPEECH_KEY", "")
        self._region = config.get("azure_speech_region") or os.getenv("AZURE_SPEECH_REGION", "centralindia")
        self._languages = config.get("languages", self._DEFAULT_LANGUAGES)
        self._telephony = config.get("telephony_provider", "")
        self._encoding = "mulaw" if self._telephony == "twilio" else "linear16"
        self._sr = int(config.get("sampling_rate", 8000))
        self._push_stream = None
        self._recognizer = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._dead: bool = False

    async def start(self) -> None:
        import azure.cognitiveservices.speech as speechsdk
        from azure.cognitiveservices.speech.audio import AudioStreamWaveFormat

        self._loop = asyncio.get_event_loop()

        speech_config = speechsdk.SpeechConfig(subscription=self._key, region=self._region)
        speech_config.set_property(
            property_id=speechsdk.PropertyId.SpeechServiceConnection_LanguageIdMode,
            value="Continuous",
        )

        audio_format = speechsdk.audio.AudioStreamFormat(
            samples_per_second=self._sr,
            bits_per_sample=8 if self._encoding == "mulaw" else 16,
            channels=1,
            wave_stream_format=AudioStreamWaveFormat.MULAW if self._encoding == "mulaw" else AudioStreamWaveFormat.PCM,
        )
        self._push_stream = speechsdk.audio.PushAudioInputStream(audio_format)
        audio_config = speechsdk.audio.AudioConfig(stream=self._push_stream)

        auto_detect_config = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(
            languages=self._languages
        )
        self._recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config,
            audio_config=audio_config,
            auto_detect_source_language_config=auto_detect_config,
        )
        self._recognizer.recognized.connect(self._on_recognized)
        self._recognizer.canceled.connect(self._on_canceled)
        self._recognizer.start_continuous_recognition()
        logger.info(f"AzureLID: started continuous LID for languages={self._languages}")

    @staticmethod
    def _duration_to_conf(duration_ticks: int) -> float:
        duration_ms = duration_ticks / 10_000
        if duration_ms < 500:
            return 0.60
        if duration_ms < 1000:
            return 0.80
        return 1.00

    def _on_recognized(self, evt) -> None:
        if self._loop is None or self._dead:
            return
        try:
            import azure.cognitiveservices.speech as speechsdk
            result = evt.result
            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                lang_result = speechsdk.AutoDetectSourceLanguageResult(result)
                detected = lang_result.language
                if detected and detected != "Unknown":
                    short = detected.split("-")[0].lower()
                    conf = self._duration_to_conf(result.duration)
                    duration_ms = result.duration / 10_000
                    logger.debug(
                        f"AzureLID: detected {detected!r} (short={short!r}, "
                        f"duration={duration_ms:.0f}ms, conf={conf:.2f})"
                    )
                    asyncio.run_coroutine_threadsafe(self.on_language(short, conf), self._loop)
        except Exception as e:
            logger.warning(f"AzureLID recognized callback error: {e}")

    def _on_canceled(self, evt) -> None:
        logger.warning(f"AzureLID: recognition canceled — {evt.reason}. LID inactive.")
        self._dead = True

    def feed(self, audio_bytes: bytes) -> None:
        if self._dead or self._push_stream is None:
            return
        try:
            self._push_stream.write(audio_bytes)
        except Exception as e:
            logger.warning(f"AzureLID feed error: {e}")
            self._dead = True

    async def stop(self) -> None:
        try:
            if self._recognizer:
                self._recognizer.stop_continuous_recognition()
            if self._push_stream:
                self._push_stream.close()
        except Exception as e:
            logger.warning(f"AzureLID stop error: {e}")
        logger.info("AzureLID: stopped")


# ── Factory ────────────────────────────────────────────────────────────────────


class LIDProvider:
    @classmethod
    def create(
        cls, provider: str, on_language: OnLanguageCallback, config: dict
    ) -> "SarvamLID | AzureLID":
        p = provider.lower()
        if p == "sarvam":
            return SarvamLID(on_language=on_language, config=config)
        if p in ("azure", "azure-lid", "azurelid"):
            return AzureLID(on_language=on_language, config=config)
        logger.warning(f"LIDProvider: unknown provider '{provider}', falling back to azure")
        return AzureLID(on_language=on_language, config=config)
