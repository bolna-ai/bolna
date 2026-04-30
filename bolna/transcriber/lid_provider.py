"""
lid_provider.py — Model-agnostic Language Identification (LID) interface.

Two backends are shipped:

  SarvamLID  — saaras:v3 streaming WebSocket with language_code=unknown.
               Natively streaming; emits language as audio arrives.
               No local model needed, API-based.

  WhisperLID — OpenAI Whisper tiny/base, LID-only mode (encoder + language head,
               no decoder). Trained on diverse audio including phone calls.
               Handles 8kHz mulaw telephony audio well.

Usage (in TranscriberPool):
    lid = LIDProvider.create(provider="whisper", config={...}, on_language=callback)
    await lid.start()
    lid.feed(audio_chunk_bytes)
    await lid.stop()
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import wave
from typing import Awaitable, Callable, Optional

logger = logging.getLogger(__name__)

# Signature: async def on_language(lang: str, confidence: float) -> None
OnLanguageCallback = Callable[[str, float], Awaitable[None]]


# ── 1. Sarvam saaras:v3 (streaming WebSocket) ─────────────────────────────────


class SarvamLID:
    """
    LID via Sarvam saaras:v3 with language_code=unknown.

    Opens a dedicated WebSocket to Sarvam (separate from the ASR connection).
    Audio chunks are forwarded in real-time; the server emits language_code
    in each data payload alongside the transcript.

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

        self._queue: asyncio.Queue = asyncio.Queue()
        self._ws = None
        self._sender_task: Optional[asyncio.Task] = None
        self._receiver_task: Optional[asyncio.Task] = None

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
        self._queue.put_nowait(audio_bytes)

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

    async def _receiver_loop(self) -> None:
        try:
            async for raw in self._ws:
                try:
                    data = json.loads(raw) if isinstance(raw, str) else {}
                    if data.get("type") == "data":
                        payload = data.get("data", {})
                        lang = payload.get("language_code", "")
                        # Sarvam returns language_probability=None in unknown mode;
                        # the language_code itself is the detection signal.
                        conf = float(payload.get("language_probability") or 0.0)
                        if lang and lang != "unknown":
                            short = lang.split("-")[0].lower()
                            logger.info(f"SarvamLID: detected {lang!r} (short={short!r})")
                            await self.on_language(short, conf)
                except Exception as e:
                    logger.error(f"SarvamLID receiver parse error: {e}")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"SarvamLID receiver error: {e}")

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


# ── 2. Whisper LID-only (local, CPU, telephony-friendly) ──────────────────────


class WhisperLID:
    """
    LID via OpenAI Whisper (tiny/base) in language-detection-only mode.

    Runs only the encoder + language detection head — skips the decoder
    entirely so there's no transcription overhead. Whisper was trained on
    diverse audio including phone calls and handles 8kHz mulaw telephony well.

    Uses energy-based VAD gating to skip silence/noise before classifying.

    Requires:  pip install openai-whisper torch

    Config keys:
        model_name         — whisper model size (default: tiny)
        classify_every_ms  — how often to classify after buffer fills (default 800)
        min_buffer_ms      — minimum speech audio before first classify (default 1500)
        vad_rms_threshold  — silence gate RMS threshold (default 500)
        telephony_provider — "twilio" | "plivo" | other
        sampling_rate      — input sample rate (default 8000)
    """

    # Process-level singleton — model loads only once per process
    _model = None
    _model_lock: Optional[asyncio.Lock] = None

    def __init__(self, on_language: OnLanguageCallback, config: dict):
        self.on_language = on_language
        self.config = config
        self._model_name = config.get("model_name", "tiny")
        self._classify_every_ms = int(config.get("classify_every_ms", 800))
        self._min_buffer_ms = int(config.get("min_buffer_ms", 1500))
        self._vad_rms_threshold = int(config.get("vad_rms_threshold", 500))
        self._telephony = config.get("telephony_provider", "")
        self._input_sr = 8000 if self._telephony in ("twilio", "plivo") else int(config.get("sampling_rate", 8000))
        self._encoding = "mulaw" if self._telephony == "twilio" else "linear16"

        self._buffer = bytearray()
        self._buffer_ms = 0
        self._last_classify_ms = 0
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    @classmethod
    def _get_lock(cls) -> asyncio.Lock:
        if cls._model_lock is None:
            cls._model_lock = asyncio.Lock()
        return cls._model_lock

    @classmethod
    async def _load_model(cls, model_name: str):
        async with cls._get_lock():
            if cls._model is None:
                logger.info(f"WhisperLID: loading model whisper-{model_name}...")
                import whisper

                cls._model = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: whisper.load_model(model_name),
                )
                logger.info("WhisperLID: model loaded")
        return cls._model

    def _pcm_to_float(self, pcm_bytes: bytes):
        """Convert raw PCM bytes → float32 numpy array at 16kHz for Whisper."""
        import audioop

        import numpy as np

        raw = pcm_bytes
        if self._encoding == "mulaw":
            raw = audioop.ulaw2lin(raw, 2)
        # Whisper expects 16kHz
        if self._input_sr != 16000:
            raw, _ = audioop.ratecv(raw, 2, 1, self._input_sr, 16000, None)
        return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    def _classify_sync(self, pcm_bytes: bytes) -> tuple[str, float]:
        import whisper

        model = self.__class__._model
        audio = self._pcm_to_float(pcm_bytes)

        # Pad or trim to 30s as Whisper requires, then run encoder only
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(model.device)

        # Detect language using only the encoder + language head (no decoder)
        _, lang_probs = model.detect_language(mel)
        lang = max(lang_probs, key=lang_probs.get)
        conf = lang_probs[lang]
        return lang, float(conf)

    async def start(self) -> None:
        self._loop = asyncio.get_event_loop()
        await self._load_model(self._model_name)
        logger.info("WhisperLID: ready")

    def feed(self, audio_bytes: bytes) -> None:
        """Accept a raw audio chunk; skip silence via energy VAD, classify when buffer fills."""
        import audioop

        # Don't classify until model is fully loaded
        if self.__class__._model is None:
            return

        raw = audio_bytes
        if self._encoding == "mulaw":
            raw = audioop.ulaw2lin(raw, 2)

        # Energy-based VAD gate — skip silent/noise frames
        rms = audioop.rms(raw, 2)
        if rms < self._vad_rms_threshold:
            return

        self._buffer.extend(raw)
        self._buffer_ms = len(self._buffer) * 1000 // (self._input_sr * 2)

        if (
            self._buffer_ms >= self._min_buffer_ms
            and self._buffer_ms - self._last_classify_ms >= self._classify_every_ms
        ):
            self._last_classify_ms = self._buffer_ms
            if self._loop:
                asyncio.run_coroutine_threadsafe(self._classify_and_emit(), self._loop)

    async def _classify_and_emit(self) -> None:
        snapshot = bytes(self._buffer)
        try:
            lang, conf = await asyncio.get_event_loop().run_in_executor(None, self._classify_sync, snapshot)
            logger.info(f"WhisperLID: {lang} conf={conf:.2f} buf={self._buffer_ms}ms")
            await self.on_language(lang, conf)
        except Exception as e:
            logger.warning(f"WhisperLID classify error: {e}")

    async def stop(self) -> None:
        logger.info("WhisperLID: stopped")


# ── Factory ────────────────────────────────────────────────────────────────────


class LIDProvider:
    @classmethod
    def create(cls, provider: str, on_language: OnLanguageCallback, config: dict) -> "SarvamLID | WhisperLID":
        provider = provider.lower()
        if provider == "sarvam":
            return SarvamLID(on_language=on_language, config=config)
        if provider in ("whisper", "whisper-lid", "openai-whisper"):
            return WhisperLID(on_language=on_language, config=config)
        logger.warning(f"LIDProvider: unknown provider '{provider}', falling back to sarvam")
        return SarvamLID(on_language=on_language, config=config)
