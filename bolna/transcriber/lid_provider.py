"""
lid_provider.py — Model-agnostic Language Identification (LID) interface.

Provides a common LIDProvider ABC that any LID backend implements.
Two backends are shipped:

  SarvamLID     — saaras:v3 streaming WebSocket with language_code=unknown.
                  Natively streaming; emits language as audio arrives.
                  100% accuracy on 8kHz telephony audio (benchmark result).

  VoxLinguaLID  — SpeechBrain VoxLingua107 ECAPA-TDNN, local CPU model.
                  360 MB, no API key, 90% accuracy, ~274ms inference.
                  Accumulates audio in a rolling buffer; classify on each chunk.

Usage (in TranscriberPool):
    lid = LIDProvider.create(provider="sarvam", config={...}, on_language=callback)
    await lid.start()
    lid.feed(audio_chunk_bytes)   # called for every incoming audio packet
    await lid.stop()
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import time
import wave
import io
from abc import ABC, abstractmethod
from typing import Awaitable, Callable, Optional

logger = logging.getLogger(__name__)

# Signature: async def on_language(lang: str, confidence: float) -> None
OnLanguageCallback = Callable[[str, float], Awaitable[None]]


# ── Base class ─────────────────────────────────────────────────────────────────

class LIDProvider(ABC):
    """Abstract base for all LID backends."""

    def __init__(self, on_language: OnLanguageCallback, config: dict):
        self.on_language = on_language
        self.config = config

    @abstractmethod
    async def start(self) -> None:
        """Connect / load model. Called once before feeding audio."""

    @abstractmethod
    def feed(self, audio_bytes: bytes) -> None:
        """Accept a raw PCM audio chunk (linear16, 8kHz or 16kHz)."""

    @abstractmethod
    async def stop(self) -> None:
        """Disconnect / release resources."""

    @classmethod
    def create(
        cls,
        provider: str,
        on_language: OnLanguageCallback,
        config: dict,
    ) -> "LIDProvider":
        """Factory — returns the correct backend by name."""
        provider = provider.lower()
        if provider == "sarvam":
            return SarvamLID(on_language=on_language, config=config)
        if provider in ("voxlingua", "speechbrain"):
            return VoxLinguaLID(on_language=on_language, config=config)
        raise ValueError(
            f"Unknown LID provider '{provider}'. "
            "Available: 'sarvam', 'voxlingua'"
        )


# ── 1. Sarvam saaras:v3 (streaming WebSocket) ─────────────────────────────────

class SarvamLID(LIDProvider):
    """
    LID via Sarvam saaras:v3 with language_code=unknown.

    Opens a dedicated WebSocket to Sarvam (separate from the ASR connection).
    Audio chunks are forwarded in real-time; the server emits language_code
    in each data payload alongside the transcript.

    Config keys (all optional, fall back to env):
        sarvam_api_key    — SARVAM_API_KEY env var
        sarvam_host       — api.sarvam.ai
        telephony_provider — "twilio" | "plivo" | other
        sampling_rate     — 16000
    """

    _WS_BASE = "wss://{host}/speech-to-text/ws"

    def __init__(self, on_language: OnLanguageCallback, config: dict):
        super().__init__(on_language, config)
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
        host = self._host
        params = {
            "model": "saaras:v3",
            "mode": "transcribe",
            "language-code": "unknown",   # triggers auto-LID
            "high_vad_sensitivity": "true",
        }
        qs = "&".join(f"{k}={v}" for k, v in params.items())
        return f"{self._WS_BASE.format(host=host)}?{qs}"

    def _convert_to_wav_b64(self, raw: bytes) -> Optional[str]:
        """Convert incoming telephony audio to 16kHz WAV base64."""
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
                    logger.info(f"SarvamLID raw response: {data}")
                    if data.get("type") == "data":
                        payload = data.get("data", {})
                        lang = payload.get("language_code", "")
                        conf = float(payload.get("language_probability") or 0.0)
                        logger.info(f"SarvamLID detected: lang={lang!r} conf={conf:.2f}")
                        if lang and lang != "unknown":
                            # Normalise to 2-letter ISO code (hi-IN → hi)
                            short = lang.split("-")[0].lower()
                            logger.info(f"SarvamLID emitting: short={short!r} conf={conf:.2f}")
                            await self.on_language(short, conf)
                except Exception as e:
                    logger.error(f"SarvamLID receiver parse error: {e} raw={raw!r}")
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


# ── 2. SpeechBrain VoxLingua107 (local, CPU) ──────────────────────────────────

class VoxLinguaLID(LIDProvider):
    """
    LID via SpeechBrain VoxLingua107 ECAPA-TDNN (local, no API key).

    Audio chunks are accumulated in a rolling buffer. Every
    `classify_every_ms` milliseconds (default 500ms) the buffer is classified.
    The model is loaded once and cached for the lifetime of the process.

    Requires:  pip install speechbrain soundfile torch torchaudio

    Config keys:
        classify_every_ms  — how often to run classification (default 500)
        min_buffer_ms      — minimum audio before first classify (default 800)
        model_save_dir     — where to cache the HF model (default models/voxlingua)
    """

    # Process-level singleton so the 360MB model loads only once
    _model = None
    _model_lock = asyncio.Lock()

    def __init__(self, on_language: OnLanguageCallback, config: dict):
        super().__init__(on_language, config)
        self._classify_every_ms = int(config.get("classify_every_ms", 500))
        self._min_buffer_ms = int(config.get("min_buffer_ms", 800))
        self._model_dir = config.get("model_save_dir", "models/voxlingua")
        self._input_sr = int(config.get("sampling_rate", 8000))
        self._encoding = config.get("encoding", "linear16")
        self._telephony = config.get("telephony_provider", "")
        if self._telephony == "twilio":
            self._encoding = "mulaw"
            self._input_sr = 8000

        self._buffer = bytearray()          # raw linear16 PCM at input_sr
        self._buffer_ms = 0
        self._last_classify_ms = 0
        self._classify_task: Optional[asyncio.Task] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    @classmethod
    async def _load_model(cls, save_dir: str):
        async with cls._model_lock:
            if cls._model is None:
                logger.info("VoxLinguaLID: loading model (first time, ~360MB)…")
                from speechbrain.inference.classifiers import EncoderClassifier
                cls._model = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: EncoderClassifier.from_hparams(
                        source="TalTechNLP/voxlingua107-epaca-tdnn",
                        savedir=save_dir,
                        run_opts={"device": "cpu"},
                    ),
                )
                logger.info("VoxLinguaLID: model loaded")
        return cls._model

    def _pcm_to_tensor(self, pcm_bytes: bytes):
        """Convert raw linear16 PCM bytes → float32 tensor at 16kHz."""
        import torch, torchaudio, audioop
        raw = pcm_bytes
        if self._encoding == "mulaw":
            raw = audioop.ulaw2lin(raw, 2)
        # Resample to 16kHz for the model
        if self._input_sr != 16000:
            raw, _ = audioop.ratecv(raw, 2, 1, self._input_sr, 16000, None)
        import numpy as np
        arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        return torch.from_numpy(arr).unsqueeze(0)   # [1, T]

    def _classify_sync(self, pcm_bytes: bytes) -> tuple[str, float]:
        import torch
        model = self.__class__._model
        sig = self._pcm_to_tensor(pcm_bytes)
        pred = model.classify_batch(sig)
        label = pred[3][0]                          # "hi: Hindi"
        scores = pred[1].squeeze()
        conf = float(torch.exp(scores.max()))
        lang = label.split(":")[0].strip().lower()[:2]
        return lang, conf

    async def start(self) -> None:
        self._loop = asyncio.get_event_loop()
        await self._load_model(self._model_dir)
        logger.info("VoxLinguaLID: ready")

    def feed(self, audio_bytes: bytes) -> None:
        """Accept a raw audio chunk and classify when enough data has accumulated."""
        import audioop
        raw = audio_bytes
        if self._encoding == "mulaw":
            raw = audioop.ulaw2lin(raw, 2)
        self._buffer.extend(raw)
        # Each sample = 2 bytes at input_sr → ms of audio in buffer
        self._buffer_ms = len(self._buffer) * 1000 // (self._input_sr * 2)

        if (self._buffer_ms >= self._min_buffer_ms and
                self._buffer_ms - self._last_classify_ms >= self._classify_every_ms):
            self._last_classify_ms = self._buffer_ms
            # Fire-and-forget classify in executor so we don't block the event loop
            if self._loop:
                asyncio.run_coroutine_threadsafe(self._classify_and_emit(), self._loop)

    async def _classify_and_emit(self) -> None:
        snapshot = bytes(self._buffer)
        try:
            lang, conf = await asyncio.get_event_loop().run_in_executor(
                None, self._classify_sync, snapshot
            )
            logger.debug(f"VoxLinguaLID: {lang} conf={conf:.2f} buf={self._buffer_ms}ms")
            await self.on_language(lang, conf)
        except Exception as e:
            logger.warning(f"VoxLinguaLID classify error: {e}")

    async def stop(self) -> None:
        if self._classify_task and not self._classify_task.done():
            self._classify_task.cancel()
        logger.info("VoxLinguaLID: stopped")
