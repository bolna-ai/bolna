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
import logging
import os
import wave
from typing import Awaitable, Callable, Optional

logger = logging.getLogger(__name__)

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


# Thin factory shim for backward compatibility
class LIDProvider:
    @classmethod
    def create(cls, provider: str, on_language: OnLanguageCallback, config: dict) -> SarvamLID:
        if provider.lower() != "sarvam":
            logger.warning(f"LIDProvider: unknown provider '{provider}', falling back to sarvam")
        return SarvamLID(on_language=on_language, config=config)
