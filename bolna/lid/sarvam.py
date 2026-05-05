import asyncio
import audioop
import base64
import io
import json
import os
import wave

from dotenv import load_dotenv

from bolna.helpers.logger_config import configure_logger

from .base import LIDBackend

load_dotenv()
logger = configure_logger(__name__)


class SarvamLID(LIDBackend):
    """
    LID via Sarvam saaras:v3 with language_code=unknown.

    Streams audio over a persistent WebSocket; Sarvam returns language_code
    inline per utterance. Zero added latency to the ASR path.

    Config keys (all optional, fall back to env vars):
        sarvam_api_key     — SARVAM_API_KEY env var
        sarvam_host        — api.sarvam.ai
        telephony_provider — "twilio" | "plivo" | other
        sampling_rate      — 16000
    """

    _WS_BASE = "wss://{host}/speech-to-text/ws"

    def __init__(self, on_language, config):
        super().__init__(on_language, config)
        self._api_key = config.get("sarvam_api_key") or os.getenv("SARVAM_API_KEY", "")
        self._host = config.get("sarvam_host") or os.getenv("SARVAM_HOST", "api.sarvam.ai")
        self._telephony = config.get("telephony_provider", "")
        self._sr = int(config.get("sampling_rate", 16000))
        self._input_sr = 8000 if self._telephony in ("twilio", "plivo") else self._sr
        self._encoding = "mulaw" if self._telephony == "twilio" else "linear16"
        self._queue = asyncio.Queue(maxsize=200)
        self._ws = None
        self._sender_task = None
        self._receiver_task = None
        self._dead = False

    def _build_url(self) -> str:
        params = {
            "model": "saaras:v3",
            "mode": "transcribe",
            "language-code": "unknown",
            "high_vad_sensitivity": "true",
        }
        qs = "&".join(f"{k}={v}" for k, v in params.items())
        return f"{self._WS_BASE.format(host=self._host)}?{qs}"

    def _convert_to_wav_b64(self, raw):
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

    async def start(self):
        import websockets as ws_lib

        url = self._build_url()
        headers = {"api-subscription-key": self._api_key}
        logger.info(f"SarvamLID: connecting to {url}")
        self._ws = await ws_lib.connect(url, additional_headers=headers)
        self._sender_task = asyncio.create_task(self._sender_loop())
        self._receiver_task = asyncio.create_task(self._receiver_loop())
        logger.info("SarvamLID: connected")

    def feed(self, audio_bytes):
        if self._dead:
            logger.warning("SarvamLID: feed() called but WS is dead — chunk dropped (LID inactive)")
            return
        try:
            self._queue.put_nowait(audio_bytes)
        except asyncio.QueueFull:
            logger.debug("SarvamLID: audio queue full — chunk dropped (backpressure)")

    async def _sender_loop(self):
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

    async def _receiver_loop(self):
        try:
            async for raw in self._ws:
                try:
                    data = json.loads(raw) if isinstance(raw, str) else {}
                    if data.get("type") == "data":
                        payload = data.get("data", {})
                        lang = payload.get("language_code", "")
                        # Sarvam returns language_probability=None in unknown-language mode.
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

    async def stop(self):
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
