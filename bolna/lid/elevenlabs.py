import asyncio
import base64
import json
import os

from dotenv import load_dotenv

from bolna.helpers.logger_config import configure_logger

from .base import LIDBackend

load_dotenv()
logger = configure_logger(__name__)


class ElevenLabsLID(LIDBackend):
    """
    LID via ElevenLabs Scribe v2 Realtime WebSocket.

    Streams 8kHz μ-law audio directly — no upsampling needed. Scribe v2 Realtime
    natively accepts ulaw_8000 and returns language_code per committed utterance
    when include_language_detection=true.

    Unlike Azure (batch push stream) this is a true WebSocket where audio chunks
    are sent as base64-encoded JSON messages, mirroring the SarvamLID pattern.

    Config keys (all optional, fall back to env vars):
        elevenlabs_api_key  — ELEVENLABS_API_KEY env var
        telephony_provider  — "twilio" | "plivo" | other
        sampling_rate       — 8000 (telephony default)
        model_id            — scribe_v2_experimental (default)
    """

    _WS_URL = "wss://api.elevenlabs.io/v1/speech-to-text/realtime"
    _WS_URL_IN = "wss://api.in.residency.elevenlabs.io/v1/speech-to-text/realtime"

    def __init__(self, on_language, config):
        super().__init__(on_language, config)
        self._api_key = config.get("elevenlabs_api_key") or os.getenv("ELEVENLABS_API_KEY", "")
        self._telephony = config.get("telephony_provider", "")
        self._sr = int(config.get("sampling_rate", 8000))
        self._model_id = config.get("model_id", "scribe_v2_realtime")
        # India-residency keys must use the India endpoint
        self._ws_url = self._WS_URL_IN if "_residency_in" in self._api_key else self._WS_URL
        # ulaw_8000 is the native telephony format — no conversion needed
        self._audio_format = "ulaw_8000" if self._telephony in ("twilio", "plivo") else "pcm_16000"
        self._queue = asyncio.Queue(maxsize=200)
        self._ws = None
        self._sender_task = None
        self._receiver_task = None
        self._dead = False

    def _build_url(self) -> str:
        params = {
            "model_id": self._model_id,
            "audio_format": self._audio_format,
            "commit_strategy": "vad",
            "include_language_detection": "true",
            "vad_silence_threshold_secs": "0.6",
        }
        qs = "&".join(f"{k}={v}" for k, v in params.items())
        return f"{self._ws_url}?{qs}"

    async def start(self):
        import websockets as ws_lib

        url = self._build_url()
        headers = {"xi-api-key": self._api_key}
        logger.info(f"ElevenLabsLID: connecting — format={self._audio_format!r}, model={self._model_id!r}, url={self._ws_url!r}")
        self._ws = await ws_lib.connect(url, additional_headers=headers)
        self._sender_task = asyncio.create_task(self._sender_loop())
        self._receiver_task = asyncio.create_task(self._receiver_loop())
        logger.info("ElevenLabsLID: connected")

    def feed(self, audio_bytes):
        if self._dead:
            return
        try:
            self._queue.put_nowait(audio_bytes)
        except asyncio.QueueFull:
            logger.debug("ElevenLabsLID: audio queue full — chunk dropped")

    async def _sender_loop(self):
        try:
            while True:
                chunk = await self._queue.get()
                if chunk is None:
                    break
                msg = {
                    "message_type": "input_audio_chunk",
                    "audio_base_64": base64.b64encode(chunk).decode(),
                    "commit": False,
                    "sample_rate": self._sr,
                }
                await self._ws.send(json.dumps(msg))
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"ElevenLabsLID sender error: {e}")
            self._dead = True
            logger.warning("ElevenLabsLID: sender loop exited — LID inactive for remainder of call")

    async def _receiver_loop(self):
        try:
            async for raw in self._ws:
                try:
                    data = json.loads(raw) if isinstance(raw, str) else {}
                    msg_type = data.get("message_type", "")

                    if msg_type == "session_started":
                        logger.info(f"ElevenLabsLID: session started — config={data.get('config', {})}")

                    elif msg_type == "committed_transcript":
                        lang = data.get("language_code") or ""
                        # language_probability not returned in committed_transcript;
                        # use 1.0 as confidence since VAD-committed utterances are reliable.
                        if lang and lang != "und":
                            short = lang.split("-")[0].lower()
                            logger.info(f"ElevenLabsLID: detected {lang!r} (short={short!r})")
                            await self.on_language(short, 1.0)

                    elif msg_type in (
                        "auth_error", "quota_exceeded", "transcriber_error",
                        "rate_limited", "session_time_limit_exceeded", "resource_exhausted",
                    ):
                        logger.warning(f"ElevenLabsLID: fatal error {msg_type!r} — {data.get('error', '')}. LID inactive.")
                        self._dead = True
                        break

                    elif msg_type == "error":
                        logger.warning(f"ElevenLabsLID: error — {data.get('error', '')}")

                except Exception as e:
                    logger.error(f"ElevenLabsLID receiver parse error: {e}")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"ElevenLabsLID receiver error: {e}")
            self._dead = True
            logger.warning("ElevenLabsLID: receiver loop exited — LID inactive for remainder of call")

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
        logger.info("ElevenLabsLID: stopped")
