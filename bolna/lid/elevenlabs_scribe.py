import asyncio
import base64
import json
import os

from dotenv import load_dotenv

from bolna.helpers.logger_config import configure_logger

from .base import LIDBackend

load_dotenv()
logger = configure_logger(__name__)


class ElevenLabsScribeLID(LIDBackend):
    """
    LID via ElevenLabs Scribe v2 Realtime WebSocket.

    Streams raw PCM audio; ElevenLabs returns language_code on each
    committed_transcript_with_timestamps event.  No confidence score is
    available in streaming — on_language is called with confidence=None.

    Config keys (all optional, fall back to env vars):
        elevenlabs_api_key  — ELEVENLABS_API_KEY env var
        elevenlabs_host     — ELEVENLABS_HOST env var (default api.elevenlabs.io)
        telephony_provider  — "twilio" | "plivo" | other
        sampling_rate       — 16000
    """

    _WS_PATH = "/v1/speech-to-text/realtime"

    def __init__(self, on_language, config):
        super().__init__(on_language, config)
        self.api_key = config.get("elevenlabs_api_key") or os.getenv("ELEVENLABS_API_KEY", "")
        self.telephony = config.get("telephony_provider", "")
        self.sr = int(config.get("sampling_rate", 16000))
        self.host = config.get("elevenlabs_host") or os.getenv("ELEVENLABS_HOST", "api.elevenlabs.io")

        # Map to ElevenLabs audio_format query param
        if self.telephony == "twilio":
            self.audio_format = "ulaw_8000"
        elif self.telephony == "plivo":
            self.audio_format = "pcm_8000"
        else:
            self.audio_format = f"pcm_{self.sr}"

        # How many raw bytes = 500ms of audio for this format, used for commit pacing.
        # ElevenLabs requires ≥300ms before commit — 500ms gives the model more context
        # per segment for better language identification accuracy.
        # ulaw_8000: 1 byte/sample × 8000Hz × 0.5s = 4000 bytes
        # pcm_8000:  2 bytes/sample × 8000Hz × 0.5s = 8000 bytes
        # pcm_16000: 2 bytes/sample × 16000Hz × 0.5s = 16000 bytes
        if self.audio_format == "ulaw_8000":
            self.commit_threshold_bytes = 4000
        elif self.audio_format == "pcm_8000":
            self.commit_threshold_bytes = 8000
        else:
            self.commit_threshold_bytes = int(2 * self.sr * 0.500)

        self.queue = asyncio.Queue(maxsize=20)
        self.ws = None
        self.sender_task = None
        self.receiver_task = None
        self.dead = False
        # Set once session_started is received — sender waits on this before
        # sending any audio to avoid "Message must be a valid protocol message"
        self.session_ready = asyncio.Event()

    def _build_url(self) -> str:
        params = {
            "model_id": "scribe_v2_realtime",
            "audio_format": self.audio_format,
            "commit_strategy": "manual",
            "include_timestamps": "true",
            "include_language_detection": "true",
        }
        qs = "&".join(f"{k}={v}" for k, v in params.items())
        return f"wss://{self.host}{self._WS_PATH}?{qs}"

    async def start(self):
        import websockets as ws_lib

        url = self._build_url()
        headers = {"xi-api-key": self.api_key}
        logger.info(f"ElevenLabsScribeLID: connecting to {url}")
        self.ws = await ws_lib.connect(url, additional_headers=headers)
        self.sender_task = asyncio.create_task(self.sender_loop())
        self.receiver_task = asyncio.create_task(self.receiver_loop())
        logger.info("ElevenLabsScribeLID: connected")

    def feed(self, audio_bytes):
        if self.dead:
            logger.warning("ElevenLabsScribeLID: feed() called but WS is dead — chunk dropped")
            return
        try:
            self.queue.put_nowait(audio_bytes)
        except asyncio.QueueFull:
            logger.debug("ElevenLabsScribeLID: audio queue full — chunk dropped (backpressure)")

    async def sender_loop(self):
        import websockets as ws_lib

        try:
            # Wait for session_started before sending any audio
            await asyncio.wait_for(self.session_ready.wait(), timeout=10.0)
            logger.info("ElevenLabsScribeLID: session ready — starting audio stream")

            bytes_since_commit = 0
            while True:
                chunk = await self.queue.get()
                if chunk is None:
                    break
                b64 = base64.b64encode(chunk).decode()
                bytes_since_commit += len(chunk)
                # Commit every ~500ms worth of audio — ElevenLabs requires ≥300ms before commit,
                # and larger windows give the model more context per segment.
                # Byte-based accumulation means commit cadence is correct regardless of
                # incoming chunk size (100ms from Twilio, 80ms from SIP, 100ms in tests).
                should_commit = bytes_since_commit >= self.commit_threshold_bytes
                await self.ws.send(
                    json.dumps(
                        {
                            "message_type": "input_audio_chunk",
                            "audio_base_64": b64,
                            "commit": should_commit,
                            "sample_rate": self.sr,
                            "previous_text": None,
                        }
                    )
                )
                if should_commit:
                    logger.debug(f"ElevenLabsScribeLID: committed segment after {bytes_since_commit} bytes")
                    bytes_since_commit = 0

            # Flush any pending audio — force VAD commit on whatever is buffered
            logger.info("ElevenLabsScribeLID: sending final commit flush")
            await self.ws.send(
                json.dumps(
                    {
                        "message_type": "input_audio_chunk",
                        "audio_base_64": "",
                        "commit": True,
                        "sample_rate": self.sr,
                    }
                )
            )
        except asyncio.TimeoutError:
            logger.error("ElevenLabsScribeLID: timed out waiting for session_started")
            self.dead = True
        except asyncio.CancelledError:
            pass
        except ws_lib.ConnectionClosedOK:
            logger.info("ElevenLabsScribeLID: sender loop: connection closed cleanly")
        except Exception as e:
            logger.error(f"ElevenLabsScribeLID sender error: {e}")
            self.dead = True
            logger.warning("ElevenLabsScribeLID: sender loop exited abnormally — LID inactive for remainder of call")

    async def receiver_loop(self):
        import websockets as ws_lib

        try:
            async for raw in self.ws:
                try:
                    logger.info(f"ElevenLabsScribeLID raw: {raw}")
                    data = json.loads(raw) if isinstance(raw, str) else {}
                    msg_type = data.get("message_type", "")

                    if msg_type == "session_started":
                        logger.info(f"ElevenLabsScribeLID: session_started session_id={data.get('session_id')}")
                        self.session_ready.set()

                    elif msg_type == "committed_transcript_with_timestamps":
                        text = data.get("text", "").strip()
                        lang = data.get("language_code") or ""
                        logger.info(f"ElevenLabsScribeLID: committed transcript text={text!r} language_code={lang!r}")
                        if lang:
                            # Normalise to short ISO-639-1 (e.g. "eng" → "en", "hin" → "hi")
                            short = lang[:2].lower()
                            logger.info(f"ElevenLabsScribeLID: detected {lang!r} (short={short!r})")
                            await self.on_language(short, None)

                    elif msg_type == "commit_throttled":
                        logger.warning(f"ElevenLabsScribeLID commit throttled: {data.get('error')}")

                    elif msg_type == "input_error":
                        logger.warning(f"ElevenLabsScribeLID input error: {data.get('error')}")

                    elif msg_type == "transcriber_error":
                        logger.error(f"ElevenLabsScribeLID transcriber error: {data.get('error')}")

                    elif msg_type == "error":
                        logger.error(f"ElevenLabsScribeLID error: {data.get('error')}")

                except Exception as e:
                    logger.error(f"ElevenLabsScribeLID receiver parse error: {e}")
        except asyncio.CancelledError:
            pass
        except ws_lib.ConnectionClosedOK:
            logger.info("ElevenLabsScribeLID: receiver loop: connection closed cleanly")
        except Exception as e:
            logger.error(f"ElevenLabsScribeLID receiver error: {e}")
            self.dead = True
            logger.warning("ElevenLabsScribeLID: receiver loop exited abnormally — LID inactive for remainder of call")

    async def stop(self):
        try:
            self.queue.put_nowait(None)
        except asyncio.QueueFull:
            while not self.queue.empty():
                try:
                    self.queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            self.queue.put_nowait(None)
        for task in (self.sender_task, self.receiver_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        if self.ws:
            try:
                await self.ws.close()
            except Exception:
                pass
        logger.info("ElevenLabsScribeLID: stopped")
