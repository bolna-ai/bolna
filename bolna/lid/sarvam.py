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

    def __init__(self, on_language, config, on_turn=None):
        super().__init__(on_language, config, on_turn)
        self._api_key = config.get("sarvam_api_key") or os.getenv("SARVAM_API_KEY", "")
        self._host = config.get("sarvam_host") or os.getenv("SARVAM_HOST", "api.sarvam.ai")
        self._telephony = config.get("telephony_provider", "")
        self._sr = int(config.get("sampling_rate", 16000))
        self._input_sr = 8000 if self._telephony in ("twilio", "plivo") else self._sr
        self._encoding = "mulaw" if self._telephony == "twilio" else "linear16"
        self._queue = asyncio.Queue(maxsize=20)
        self._ws = None
        self._sender_task = None
        self._receiver_task = None
        self._dead = False
        self._resample_state = None
        # Per-turn accumulation for the on_turn callback: Sarvam delivers a turn's
        # text across one or more "data" segments, then a VAD END_SPEECH event.
        self._turn_transcript = ""
        self._turn_detected_lang = None

    def _reset_turn_state(self):
        """Clear the per-turn transcript accumulator (on speech start and after a turn is emitted)."""
        self._turn_transcript = ""
        self._turn_detected_lang = None

    def _build_url(self) -> str:
        params = {
            "model": "saaras:v3",
            "mode": "transcribe",
            "language-code": "unknown",
            "high_vad_sensitivity": "true",
            # Emit START_SPEECH / END_SPEECH events so we can deliver a full-turn
            # transcript to on_turn at end-of-speech (not per-segment).
            "vad_signals": "true",
        }
        qs = "&".join(f"{k}={v}" for k, v in params.items())
        return f"{self._WS_BASE.format(host=self._host)}?{qs}"

    def _convert_to_wav_b64(self, raw):
        try:
            if self._encoding == "mulaw":
                raw = audioop.ulaw2lin(raw, 2)
            if self._input_sr != self._sr:
                raw, self._resample_state = audioop.ratecv(raw, 2, 1, self._input_sr, self._sr, self._resample_state)
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
                        transcript = (payload.get("transcript") or "").strip()
                        lang = payload.get("language_code", "")
                        lang_prob = payload.get("language_probability")
                        metrics = payload.get("metrics") or {}
                        audio_duration = float(metrics.get("audio_duration") or 0.0)
                        processing_latency = float(metrics.get("processing_latency") or 0.0)
                        logger.info(
                            f"SarvamLID raw: lang={lang!r} language_probability={lang_prob} "
                            f"transcript={transcript[:60]!r} audio_duration={audio_duration:.3f}s "
                            f"processing_latency={processing_latency:.3f}s"
                        )

                        short = lang.split("-")[0].lower() if lang and lang != "unknown" else None

                        # Accumulate this turn's transcript + latest detected language
                        # for the on_turn callback fired at END_SPEECH.
                        if self.on_turn is not None:
                            if transcript:
                                self._turn_transcript = " ".join(
                                    filter(None, [self._turn_transcript, transcript])
                                )
                            if short:
                                self._turn_detected_lang = short

                        # Legacy per-segment language signal (kept for backends/telemetry
                        # that still consume on_language). Skipped when no confidence.
                        if self.on_language is not None and lang_prob is not None and short:
                            conf = float(lang_prob)
                            logger.info(
                                f"SarvamLID: detected {lang!r} (short={short!r}, duration={audio_duration:.2f}s, conf={conf:.2f})"
                            )
                            asyncio.create_task(self.on_language(short, conf))

                    elif data.get("type") == "events":
                        vad = data.get("data", {})
                        signal = vad.get("signal_type")
                        if signal == "START_SPEECH":
                            self._reset_turn_state()
                        elif signal == "END_SPEECH" and self.on_turn is not None:
                            turn_text = self._turn_transcript.strip()
                            detected_lang = self._turn_detected_lang
                            self._reset_turn_state()
                            if turn_text:
                                logger.info(
                                    f"SarvamLID turn: transcript={turn_text[:80]!r} detected_lang={detected_lang!r}"
                                )
                                asyncio.create_task(self.on_turn(turn_text, detected_lang))
                except Exception as e:
                    logger.error(f"SarvamLID receiver parse error: {e}")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"SarvamLID receiver error: {e}")
            self._dead = True
            logger.warning("SarvamLID: receiver loop exited abnormally — LID inactive for remainder of call")

    async def stop(self):
        try:
            self._queue.put_nowait(None)
        except asyncio.QueueFull:
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
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
