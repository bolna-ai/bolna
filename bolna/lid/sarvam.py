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
        # Target rate Saaras receives (it wants 16k). Per-provider INPUT rate/encoding
        # must mirror sarvam_transcriber._configure_audio_params — telephony providers
        # stream 8kHz, so we resample 8k→16k. Getting this wrong (e.g. treating vobiz's
        # 8kHz as 16kHz) feeds garbled audio to Saaras → no transcripts → no detection.
        self._sr = int(config.get("sampling_rate", 16000))
        if self._telephony in ("plivo", "vobiz", "exotel"):
            self._encoding = "linear16"
            self._input_sr = 8000
        elif self._telephony == "twilio":
            self._encoding = "mulaw"
            self._input_sr = 8000
        else:
            self._encoding = "linear16"
            self._input_sr = self._sr
        self._resample_state = None

    def _on_reconnect_reset(self):
        self._resample_state = None

    def _build_url(self) -> str:
        params = {
            "model": "saaras:v3",
            "mode": "transcribe",
            "language-code": "unknown",
            "high_vad_sensitivity": "true",
            # Matches the working sarvam_transcriber config. We read transcripts from
            # the "data" messages and buffer them; VAD events are not relied upon.
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
            self._schedule_reconnect("sender error")

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

                        # saaras emits one "data" message per VAD segment (multiple per
                        # spoken turn). Accumulate into the rolling buffer; the caller
                        # drains it once per conversational turn via take_turn_transcript().
                        seg_prob = float(lang_prob) if lang_prob is not None else None
                        self._accumulate(transcript, short, audio_duration, seg_prob)

                        # Legacy per-segment language signal (kept for backends/telemetry
                        # that still consume on_language). Skipped when no confidence.
                        if self.on_language is not None and lang_prob is not None and short:
                            conf = float(lang_prob)
                            logger.info(
                                f"SarvamLID: detected {lang!r} (short={short!r}, duration={audio_duration:.2f}s, conf={conf:.2f})"
                            )
                            asyncio.create_task(self.on_language(short, conf))
                except Exception as e:
                    logger.error(f"SarvamLID receiver parse error: {e}")
        except asyncio.CancelledError:
            return
        except Exception as e:
            logger.error(f"SarvamLID receiver error: {e}")
        # Reached on BOTH a server-side graceful close (the async-for simply ends,
        # no exception raised) and receive errors. The graceful case previously
        # exited silently — detector mute for the rest of the call with zero log
        # lines (QA call edbdb998: 0 segments in 21s, Tamil never detected).
        self._schedule_reconnect("receiver closed")

    async def stop(self):
        await self._shutdown_connection()
