import asyncio
import audioop
import base64
import io
import json
import os
import time
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

    # Mid-call reconnects allowed when saaras closes the socket on us (observed
    # in QA: server-side graceful close left the detector silently mute for the
    # whole call). Bounded so a rejecting server can't loop us.
    _MAX_RECONNECTS = 2
    _RECONNECT_DELAY_S = 0.5

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
        self._queue = asyncio.Queue(maxsize=20)
        self._ws = None
        self._sender_task = None
        self._receiver_task = None
        self._dead = False
        self._stopping = False
        self._reconnecting = False
        self._reconnect_attempts = 0
        self._dead_drop_logged = False
        self._resample_state = None
        # Rolling buffer of unbiased recognition for the current conversational turn.
        # saaras emits one "data" message per VAD segment (several per turn), so we
        # accumulate here and let the caller drain it once per turn via
        # take_turn_transcript() — aligned to the main transcriber's turn boundary.
        self._buffer_text = ""
        self._buffer_lang = None
        self._buffer_lang_streak = 0
        self._buffer_max_segment_s = 0.0
        self._buffer_last_segment_ts = None
        # Set whenever a segment lands; cleared on drain. Lets the idle-flush watcher
        # sleep until speech actually arrives instead of polling on a fixed grid.
        self._buffer_event = asyncio.Event()

    def _accumulate(self, transcript, lang, audio_s: float = 0.0):
        """Append a recognized segment to the current-turn buffer."""
        if transcript:
            self._buffer_text = " ".join(filter(None, [self._buffer_text, transcript]))
            self._buffer_last_segment_ts = time.monotonic()
            self._buffer_event.set()
            # Longest single segment in the buffer — acknowledgment-length audio
            # (<~1s) is where saaras mis-tags languages (e.g. Tamil 'ஆமா' heard as
            # Hindi 'हाँ'), so switch decisions gate on having at least one
            # substantive segment.
            self._buffer_max_segment_s = max(self._buffer_max_segment_s, audio_s or 0.0)
            if lang:
                # Consecutive same-language segments = saaras double-confirmation;
                # the watcher uses the streak to fire the decision without waiting
                # out the idle window.
                self._buffer_lang_streak = self._buffer_lang_streak + 1 if lang == self._buffer_lang else 1
                self._buffer_lang = lang

    def buffer_max_segment_seconds(self) -> float:
        """Duration of the longest buffered segment (peek, no drain)."""
        return self._buffer_max_segment_s

    def buffer_event(self):
        """asyncio.Event that is set while undrained speech sits in the buffer."""
        return self._buffer_event

    def buffer_age_seconds(self):
        """Seconds since the last buffered segment, or None if the buffer is empty.

        Used by the idle-flush fallback: a non-empty buffer that has gone quiet with
        no main-transcriber turn means the active (locked) ASR couldn't decode the
        caller's speech.
        """
        if not self._buffer_text.strip() or self._buffer_last_segment_ts is None:
            return None
        return time.monotonic() - self._buffer_last_segment_ts

    def buffer_language(self):
        """Latest detected language of the buffered speech (peek, no drain).

        Lets the idle-flush watcher use a shorter idle threshold when the caller is
        audibly NOT speaking the active language — no reason to wait the full
        accumulate window when saaras has already tagged the speech as different.
        """
        return self._buffer_lang

    def buffer_language_streak(self):
        """How many consecutive buffered segments carried the current buffer_language."""
        return self._buffer_lang_streak

    def take_turn_transcript(self):
        """Return (transcript, detected_lang) accumulated so far and clear the buffer.

        Called once per conversational turn by TranscriberPool/TaskManager.
        """
        text = self._buffer_text.strip()
        lang = self._buffer_lang
        self._buffer_text = ""
        self._buffer_lang = None
        self._buffer_lang_streak = 0
        self._buffer_max_segment_s = 0.0
        self._buffer_last_segment_ts = None
        self._buffer_event.clear()
        return text, lang

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

    def feed(self, audio_bytes):
        if self._dead:
            # Log the first drop loudly, then go quiet — at ~50 chunks/s a
            # per-chunk warning floods the call log while a reconnect runs.
            if not self._dead_drop_logged:
                logger.warning("SarvamLID: feed() called but WS is dead — dropping chunks until reconnected")
                self._dead_drop_logged = True
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
                        self._accumulate(transcript, short, audio_duration)

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

    def _schedule_reconnect(self, source: str):
        """Mark the socket dead and kick off one reconnect task (idempotent)."""
        if self._stopping:
            return
        self._dead = True
        if self._reconnecting:
            return
        self._reconnecting = True
        logger.error(f"SarvamLID: socket dead ({source}) — detector mute, attempting reconnect")
        asyncio.create_task(self._reconnect())

    async def _reconnect(self):
        try:
            if self._reconnect_attempts >= self._MAX_RECONNECTS:
                logger.error(
                    f"SarvamLID: reconnect cap ({self._MAX_RECONNECTS}) reached — LID inactive for remainder of call"
                )
                return
            self._reconnect_attempts += 1
            # Tear down whatever is left of the old connection. Cancelling the old
            # receiver here matters on the sender-triggered path: otherwise it would
            # observe the close of the OLD socket after we have already reconnected
            # and schedule a spurious second reconnect against the healthy one.
            for task in (self._sender_task, self._receiver_task):
                if task and not task.done():
                    task.cancel()
            if self._ws:
                try:
                    await self._ws.close()
                except Exception:
                    pass
            self._resample_state = None
            await asyncio.sleep(self._RECONNECT_DELAY_S)
            try:
                await self.start()
            except Exception as e:
                logger.error(f"SarvamLID: reconnect attempt {self._reconnect_attempts} failed: {e}")
                return
            self._dead = False
            self._dead_drop_logged = False
            logger.info(f"SarvamLID: reconnected (attempt {self._reconnect_attempts}/{self._MAX_RECONNECTS})")
        finally:
            self._reconnecting = False

    async def stop(self):
        self._stopping = True
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
