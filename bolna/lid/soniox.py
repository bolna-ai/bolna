import asyncio
import json
import os

from dotenv import load_dotenv

from bolna.constants import (
    SONIOX_DEFAULT_MULTILINGUAL_HINTS,
    SONIOX_ENDPOINT_TOKEN,
    SONIOX_WEBSOCKET_HOST,
)
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.ssl_context import get_ssl_context

from .base import LIDBackend

load_dotenv()
logger = configure_logger(__name__)


class SonioxLID(LIDBackend):
    """
    LID via Soniox stt-rt-v5 with the multilingual hint set (no language lock).

    Streams raw audio over a persistent WebSocket; Soniox returns tokens tagged
    with a per-token language. Finalized tokens are collected into one segment per
    utterance (flushed on the <end> endpoint token) so segment cadence — and the
    substance gate driven by segment duration — matches the Sarvam backend.
    Soniox reports no language probability, so segment prob is None.

    Config keys (all optional, fall back to env vars):
        soniox_api_key     — SONIOX_API_KEY env var
        soniox_host        — stt-rt.soniox.com
        telephony_provider — "twilio" | "plivo" | other
        sampling_rate      — 16000 (non-telephony input rate)
    """

    _MODEL = "stt-rt-v5"

    def __init__(self, on_language, config):
        super().__init__(on_language, config)
        self._api_key = config.get("soniox_api_key") or os.getenv("SONIOX_API_KEY", "")
        self._host = config.get("soniox_host") or SONIOX_WEBSOCKET_HOST
        self._telephony = config.get("telephony_provider", "")
        # Per-provider INPUT rate/encoding mirrors SarvamLID: telephony providers
        # stream 8kHz. Soniox accepts mulaw/pcm_s16le natively at the declared
        # sample_rate, so audio is forwarded raw — no resampling.
        if self._telephony in ("plivo", "vobiz", "exotel"):
            self._audio_format = "pcm_s16le"
            self._input_sr = 8000
        elif self._telephony == "twilio":
            self._audio_format = "mulaw"
            self._input_sr = 8000
        else:
            self._audio_format = "pcm_s16le"
            self._input_sr = int(config.get("sampling_rate", 16000))
        # Finalized tokens of the utterance currently being spoken; flushed into the
        # turn buffer on the <end> endpoint token. Single dict so init/flush reset
        # the whole segment state in one place.
        self._pending = self._empty_pending()

    @staticmethod
    def _empty_pending() -> dict:
        return {"text": "", "lang_counts": {}, "last_lang": None, "start_ms": None, "end_ms": None}

    def _build_config(self) -> dict:
        """First WebSocket frame: auth + stream config (Soniox carries the api_key here).

        KEEP IN SYNC with SonioxTranscriber._build_config (bolna/transcriber/
        soniox_transcriber.py) — same wire protocol, independently instantiated
        (the LID tap is always unbiased/multilingual, so it does not share the
        transcriber's per-call language/keyword options)."""
        return {
            "api_key": self._api_key,
            "model": self._MODEL,
            "audio_format": self._audio_format,
            "sample_rate": self._input_sr,
            "num_channels": 1,
            "enable_endpoint_detection": True,
            "enable_language_identification": True,
            # Unbiased across the supported Indian set — the detector must hear
            # whatever language is actually spoken, not the agent's locked one.
            "language_hints": list(SONIOX_DEFAULT_MULTILINGUAL_HINTS),
        }

    def _flush_pending_segment(self):
        """Move the finalized utterance into the per-turn buffer (one segment per <end>)."""
        pending = self._pending
        self._pending = self._empty_pending()
        text = pending["text"].strip()
        if not text:
            return
        if pending["lang_counts"]:
            # Dominant token language of the utterance; tie → latest seen.
            best = max(pending["lang_counts"].values())
            top = [lang for lang, n in pending["lang_counts"].items() if n == best]
            lang = pending["last_lang"] if pending["last_lang"] in top else top[-1]
        else:
            lang = None
        if pending["start_ms"] is not None and pending["end_ms"] is not None:
            audio_s = max(0.0, (pending["end_ms"] - pending["start_ms"]) / 1000.0)
        else:
            audio_s = 0.0
        logger.info(f"SonioxLID segment: lang={lang!r} transcript={text[:60]!r} audio_s={audio_s:.3f}")
        self._accumulate(text, lang, audio_s, prob=None)
        if self.on_language is not None and lang:
            # Legacy per-segment signal; Soniox has no language score → confidence None.
            asyncio.create_task(self.on_language(lang, None))

    def _handle_message(self, data: dict):
        """Fold one Soniox response into the pending segment; flush on <end>."""
        if data.get("error_code") is not None:
            raise ConnectionError(f"{data.get('error_code')}: {data.get('error_message')}")
        pending = self._pending
        for token in data.get("tokens", []):
            text = token.get("text", "")
            if not text:
                continue
            if text == SONIOX_ENDPOINT_TOKEN:
                self._flush_pending_segment()
                pending = self._pending
                continue
            if not token.get("is_final"):
                # Non-final tokens still mutate; only committed text enters the segment.
                continue
            pending["text"] += text
            lang = token.get("language")
            if lang:
                short = lang.split("-")[0].lower()
                pending["lang_counts"][short] = pending["lang_counts"].get(short, 0) + 1
                pending["last_lang"] = short
            start_ms = token.get("start_ms")
            end_ms = token.get("end_ms")
            first_seen = start_ms if start_ms is not None else end_ms
            if first_seen is not None and pending["start_ms"] is None:
                pending["start_ms"] = first_seen
            if end_ms is not None:
                pending["end_ms"] = end_ms if pending["end_ms"] is None else max(pending["end_ms"], end_ms)

    async def start(self):
        import websockets as ws_lib

        protocol = os.getenv("SONIOX_HOST_PROTOCOL", "wss")
        url = f"{protocol}://{self._host}/transcribe-websocket"
        logger.info(f"SonioxLID: connecting to {url}")
        # Same shared SSL context as SonioxTranscriber — one place to fix cert handling.
        self._ws = await ws_lib.connect(url, ssl=get_ssl_context(url))
        await self._ws.send(json.dumps(self._build_config()))
        self._sender_task = asyncio.create_task(self._sender_loop())
        self._receiver_task = asyncio.create_task(self._receiver_loop())
        logger.info(f"SonioxLID: connected (audio_format={self._audio_format}, sample_rate={self._input_sr})")

    async def _sender_loop(self):
        try:
            while True:
                chunk = await self._queue.get()
                if chunk is None:
                    break
                await self._ws.send(chunk)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"SonioxLID sender error: {e}")
            self._schedule_reconnect("sender error")

    async def _receiver_loop(self):
        try:
            async for raw in self._ws:
                try:
                    data = json.loads(raw) if isinstance(raw, str) else {}
                    self._handle_message(data)
                    if data.get("finished"):
                        logger.info("SonioxLID: session finished")
                        break
                except ConnectionError as e:
                    logger.error(f"SonioxLID server error: {e}")
                    break
                except Exception as e:
                    logger.error(f"SonioxLID receiver parse error: {e}")
        except asyncio.CancelledError:
            return
        except Exception as e:
            logger.error(f"SonioxLID receiver error: {e}")
        # Reached on both a server-side graceful close and receive errors — same
        # reconnect handling as SarvamLID so the detector never goes silently mute.
        self._schedule_reconnect("receiver closed")

    async def stop(self):
        # Don't lose the utterance in flight at teardown.
        self._flush_pending_segment()
        await self._shutdown_connection()
