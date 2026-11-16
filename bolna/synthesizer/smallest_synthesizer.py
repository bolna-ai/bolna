import asyncio
import audioop
import base64
import json
import os
import time
import uuid

import aiohttp
import websockets

from .stream_synthesizer import StreamSynthesizer
from bolna.constants import SMALLEST_TTS_SPEED_MAX, SMALLEST_TTS_SPEED_MIN
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.ssl_context import get_ssl_context

logger = configure_logger(__name__)


class SmallestSynthesizer(StreamSynthesizer):
    # Attribution tag sent to Smallest AI so requests are tracked as bolna traffic.
    SOURCE = "bolna"

    def __init__(
        self,
        voice_id,
        model="lightning_v3.1",
        language="en",
        audio_format="mp3",
        sampling_rate="8000",
        stream=False,
        buffer_size=400,
        synthesizer_key=None,
        speed=1.0,
        **kwargs,
    ):
        super().__init__(
            stream=stream,
            provider_name="smallest",
            buffer_size=buffer_size,
            **kwargs,
        )
        self.api_key = os.environ["SMALLEST_API_KEY"] if synthesizer_key is None else synthesizer_key
        self.voice_id = voice_id
        self.model = model
        self.sampling_rate = int(sampling_rate)
        self.language = language
        self.speed = float(speed)
        if not SMALLEST_TTS_SPEED_MIN <= self.speed <= SMALLEST_TTS_SPEED_MAX:
            raise ValueError(f"Smallest speed must be between {SMALLEST_TTS_SPEED_MIN} and {SMALLEST_TTS_SPEED_MAX}")
        # Buffered-HTTP mode sends one WAV-containerized clip per ~40-char buffer, and the
        # telephony handler mulaw-encodes it without stripping the header — audible glitches.
        self.stream = True

        # Unified Waves endpoints (docs.smallest.ai -> /text-to-speech).
        # HTTP: POST /waves/v1/tts, streaming: WSS /waves/v1/tts/live
        self.api_url = "https://api.smallest.ai/waves/v1/tts"
        self.ws_url = "wss://api.smallest.ai/waves/v1/tts/live"

        # Context management (docs.smallest.ai -> text-to-speech-lightning/continuations):
        # groups fragments of one turn under a context_id so interruption can send
        # cancel_request/context_close instead of needing to touch the connection — a bare
        # second request with no context_id is silently dropped if one is already streaming.
        self.context_id = None
        self._ctx_turn_id = None
        self._ctx_sequence_id = None
        self._ctx_finalized = True  # forces a fresh context_id on the very first push

        # Responses don't echo context_id (per docs — only session_id, which doesn't
        # correlate to our turns), so cancel_request/context_close are best-effort against
        # whatever's still buffered server-side; a few already-in-flight chunks can still
        # land after cancelling. Filtered here via bolna's own sequence_id bookkeeping.
        self.ignored_sequence_ids = set()

    # ------------------------------------------------------------------
    # StreamSynthesizer hooks
    # ------------------------------------------------------------------

    def _get_audio_format(self):
        return "mulaw"

    def _process_audio_chunk(self, chunk):
        # Drop trailing audio for a turn handle_interruption() already cancelled.
        if self.meta_info.get("sequence_id") in self.ignored_sequence_ids:
            return None
        # /waves/v1/tts/live ignores output_format (verified empirically — ulaw/pcm/wav/
        # omitted all return identical bytes) and always sends raw, headerless 16-bit PCM,
        # so convert to mulaw here rather than relying on a per-provider telephony fallback
        # (Plivo/Vobiz don't convert — they trust the format tag and would ship these bytes
        # as "audio/x-mulaw" untouched, which is static since they're still 16-bit PCM).
        return audioop.lin2ulaw(chunk, 2)

    def form_payload(self, text, continue_stream=True):
        return {
            "context_id": self.context_id,
            "voice_id": self.voice_id,
            "text": text,
            "model": self.model,
            "language": self.language,
            "sample_rate": self.sampling_rate,
            "speed": self.speed,
            "continue": continue_stream,
        }

    def _on_push(self, meta_info, text):
        """Fresh context_id on turn/sequence change or after the previous one finalized —
        mirrors CartesiaSynthesizer's _on_push (same reason: this must run before the
        shared _stamp_turn_start's bookkeeping gets reused for the next comparison)."""
        if (
            not self.context_id
            or self._ctx_finalized
            or self._ctx_turn_id != meta_info.get("turn_id")
            or self._ctx_sequence_id != meta_info.get("sequence_id")
        ):
            self.context_id = str(uuid.uuid4())
            self._ctx_turn_id = meta_info.get("turn_id")
            self._ctx_sequence_id = meta_info.get("sequence_id")
        self._ctx_finalized = meta_info.get("end_of_llm_stream", False)

    # ------------------------------------------------------------------
    # Interruption
    # ------------------------------------------------------------------

    async def handle_interruption(self):
        try:
            # -1 (and None) mark canned/non-turn speech (handoff, welcome message) and can
            # legitimately repeat across separate utterances — never treat it as stale.
            if self.current_sequence_id not in (None, -1):
                self.ignored_sequence_ids.add(self.current_sequence_id)
            if self.context_id and self._is_ws_connected():
                # cancel_request discards buffered-but-unspoken text; context_close tears the
                # context down immediately instead of waiting out its idle timeout.
                await self._send_json({"context_id": self.context_id, "cancel_request": True})
                await self._send_json({"context_id": self.context_id, "context_close": True})
            self.context_id = None
            self._ctx_finalized = True
            # The interrupted context's end-of-stream is now dropped, so the
            # next turn must be re-detected as new to clear stale queue entries.
            self.current_turn_start_time = None
        except Exception as e:
            logger.error(f"Error in handle_interruption: {e}")

    # ------------------------------------------------------------------
    # sender / receiver
    # ------------------------------------------------------------------

    async def sender(self, text, sequence_id, end_of_llm_stream=False):
        try:
            if self.conversation_ended:
                return
            if not self.should_synthesize_response(sequence_id):
                logger.info(f"Not synthesizing: sequence_id {sequence_id} not current")
                return

            await self._wait_for_ws()

            if text != "":
                try:
                    if self.ws_send_time is None:
                        self.ws_send_time = time.perf_counter()
                    await self._send_json(self.form_payload(text, continue_stream=not end_of_llm_stream))
                except Exception as e:
                    logger.error(f"Error sending chunk: {e}")
                    self.connection_error = str(e)
                    return

            if end_of_llm_stream:
                self.last_text_sent = True
                if text == "":
                    # No new text on this final call — still need to tell Smallest the
                    # context is done rather than leaving it to its idle timeout.
                    try:
                        await self._send_json(self.form_payload("", continue_stream=False))
                    except Exception as e:
                        logger.error(f"Error sending end-of-stream signal: {e}")
                        self.connection_error = str(e)

        except asyncio.CancelledError:
            logger.info("Sender task was cancelled.")
        except Exception as e:
            logger.error(f"Unexpected error in sender: {e}")

    async def receiver(self):
        not_connected_since = None
        while True:
            try:
                if self.conversation_ended:
                    return
                if not self._is_ws_connected():
                    if self.connection_error:
                        return
                    now = time.perf_counter()
                    if not_connected_since is None:
                        not_connected_since = now
                    elif now - not_connected_since > 30:
                        logger.error("Smallest receiver: WebSocket never connected after 30s, giving up.")
                        self.connection_error = self.connection_error or "WebSocket never connected"
                        return
                    logger.info("WebSocket is not connected, skipping receive.")
                    await asyncio.sleep(0.1)
                    continue
                else:
                    not_connected_since = None

                response = await self.websocket.recv()
                data = json.loads(response)

                if data.get("status") == "chunk":
                    yield base64.b64decode(data["data"]["audio"])
                elif data.get("status") == "complete":
                    yield b"\x00"

            except websockets.exceptions.ConnectionClosed:
                break
            except Exception as e:
                logger.error(f"Error occurred in receiver - {e}")

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    async def establish_connection(self):
        try:
            start_time = time.perf_counter()
            websocket = await asyncio.wait_for(
                websockets.connect(
                    self.ws_url,
                    additional_headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "X-Source": self.SOURCE,
                    },
                    ssl=get_ssl_context(self.ws_url),
                ),
                timeout=10.0,
            )
            if not self.connection_time:
                self.connection_time = round((time.perf_counter() - start_time) * 1000)
            logger.info(f"Connected to {self.ws_url}")
            return websocket
        except asyncio.TimeoutError:
            logger.error("Timeout while connecting to Smallest websocket")
            return None
        except Exception as e:
            logger.error(f"Failed to connect to Smallest: {e}")
            return None

    def _process_http_audio(self, audio):
        # Guard against null response from API
        return audio if audio else b"\x00"

    def _get_http_audio_format(self):
        return "wav"

    # ------------------------------------------------------------------
    # HTTP (one-shot synthesize(), e.g. pre-warmed handoff clips — not the live call path)
    # ------------------------------------------------------------------

    def _http_payload(self, text):
        return {
            "text": text,
            "voice_id": self.voice_id,
            "model": self.model,
            "sample_rate": self.sampling_rate,
            # Unlike the WS route, the sync endpoint honors output_format — keep it "wav".
            "output_format": "wav",
            "speed": self.speed,
        }

    async def _generate_http(self, text):
        logger.info(f"text {text}")
        payload = self._http_payload(text)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "audio/wav",
            "X-Source": self.SOURCE,
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(self.api_url, headers=headers, json=payload) as response:
                if response.status == 200:
                    return await response.read()
                else:
                    logger.error(f"Error: {response.status} - {await response.text()}")
                    return None

    async def synthesize(self, text):
        return await self._generate_http(text)
