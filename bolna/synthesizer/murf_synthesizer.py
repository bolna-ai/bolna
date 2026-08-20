import asyncio
import base64
import json
import os
import time
import uuid
from urllib.parse import urlencode

import websockets
from websockets.exceptions import InvalidHandshake

from .stream_synthesizer import StreamSynthesizer
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.ssl_context import get_ssl_context

logger = configure_logger(__name__)

DEFAULT_MURF_HOST = "global.api.murf.ai"


def _resolve_murf_host(base_url=None):
    """Resolve Murf host from config or global default. Accepts host or full URL."""
    host = (base_url or DEFAULT_MURF_HOST).strip()
    for prefix in ("wss://", "ws://", "https://", "http://"):
        if host.startswith(prefix):
            host = host[len(prefix) :]
            break
    return host.split("/")[0] or DEFAULT_MURF_HOST


class MurfSynthesizer(StreamSynthesizer):
    def __init__(
        self,
        voice,
        model="falcon-2",
        locale="en-US",
        style="Conversational",
        audio_format="pcm",
        sampling_rate="24000",
        stream=False,
        buffer_size=400,
        synthesizer_key=None,
        caching=True,
        base_url=None,
        min_buffer_size=40,
        max_buffer_delay_in_ms=0,
        rate=0,
        pitch=0,
        **kwargs,
    ):
        super().__init__(
            stream=True,  # Murf always streams over WebSocket
            provider_name="murf",
            buffer_size=buffer_size,
            **kwargs,
        )
        self.api_key = os.environ["MURF_API_KEY"] if synthesizer_key is None else synthesizer_key
        self.voice = voice
        self.model = model
        self.locale = locale
        self.style = style
        self.rate = rate
        self.pitch = pitch
        # Telephony (Twilio/etc.) passes use_mulaw=True → native ULAW@8k.
        # Web/FreeSWITCH pass False → PCM @ sampling_rate (typically 24k).
        self.use_mulaw = kwargs.get("use_mulaw", True)
        if self.use_mulaw:
            self.sampling_rate = 8000
            self.murf_format = "ULAW"
        else:
            self.sampling_rate = int(sampling_rate) if sampling_rate is not None else 24000
            self.murf_format = "PCM"
        self.min_buffer_size = min_buffer_size
        self.max_buffer_delay_in_ms = max_buffer_delay_in_ms
        self.stream = True

        self.murf_host = _resolve_murf_host(base_url)
        query = urlencode(
            {
                "api-key": self.api_key,
                "model": self.model,
                "sample_rate": str(self.sampling_rate),
                "channel_type": "MONO",
                "format": self.murf_format,
            }
        )
        self.ws_url = f"wss://{self.murf_host}/v1/speech/stream-input?{query}"

        # Context tracking for multi-turn / interruption
        self.context_id = None
        self.turn_id = 0
        self.sequence_id = 0
        self.context_ids_to_ignore = set()
        self.context_finalized = False
        self._pending_context_setup = False

    def get_sleep_time(self):
        return 0.01

    # ------------------------------------------------------------------
    # StreamSynthesizer hooks
    # ------------------------------------------------------------------

    def _get_audio_format(self):
        return "mulaw" if self.use_mulaw else "pcm"

    def _on_push(self, meta_info, text):
        """Fresh context_id on turn/sequence change or after the previous finalized."""
        if meta_info.get("message_category") == "handoff" and meta_info.get("end_of_llm_stream"):
            meta_info["end_of_llm_stream"] = False
        if not self.context_id:
            self._update_context(meta_info)
        elif self.context_finalized:
            self._update_context(meta_info)
        elif self.turn_id != meta_info.get("turn_id", 0) or self.sequence_id != meta_info.get("sequence_id", 0):
            self._update_context(meta_info)
        self.context_finalized = meta_info.get("end_of_llm_stream", False)

    def _update_context(self, meta_info):
        self.context_id = str(uuid.uuid4())
        self.turn_id = meta_info.get("turn_id", 0)
        self.sequence_id = meta_info.get("sequence_id", 0)
        self._pending_context_setup = True

    async def _send_context_setup(self):
        """Send voice_config + buffer settings for a new context_id."""
        config_msg = {
            "context_id": self.context_id,
            "voice_config": {
                "voiceId": self.voice,
                "locale": self.locale,
                "style": self.style,
                "rate": self.rate,
                "pitch": self.pitch,
            },
            "min_buffer_size": self.min_buffer_size,
            "max_buffer_delay_in_ms": self.max_buffer_delay_in_ms,
        }
        await self._send_json(config_msg)
        self._pending_context_setup = False

    # ------------------------------------------------------------------
    # Interruption
    # ------------------------------------------------------------------

    async def handle_interruption(self):
        try:
            if self.context_id:
                self.context_ids_to_ignore.add(self.context_id)
                interrupt_message = {"context_id": self.context_id, "clear": True}
                if self._is_ws_connected():
                    await self.websocket.send(json.dumps(interrupt_message))
                self.context_id = None
                self._pending_context_setup = False
                self.current_turn_start_time = None
        except Exception as e:
            logger.error(f"Error in handle_interruption: {e}")

    # ------------------------------------------------------------------
    # Payload
    # ------------------------------------------------------------------

    def form_payload(self, text, end=False):
        return {
            "context_id": self.context_id,
            "text": text if text is not None else "",
            "end": end,
        }

    # ------------------------------------------------------------------
    # sender / receiver
    # ------------------------------------------------------------------

    async def sender(self, text, sequence_id, end_of_llm_stream=False):
        try:
            if self.conversation_ended:
                return
            if not self.should_synthesize_response(sequence_id):
                return

            await self._wait_for_ws()
            if self.conversation_ended or self.connection_error or not self._is_ws_connected():
                return

            if self._pending_context_setup:
                try:
                    await self._send_context_setup()
                except Exception as e:
                    logger.error(f"Error sending Murf context setup: {e}")
                    self.connection_error = str(e)
                    return

            if text != "":
                try:
                    if self.ws_send_time is None:
                        self.ws_send_time = time.perf_counter()
                    await self._send_json(self.form_payload(text, end=False))
                except Exception as e:
                    logger.error(f"Error sending chunk context_id={self.context_id}: {e}")
                    self.connection_error = str(e)
                    return

            if end_of_llm_stream:
                self.last_text_sent = True
                try:
                    await self._send_json(self.form_payload("", end=True))
                except Exception as e:
                    logger.error(f"Error sending end-of-stream signal context_id={self.context_id}: {e}")
                    self.connection_error = str(e)

        except asyncio.CancelledError:
            return
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
                        logger.error("Murf receiver: WebSocket never connected after 30s, giving up.")
                        self.connection_error = self.connection_error or "WebSocket never connected"
                        return
                    await asyncio.sleep(0.1)
                    continue
                else:
                    not_connected_since = None

                response = await self.websocket.recv()
                data = json.loads(response)

                if data.get("context_id") in self.context_ids_to_ignore:
                    continue

                if data.get("error") or data.get("errorMessage"):
                    error = data.get("error") or data.get("errorMessage")
                    logger.error(f"Murf error response: {error}")
                    self.connection_error = str(error)
                    continue

                if data.get("audio"):
                    yield base64.b64decode(data["audio"])

                if data.get("final"):
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
                websockets.connect(self.ws_url, ssl=get_ssl_context(self.ws_url)), timeout=10.0
            )
            if not self.connection_time:
                self.connection_time = round((time.perf_counter() - start_time) * 1000)
            return websocket
        except asyncio.TimeoutError:
            logger.error("Timeout while connecting to Murf websocket")
            return None
        except InvalidHandshake as e:
            error_msg = str(e)
            if "401" in error_msg or "403" in error_msg:
                logger.error(f"Murf authentication failed: {e}")
            elif "404" in error_msg:
                logger.error(f"Murf endpoint not found: {e}")
            else:
                logger.error(f"Murf handshake failed: {e}")
            self.connection_error = str(e)
            return None
        except Exception as e:
            logger.error(f"Failed to connect to Murf: {e}")
            self.connection_error = str(e)
            return None

    async def synthesize(self, text):
        """Non-stream path unused for Murf agents; WS streaming is primary."""
        raise NotImplementedError("Murf synthesizer only supports WebSocket streaming")
