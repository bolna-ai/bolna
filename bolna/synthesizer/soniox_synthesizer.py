import asyncio
import base64
import json
import os
import time
import uuid

import aiohttp
import websockets
from websockets.exceptions import InvalidHandshake

from .stream_synthesizer import StreamSynthesizer
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.ssl_context import get_ssl_context

logger = configure_logger(__name__)


class SonioxSynthesizer(StreamSynthesizer):
    """Soniox real-time TTS. One WebSocket carries many streams keyed by stream_id;
    each turn opens a fresh stream (config frame, then text frames, then text_end).
    Soniox has no server-side cancel, so barge-in drops the active stream_id and
    ignores its remaining audio, mirroring the reference voice-bot integration."""

    def __init__(
        self,
        voice_id=None,
        voice="Adrian",
        language="en",
        model="tts-rt-v2",
        audio_format="wav",
        sampling_rate="24000",
        stream=False,
        buffer_size=400,
        synthesizer_key=None,
        caching=True,
        **kwargs,
    ):
        super().__init__(
            stream=True,  # Soniox always streams over the WebSocket
            provider_name="soniox",
            buffer_size=buffer_size,
            **kwargs,
        )
        self.api_key = os.environ["SONIOX_API_KEY"] if synthesizer_key is None else synthesizer_key
        self.voice = voice_id or voice
        self.model = model
        self.language = language
        self.audio_format = audio_format
        self.sampling_rate = sampling_rate
        self.use_mulaw = kwargs.get("use_mulaw", True)  # web/freeswitch pass False for raw PCM @sampling_rate
        self.stream = True

        self.soniox_host = os.getenv("SONIOX_TTS_HOST", "tts-rt.soniox.com")
        self.ws_url = f"wss://{self.soniox_host}/tts-websocket"
        self.api_url = f"https://{self.soniox_host}/tts"

        # stream_id tracks the current turn's Soniox stream, exactly as Cartesia tracks context_id.
        self.stream_id = None
        self.turn_id = 0
        self.sequence_id = 0
        self.stream_ids_to_ignore = set()
        # Config frame is sent once per stream_id; the set guards concurrent chunk senders.
        self.config_sent_streams = set()
        # end_of_llm_stream sent on the current stream means the next push must open a fresh one.
        self.stream_finalized = False

    def get_sleep_time(self):
        return 0.01

    # ------------------------------------------------------------------
    # StreamSynthesizer hooks
    # ------------------------------------------------------------------

    def _get_audio_format(self):
        return "mulaw" if self.use_mulaw else "pcm"

    def _soniox_audio_format(self):
        return "pcm_mulaw" if self.use_mulaw else "pcm_s16le"

    def _soniox_sample_rate(self):
        return 8000 if self.use_mulaw else int(self.sampling_rate)

    def _on_push(self, meta_info, text):
        """Fresh stream_id on turn/sequence change or after the previous finalized. The handoff
        does not finalize so the reply continues its stream, avoiding parallel-stream overlap."""
        if meta_info.get("message_category") == "handoff" and meta_info.get("end_of_llm_stream"):
            meta_info["end_of_llm_stream"] = False
        if not self.stream_id:
            self._update_stream(meta_info)
        elif self.stream_finalized:
            self._update_stream(meta_info)
        elif self.turn_id != meta_info.get("turn_id", 0) or self.sequence_id != meta_info.get("sequence_id", 0):
            self._update_stream(meta_info)
        self.stream_finalized = meta_info.get("end_of_llm_stream", False)

    def _update_stream(self, meta_info):
        self.stream_id = str(uuid.uuid4())
        self.turn_id = meta_info.get("turn_id", 0)
        self.sequence_id = meta_info.get("sequence_id", 0)
        logger.info(f"Soniox new stream_id={self.stream_id} turn_id={self.turn_id} sequence_id={self.sequence_id}")

    def _build_config(self, stream_id):
        return {
            "api_key": self.api_key,
            "stream_id": stream_id,
            "model": self.model,
            "language": self.language,
            "voice": self.voice,
            "audio_format": self._soniox_audio_format(),
            "sample_rate": self._soniox_sample_rate(),
        }

    # ------------------------------------------------------------------
    # Interruption
    # ------------------------------------------------------------------

    async def handle_interruption(self):
        try:
            if self.stream_id:
                # Soniox cannot cancel a running stream; ignore its remaining audio and let
                # the next turn open a fresh stream_id.
                self.stream_ids_to_ignore.add(self.stream_id)
                logger.info(f"handle_interruption: ignoring stream_id={self.stream_id}")
                self.stream_id = None
                # The interrupted stream's end-of-stream is now dropped, so the next turn
                # must be re-detected as new to clear stale queue entries.
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

            stream_id = self.stream_id
            if stream_id is None:
                return

            # Send the config frame once per stream. Adding to the set before the await
            # keeps a second concurrent chunk sender from re-sending it.
            if stream_id not in self.config_sent_streams:
                self.config_sent_streams.add(stream_id)
                try:
                    await self._send_json(self._build_config(stream_id))
                except Exception as e:
                    logger.error(f"Error sending config stream_id={stream_id}: {e}")
                    self.connection_error = str(e)
                    return

            if text != "":
                try:
                    if self.ws_send_time is None:
                        self.ws_send_time = time.perf_counter()
                    logger.info(f"Soniox sender stream_id={stream_id} text_len={len(text)}")
                    await self._send_json({"text": text, "text_end": False, "stream_id": stream_id})
                except Exception as e:
                    logger.error(f"Error sending chunk stream_id={stream_id}: {e}")
                    self.connection_error = str(e)
                    return

            if end_of_llm_stream:
                self.last_text_sent = True
                logger.info(f"Soniox sender end_of_llm_stream stream_id={stream_id}")
                try:
                    await self._send_json({"text": "", "text_end": True, "stream_id": stream_id})
                except Exception as e:
                    logger.error(f"Error sending end-of-stream signal stream_id={stream_id}: {e}")
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
                        logger.error("Soniox receiver: WebSocket never connected after 30s, giving up.")
                        self.connection_error = self.connection_error or "WebSocket never connected"
                        return
                    logger.info("WebSocket is not connected, skipping receive.")
                    await asyncio.sleep(0.1)
                    continue
                else:
                    not_connected_since = None

                response = await self.websocket.recv()
                data = json.loads(response)

                if data.get("stream_id") in self.stream_ids_to_ignore:
                    continue

                if data.get("error_code") is not None:
                    logger.error(
                        f"Soniox error stream_id={data.get('stream_id')} "
                        f"code={data.get('error_code')} message={data.get('error_message')}"
                    )
                    # End the turn gracefully so the caller is not left mute.
                    yield b"\x00"
                    continue

                audio_b64 = data.get("audio")
                if audio_b64:
                    yield base64.b64decode(audio_b64)

                if data.get("terminated"):
                    logger.info(f"Soniox recv terminated stream_id={data.get('stream_id')}")
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
            logger.info(f"Soniox WebSocket connected connection_time={self.connection_time}ms")
            return websocket
        except asyncio.TimeoutError:
            logger.error("Timeout while connecting to Soniox websocket")
            return None
        except InvalidHandshake as e:
            error_msg = str(e)
            if "401" in error_msg or "403" in error_msg:
                logger.error(f"Soniox authentication failed: {e}")
            elif "404" in error_msg:
                logger.error(f"Soniox endpoint not found: {e}")
            else:
                logger.error(f"Soniox handshake failed: {e}")
            self.connection_error = str(e)
            return None
        except Exception as e:
            logger.error(f"Failed to connect to Soniox: {e}")
            return None

    # ------------------------------------------------------------------
    # HTTP (used by synthesize() and the turn-based dashboard path)
    # ------------------------------------------------------------------

    def _get_http_audio_format(self):
        return "mp3" if self.audio_format == "mp3" else "wav"

    async def synthesize(self, text):
        return await self._generate_http(text)

    async def _generate_http(self, text):
        payload = {
            "model": self.model,
            "text": text,
            "language": self.language,
            "voice": self.voice,
            "audio_format": self._get_http_audio_format(),
            "sample_rate": int(self.sampling_rate),
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        async with aiohttp.ClientSession() as session:
            async with session.post(self.api_url, headers=headers, json=payload) as response:
                if response.status == 200:
                    return await response.read()
                else:
                    logger.error(f"Error: {response.status} - {await response.text()}")
                    return None
