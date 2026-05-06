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


class SmallestSynthesizer(StreamSynthesizer):
    def __init__(
        self,
        voice_id,
        model="lightning-v3.1",
        language="en",
        audio_format="mp3",
        sampling_rate="8000",
        stream=False,
        buffer_size=400,
        synthesizer_key=None,
        speed=1.0,
        add_wav_header=False,
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
        self.speed = speed
        self.add_wav_header = add_wav_header

        self.api_url = f"https://api.smallest.ai/waves/v1/{self.model}/get_speech"
        self.ws_url = f"wss://api.smallest.ai/waves/v1/{self.model}/get_speech/stream"

        self.ws_trace_id = None

    # ------------------------------------------------------------------
    # StreamSynthesizer hooks
    # ------------------------------------------------------------------

    def _get_audio_format(self):
        return "wav"

    def _unpack_receiver_message(self, item):
        audio, text_synthesized = item
        return audio, {"text_synthesized": text_synthesized}

    def _process_audio_chunk(self, chunk):
        return chunk

    def form_payload(self, text):
        return {
            "voice_id": self.voice_id,
            "text": text,
            "language": self.language,
            "sample_rate": self.sampling_rate,
            "speed": self.speed,
            "add_wav_header": self.add_wav_header,
        }

    # ------------------------------------------------------------------
    # Interruption
    # ------------------------------------------------------------------

    async def handle_interruption(self):
        try:
            if self.websocket and self._is_ws_connected():
                await self.websocket.close()
        except Exception:
            pass

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
                for text_chunk in self.text_chunker(text):
                    if not self.should_synthesize_response(sequence_id):
                        logger.info(f"Not synthesizing (inner): sequence_id {sequence_id} not current")
                        return
                    try:
                        if self.ws_send_time is None:
                            self.ws_send_time = time.perf_counter()
                        await self._send_json(self.form_payload(text_chunk))
                    except Exception as e:
                        logger.error(f"Error sending chunk: {e}")
                        self.connection_error = str(e)
                        return

            if end_of_llm_stream:
                self.last_text_sent = True

        except asyncio.CancelledError:
            logger.info("Sender task was cancelled.")
        except Exception as e:
            logger.error(f"Unexpected error in sender: {e}")

    async def receiver(self):
        """Yields (audio_chunk, text_synthesized) tuples, or (b'\\x00', '') for end-of-stream."""
        audio_chunk_count = 0
        last_recv_time = None
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

                recv_start = time.perf_counter()
                response = await self.websocket.recv()
                recv_duration = (time.perf_counter() - recv_start) * 1000
                data = json.loads(response)

                if data.get("status") == "chunk":
                    audio_chunk_count += 1
                    if audio_chunk_count == 1 and self.ws_send_time is not None:
                        time_since_send = (time.perf_counter() - self.ws_send_time) * 1000
                        logger.info(
                            f"Smallest recv FIRST trace_id={self.ws_trace_id} "
                            f"recv_wait={recv_duration:.0f}ms ttfb={time_since_send:.0f}ms"
                        )
                    elif recv_duration > 200:
                        gap = (recv_start - last_recv_time) * 1000 if last_recv_time else 0
                        logger.info(
                            f"Smallest recv SLOW chunk={audio_chunk_count} trace_id={self.ws_trace_id} "
                            f"recv_wait={recv_duration:.0f}ms gap={gap:.0f}ms"
                        )
                    last_recv_time = time.perf_counter()
                    yield base64.b64decode(data["data"]["audio"]), ""

                elif data.get("status") == "complete":
                    logger.info(f"Smallest recv complete trace_id={self.ws_trace_id}")
                    audio_chunk_count = 0
                    last_recv_time = None
                    yield b"\x00", ""

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
                    additional_headers={"Authorization": f"Bearer {self.api_key}"},
                    ssl=get_ssl_context(self.ws_url),
                ),
                timeout=10.0,
            )
            if hasattr(websocket, "response") and hasattr(websocket.response, "headers"):
                self.ws_trace_id = websocket.response.headers.get("x-trace-id") or str(uuid.uuid4())
            if not self.connection_time:
                self.connection_time = round((time.perf_counter() - start_time) * 1000)
            logger.info(f"Connected to Smallest trace_id={self.ws_trace_id}")
            return websocket
        except asyncio.TimeoutError:
            logger.error("Timeout while connecting to Smallest websocket")
            return None
        except InvalidHandshake as e:
            error_msg = str(e)
            if "401" in error_msg or "403" in error_msg:
                logger.error(f"Smallest authentication failed: Invalid or expired API key - {e}")
            else:
                logger.error(f"Smallest handshake failed: {e}")
            self.connection_error = str(e)
            return None
        except Exception as e:
            logger.error(f"Failed to connect to Smallest: {e}")
            return None

    # ------------------------------------------------------------------
    # HTTP fallback
    # ------------------------------------------------------------------

    def _process_http_audio(self, audio):
        return audio if audio else b"\x00"

    async def _generate_http(self, text):
        payload = {
            "text": text,
            "voice_id": self.voice_id,
            "language": self.language,
            "sample_rate": self.sampling_rate,
            "speed": self.speed,
            "add_wav_header": self.add_wav_header,
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        async with aiohttp.ClientSession() as session:
            async with session.post(self.api_url, headers=headers, json=payload) as response:
                if response.status == 200:
                    return await response.read()
                else:
                    logger.error(f"Error: {response.status} - {await response.text()}")
                    return None

    async def synthesize(self, text):
        return await self._generate_http(text)
