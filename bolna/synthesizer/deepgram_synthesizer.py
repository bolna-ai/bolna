import asyncio
import copy
import json
import os
import time
import traceback
from collections import deque

import aiohttp
import websockets
from dotenv import load_dotenv

from .stream_synthesizer import StreamSynthesizer
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.ssl_context import get_ssl_context
from bolna.helpers.utils import convert_audio_to_wav, create_ws_data_packet
from bolna.memory.cache.inmemory_scalar_cache import InmemoryScalarCache

logger = configure_logger(__name__)
load_dotenv()
DEEPGRAM_HOST = os.getenv("DEEPGRAM_HOST", "api.deepgram.com")
DEEPGRAM_TTS_URL = f"https://{DEEPGRAM_HOST}/v1/speak"
DEEPGRAM_TTS_WS_URL = f"wss://{DEEPGRAM_HOST}/v1/speak"


class DeepgramSynthesizer(StreamSynthesizer):
    def __init__(
        self,
        voice_id,
        voice,
        audio_format="pcm",
        sampling_rate="8000",
        stream=False,
        buffer_size=400,
        caching=True,
        model="aura-zeus-en",
        **kwargs,
    ):
        super().__init__(
            stream=stream,
            provider_name="deepgram",
            buffer_size=buffer_size,
            **kwargs,
        )
        self.voice = voice
        self.voice_id = voice_id
        self.sample_rate = str(sampling_rate)
        self.model = model
        self.api_key = kwargs.get("transcriber_key", os.getenv("DEEPGRAM_AUTH_TOKEN"))

        self.use_mulaw = kwargs.get("use_mulaw", False)
        if self.use_mulaw or audio_format in ("pcm", "wav"):
            self.format = "mulaw"
        else:
            self.format = audio_format

        if len(self.model.split("-")) == 2:
            self.model = f"{self.model}-{self.voice_id}"

        self.caching = caching
        if caching:
            self.cache = InmemoryScalarCache()

        self.run_id = kwargs.get("run_id")
        self.ws_url = f"{DEEPGRAM_TTS_WS_URL}?encoding={self.format}&sample_rate={self.sample_rate}&model={self.model}"
        if self.run_id:
            self.ws_url += f"&tag={self.run_id}"

        # Extra TTFB tracking for WS mode
        self.ws_send_time = None
        self.current_turn_ttfb = None

    def get_sleep_time(self):
        return 0.01 if self.stream else super().get_sleep_time()

    # ------------------------------------------------------------------
    # StreamSynthesizer hooks
    # ------------------------------------------------------------------

    def _get_audio_format(self):
        return "mulaw" if self.use_mulaw else self.format

    # ------------------------------------------------------------------
    # Interruption
    # ------------------------------------------------------------------

    async def handle_interruption(self):
        try:
            ws = self.websocket
            if ws is not None and ws.state is websockets.protocol.State.OPEN:
                await ws.send(json.dumps({"type": "Clear"}))
                logger.info("Sent Clear message to Deepgram TTS WebSocket")
        except Exception as e:
            logger.error(f"Error handling interruption: {e}")

    # ------------------------------------------------------------------
    # sender / receiver
    # ------------------------------------------------------------------

    async def sender(self, text, sequence_id, end_of_llm_stream=False):
        try:
            if self.conversation_ended:
                return
            if not self.should_synthesize_response(sequence_id):
                logger.info(f"Not synthesizing: sequence_id {sequence_id} not current")
                await self.flush_synthesizer_stream()
                return

            await self._wait_for_ws()

            if text != "":
                if not self.should_synthesize_response(sequence_id):
                    logger.info(f"Not synthesizing (inner): sequence_id {sequence_id} not current")
                    await self.flush_synthesizer_stream()
                    return
                try:
                    if self.ws_send_time is None:
                        self.ws_send_time = time.perf_counter()
                        logger.info("Deepgram WS send first_text_sent")
                    await self._send_json({"type": "Speak", "text": text})
                except Exception as e:
                    logger.error(f"Error sending chunk to Deepgram: {e}")
                    self.connection_error = str(e)
                    return

            if end_of_llm_stream:
                self.last_text_sent = True
                try:
                    await self._send_json({"type": "Flush"})
                    logger.info("Sent Flush message to Deepgram TTS WebSocket")
                except Exception as e:
                    logger.error(f"Error sending Flush to Deepgram: {e}")
                    self.connection_error = str(e)

        except asyncio.CancelledError:
            logger.info("Deepgram sender task was cancelled.")
        except Exception as e:
            logger.error(f"Unexpected error in Deepgram sender: {e}")

    async def receiver(self):
        audio_chunk_count = 0
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
                        logger.error("Deepgram receiver: WebSocket never connected after 30s, giving up.")
                        self.connection_error = self.connection_error or "WebSocket never connected"
                        return
                    logger.info("Deepgram WebSocket is not connected, skipping receive.")
                    await asyncio.sleep(0.10)
                    continue
                else:
                    not_connected_since = None

                response = await self.websocket.recv()

                if isinstance(response, bytes):
                    audio_chunk_count += 1
                    if audio_chunk_count == 1 and self.ws_send_time is not None:
                        time_since_send = (time.perf_counter() - self.ws_send_time) * 1000
                        logger.info(f"Deepgram WS recv FIRST audio chunk time_since_send={time_since_send:.0f}ms")
                    yield response
                else:
                    try:
                        data = json.loads(response)
                        msg_type = data.get("type", "")
                        if msg_type == "Metadata":
                            logger.info(f"Deepgram TTS Metadata: request_id={data.get('request_id')}")
                        elif msg_type == "Flushed":
                            logger.info(f"Deepgram TTS Flushed: sequence_id={data.get('sequence_id')}")
                            audio_chunk_count = 0
                            yield b"\x00"
                        elif msg_type == "Cleared":
                            logger.info(f"Deepgram TTS Cleared: sequence_id={data.get('sequence_id')}")
                            audio_chunk_count = 0
                        elif msg_type == "Warning":
                            logger.info(f"Deepgram TTS Warning: {data.get('description')}")
                        else:
                            logger.info(f"Deepgram TTS response: {data}")
                    except json.JSONDecodeError:
                        logger.warning("Received unexpected non-JSON text response from Deepgram")

            except websockets.exceptions.ConnectionClosed:
                logger.info("Deepgram WebSocket connection closed")
                break
            except Exception as e:
                logger.error(f"Error in Deepgram receiver: {e}")
                traceback.print_exc()

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    async def establish_connection(self):
        try:
            start_time = time.perf_counter()
            websocket = await asyncio.wait_for(
                websockets.connect(self.ws_url, additional_headers={"Authorization": f"Token {self.api_key}"}, ssl=get_ssl_context(self.ws_url)),
                timeout=10.0,
            )
            if not self.connection_time:
                self.connection_time = round((time.perf_counter() - start_time) * 1000)
            logger.info(f"Connected to Deepgram TTS WebSocket: {self.ws_url}")
            return websocket
        except asyncio.TimeoutError:
            logger.error("Timeout while connecting to Deepgram TTS WebSocket")
            return None
        except websockets.exceptions.InvalidStatusCode as e:
            if e.status_code == 401:
                logger.error("Deepgram authentication failed: Invalid API key")
            elif e.status_code == 403:
                logger.error("Deepgram authentication failed: Access forbidden")
            else:
                logger.error(f"Deepgram WebSocket connection failed with status {e.status_code}: {e}")
            self.connection_error = str(e)
            return None
        except Exception as e:
            logger.error(f"Failed to connect to Deepgram TTS WebSocket: {e}")
            return None

    async def cleanup(self):
        """Send graceful Close before standard cleanup."""
        self.conversation_ended = True
        logger.info("Cleaning up Deepgram synthesizer tasks")

        if self.sender_task:
            try:
                self.sender_task.cancel()
                await self.sender_task
            except asyncio.CancelledError:
                logger.info("Deepgram sender task cancelled during cleanup.")
            except Exception as e:
                logger.warning(f"Error cancelling sender task: {e}")

        ws = self.websocket
        if ws:
            try:
                await ws.send(json.dumps({"type": "Close"}))
                logger.info("Sent Close message to Deepgram TTS WebSocket")
            except Exception as e:
                logger.error(f"Error sending Close message: {e}")
            try:
                await ws.close()
            except Exception as e:
                logger.error(f"Error closing WebSocket: {e}")

        self.websocket = None
        logger.info("Deepgram TTS WebSocket connection closed.")

    # ------------------------------------------------------------------
    # HTTP mode (non-streaming)
    # ------------------------------------------------------------------

    async def _generate_http(self, text):
        headers = {"Authorization": f"Token {self.api_key}", "Content-Type": "application/json"}
        url = f"{DEEPGRAM_TTS_URL}?container=none&encoding={self.format}&sample_rate={self.sample_rate}&model={self.model}"
        if self.run_id:
            url += f"&tag={self.run_id}"
        logger.info(f"Sending deepgram request {url}")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json={"text": text}) as response:
                    if response.status == 200:
                        return await response.read()
                    else:
                        logger.info(f"Deepgram request status {response.status}")
                        return b"\x00"
        except Exception as e:
            logger.error(f"Deepgram HTTP error: {e}")

    async def synthesize(self, text):
        try:
            audio = await self._generate_http(text)
            if self.format == "mp3":
                audio = convert_audio_to_wav(audio, source_format="mp3")
            return audio
        except Exception as e:
            logger.error(f"Could not synthesize {e}")

    def _process_http_audio(self, audio):
        if self.format == "mp3":
            return convert_audio_to_wav(audio, source_format="mp3")
        return audio
