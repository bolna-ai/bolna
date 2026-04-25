import asyncio
import base64
import json
import os
import time

import aiohttp
import websockets

from .stream_synthesizer import StreamSynthesizer
from bolna.helpers.aiohttp_session import get_shared_aiohttp_session
from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)


class SmallestSynthesizer(StreamSynthesizer):
    def __init__(
        self,
        voice_id,
        model="lightning",
        language="en",
        audio_format="mp3",
        sampling_rate="8000",
        stream=False,
        buffer_size=400,
        synthesizer_key=None,
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

        self.api_url = f"https://waves-api.smallest.ai/api/v1/{self.model}/get_speech"
        self.ws_url = "wss://waves-api.smallest.ai/api/v1/lightning-v2/get_speech/stream?timeout=60"

    # ------------------------------------------------------------------
    # StreamSynthesizer hooks
    # ------------------------------------------------------------------

    def _get_audio_format(self):
        return "wav"

    def form_payload(self, text):
        return {
            "voice_id": self.voice_id,
            "text": text,
            "language": self.language,
            "sample_rate": self.sampling_rate,
        }

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
                    await self._send_json(self.form_payload(text))
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
                    additional_headers={"Authorization": f"Token {self.api_key}"},
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

    # ------------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------------

    async def _generate_http(self, text):
        logger.info(f"text {text}")
        payload = {
            "text": text,
            "voice_id": self.voice_id,
            "sample_rate": self.sampling_rate,
            "add_wav_header": False,
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        session = await get_shared_aiohttp_session()
        async with session.post(self.api_url, headers=headers, json=payload) as response:
            if response.status == 200:
                return await response.read()
            else:
                logger.error(f"Error: {response.status} - {await response.text()}")
                return None

    async def synthesize(self, text):
        return await self._generate_http(text)
