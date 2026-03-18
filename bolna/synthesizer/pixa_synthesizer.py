import asyncio
import audioop
import json
import os
import time
import traceback
import uuid

import aiohttp
import websockets
from websockets.exceptions import InvalidHandshake

from .stream_synthesizer import StreamSynthesizer
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.utils import create_ws_data_packet

logger = configure_logger(__name__)


class PixaSynthesizer(StreamSynthesizer):
    def __init__(self, voice_id, voice, model="luna-tts", language="hi", sampling_rate="32000",
                 stream=False, buffer_size=400, top_p=0.95, repetition_penalty=1.3,
                 synthesizer_key=None, caching=False, **kwargs):
        super().__init__(
            stream=True,  # Pixa always streams
            provider_name="pixa",
            buffer_size=buffer_size,
            **kwargs,
        )
        self.api_key = os.environ.get("PIXA_API_KEY") if synthesizer_key is None else synthesizer_key
        self.voice_id = voice_id
        self.voice = voice
        self.model = model
        self.language = language
        self.stream = True
        self.sampling_rate = sampling_rate
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty

        self.native_sampling_rate = 32000
        self.target_sampling_rate = 8000
        self.use_mulaw = True

        self.context_id = None
        self.context_ids_to_ignore = set()

        self.api_host = os.environ.get("PIXA_TTS_HOST", "hindi.heypixa.ai")
        self.ws_url = f"wss://{self.api_host}/api/v1/ws/synthesize"

    def get_sleep_time(self):
        return 0.01

    # ------------------------------------------------------------------
    # StreamSynthesizer hooks
    # ------------------------------------------------------------------

    def _is_ws_connected(self):
        # Pixa checks for OPEN specifically (rejects CLOSING state too)
        ws = self.websocket_holder["websocket"]
        return ws is not None and ws.state == websockets.protocol.State.OPEN

    def _get_audio_format(self):
        return "mulaw" if self.use_mulaw else "pcm"

    def _process_audio_chunk(self, chunk):
        # Audio is already resampled in receiver() for binary chunks
        return chunk

    def _enrich_meta_info(self, meta_info, message):
        meta_info["sample_rate"] = self.target_sampling_rate

    def _on_push(self, meta_info, text):
        if not self.context_id:
            self.context_id = str(uuid.uuid4())

    # ------------------------------------------------------------------
    # Audio resampling
    # ------------------------------------------------------------------

    def _resample_audio(self, audio_bytes):
        """Resample PCM16 audio from 32kHz to 8kHz and convert to mulaw for telephony."""
        try:
            resampled, _ = audioop.ratecv(
                audio_bytes, 2, 1,
                self.native_sampling_rate, self.target_sampling_rate, None,
            )
            if self.use_mulaw:
                resampled = audioop.lin2ulaw(resampled, 2)
            return resampled
        except Exception as e:
            logger.error(f"Error resampling audio: {e}")
            return audio_bytes

    # ------------------------------------------------------------------
    # Interruption
    # ------------------------------------------------------------------

    async def handle_interruption(self):
        try:
            if self.context_id:
                self.context_ids_to_ignore.add(self.context_id)
                interrupt_message = {"type": "cancel", "context_id": self.context_id}
                logger.info(f"handle_interruption: {interrupt_message}")
                ws = self.websocket_holder["websocket"]
                if ws and ws.state == websockets.protocol.State.OPEN:
                    await ws.send(json.dumps(interrupt_message))
                self.context_id = None
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
                    payload = {
                        "type": "text", "content": text, "is_final": False,
                    }
                    if self.context_id:
                        payload["context_id"] = self.context_id
                    await self._send_json(payload)
                except Exception as e:
                    logger.error(f"Error sending chunk: {e}")
                    self.connection_error = str(e)
                    return

            if end_of_llm_stream:
                self.last_text_sent = True
                try:
                    payload = {"type": "text", "content": "", "is_final": True}
                    if self.context_id:
                        payload["context_id"] = self.context_id
                    await self._send_json(payload)
                except Exception as e:
                    logger.error(f"Error sending end-of-stream signal: {e}")
                    self.connection_error = str(e)

        except asyncio.CancelledError:
            logger.info("Sender task was cancelled.")
        except Exception as e:
            logger.error(f"Unexpected error in sender: {e}")

    async def receiver(self):
        while True:
            try:
                if self.conversation_ended:
                    return
                ws = self.websocket_holder["websocket"]
                if ws is None or ws.state != websockets.protocol.State.OPEN:
                    if self.connection_error:
                        return
                    logger.info("WebSocket is not connected, skipping receive.")
                    await asyncio.sleep(0.1)
                    continue

                response = await ws.recv()

                if isinstance(response, bytes):
                    yield self._resample_audio(response)
                else:
                    try:
                        data = json.loads(response)
                        context_id = data.get("context_id")
                        if context_id and context_id in self.context_ids_to_ignore:
                            continue
                        if data.get("type") == "done" or data.get("done"):
                            yield b'\x00'
                        elif data.get("type") == "error":
                            logger.error(f"Pixa error: {data.get('message', 'Unknown error')}")
                        else:
                            logger.info(f"Pixa status message: {data}")
                    except json.JSONDecodeError:
                        logger.warning(f"Received non-JSON text response: {response[:100]}")

            except websockets.exceptions.ConnectionClosed:
                logger.info("Pixa WebSocket connection closed")
                break
            except Exception as e:
                logger.error(f"Error occurred in receiver: {e}")

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    async def establish_connection(self):
        try:
            start_time = time.perf_counter()
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            websocket = await asyncio.wait_for(
                websockets.connect(self.ws_url, additional_headers=headers), timeout=10.0,
            )
            config_message = {
                "type": "config",
                "voice": self.voice_id,
                "top_p": self.top_p,
                "repetition_penalty": self.repetition_penalty,
            }
            await websocket.send(json.dumps(config_message))
            if not self.connection_time:
                self.connection_time = round((time.perf_counter() - start_time) * 1000)
            logger.info(f"Connected to Pixa TTS at {self.ws_url}")
            return websocket
        except asyncio.TimeoutError:
            logger.error("Timeout while connecting to Pixa websocket")
            return None
        except InvalidHandshake as e:
            error_msg = str(e)
            if "401" in error_msg or "403" in error_msg:
                logger.error(f"Pixa authentication failed: {e}")
            elif "404" in error_msg:
                logger.error(f"Pixa endpoint not found: {e}")
            else:
                logger.error(f"Pixa handshake failed: {e}")
            self.connection_error = str(e)
            return None
        except Exception as e:
            logger.error(f"Failed to connect to Pixa: {e}")
            return None

    # ------------------------------------------------------------------
    # HTTP fallback
    # ------------------------------------------------------------------

    async def synthesize(self, text):
        try:
            url = f"https://{self.api_host}/api/v1/synthesize"
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            payload = {
                "text": text, "voice": self.voice_id,
                "top_p": self.top_p, "repetition_penalty": self.repetition_penalty,
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status != 200:
                        logger.error(f"Pixa HTTP API error: {response.status} - {await response.text()}")
                        return None
                    wav_audio = await response.read()
                    pcm_audio = wav_audio[44:] if len(wav_audio) > 44 else wav_audio
                    resampled, _ = audioop.ratecv(
                        pcm_audio, 2, 1, self.native_sampling_rate, self.target_sampling_rate, None,
                    )
                    return resampled
        except Exception as e:
            logger.error(f"Error in synthesize: {e}")
            traceback.print_exc()
            return None
