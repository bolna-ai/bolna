import asyncio
import copy
import uuid
import time
import websockets
from websockets.exceptions import InvalidHandshake
import aiohttp
import base64
import json
import audioop
import os
import traceback
from collections import deque

from .base_synthesizer import BaseSynthesizer
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.utils import create_ws_data_packet

logger = configure_logger(__name__)


class PixaSynthesizer(BaseSynthesizer):
    def __init__(self, voice_id, voice, model="luna-tts", language="hi", sampling_rate="32000",
                 stream=False, buffer_size=400, top_p=0.95, repetition_penalty=1.3,
                 synthesizer_key=None, caching=False, **kwargs):
        super().__init__(kwargs.get("task_manager_instance", None), stream)
        self.api_key = os.environ.get("PIXA_API_KEY") if synthesizer_key is None else synthesizer_key
        self.voice_id = voice_id
        self.voice = voice
        self.model = model
        self.language = language
        self.stream = True
        self.native_sampling_rate = 32000
        self.target_sampling_rate = 8000
        self.sampling_rate = sampling_rate
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty

        self.websocket_holder = {"websocket": None}
        self.connection_open = False
        self.use_mulaw = True  # For telephony compatibility
        self.first_chunk_generated = False
        self.last_text_sent = False
        self.text_queue = deque()
        self.meta_info = None
        self.caching = False
        self.synthesized_characters = 0
        self.context_id = None
        self.sender_task = None
        self.context_ids_to_ignore = set()
        self.conversation_ended = False
        self.current_turn_start_time = None
        self.current_turn_id = None
        self.current_text = ""

        self.api_host = os.environ.get("PIXA_TTS_HOST", "hindi.heypixa.ai")
        self.ws_url = f"wss://{self.api_host}/api/v1/ws/synthesize"

    def get_engine(self):
        return self.model

    def supports_websocket(self):
        return True

    def get_sleep_time(self):
        return 0.01

    def resample_audio(self, audio_bytes):
        """Resample PCM16 audio from 32kHz to 8kHz and convert to mulaw for telephony."""
        try:
            # Resample from 32kHz to 8kHz
            resampled, _ = audioop.ratecv(
                audio_bytes,
                2,  # 2 bytes per sample (16-bit PCM)
                1,  # mono
                self.native_sampling_rate,
                self.target_sampling_rate,
                None
            )
            # Convert to mulaw for telephony if enabled
            if self.use_mulaw:
                resampled = audioop.lin2ulaw(resampled, 2)
            return resampled
        except Exception as e:
            logger.error(f"Error resampling audio: {e}")
            return audio_bytes

    async def handle_interruption(self):
        try:
            if self.context_id:
                self.context_ids_to_ignore.add(self.context_id)
                interrupt_message = {
                    "type": "cancel",
                    "context_id": self.context_id
                }
                logger.info(f'handle_interruption: {interrupt_message}')
                if self.websocket_holder["websocket"] and self.websocket_holder["websocket"].state == websockets.protocol.State.OPEN:
                    await self.websocket_holder["websocket"].send(json.dumps(interrupt_message))
                self.context_id = None
        except Exception as e:
            logger.error(f"Error in handle_interruption: {e}")

    def form_payload(self, text, is_final=False):
        payload = {
            "type": "text",
            "content": text,
            "is_final": is_final
        }
        if self.context_id:
            payload["context_id"] = self.context_id
        return payload

    async def sender(self, text, sequence_id, end_of_llm_stream=False):
        try:
            if self.conversation_ended:
                return

            if not self.should_synthesize_response(sequence_id):
                logger.info(f"Not synthesizing text as the sequence_id ({sequence_id}) is not in the list of sequence_ids present in the task manager.")
                return

            while self.websocket_holder["websocket"] is None or self.websocket_holder["websocket"].state != websockets.protocol.State.OPEN:
                logger.info("Waiting for Pixa WebSocket connection to be established...")
                await asyncio.sleep(0.5)

            if text != "":
                try:
                    input_message = self.form_payload(text, is_final=False)
                    await self.websocket_holder["websocket"].send(json.dumps(input_message))
                except Exception as e:
                    logger.error(f"Error sending chunk: {e}")
                    return

            if end_of_llm_stream:
                self.last_text_sent = True
                try:
                    input_message = self.form_payload("", is_final=True)
                    await self.websocket_holder["websocket"].send(json.dumps(input_message))
                except Exception as e:
                    logger.error(f"Error sending end-of-stream signal: {e}")

        except asyncio.CancelledError:
            logger.info("Sender task was cancelled.")
        except Exception as e:
            logger.error(f"Unexpected error in sender: {e}")

    async def receiver(self):
        while True:
            try:
                if self.conversation_ended:
                    return

                if self.websocket_holder["websocket"] is None or self.websocket_holder["websocket"].state != websockets.protocol.State.OPEN:
                    logger.info("WebSocket is not connected, skipping receive.")
                    await asyncio.sleep(0.1)
                    continue

                response = await self.websocket_holder["websocket"].recv()

                # Handle binary audio data (PCM16 at 32kHz)
                if isinstance(response, bytes):
                    # Resample from 32kHz to 8kHz
                    resampled_audio = self.resample_audio(response)
                    yield resampled_audio
                else:
                    # Handle JSON status messages
                    try:
                        data = json.loads(response)
                        context_id = data.get('context_id')

                        if context_id and context_id in self.context_ids_to_ignore:
                            continue

                        if data.get('type') == 'done' or data.get('done'):
                            yield b'\x00'
                        elif data.get('type') == 'error':
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

    async def synthesize(self, text):
        """One-off synthesis using HTTP API. Returns PCM audio for preview."""
        try:
            url = f"https://{self.api_host}/api/v1/synthesize"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "text": text,
                "voice": self.voice_id,
                "top_p": self.top_p,
                "repetition_penalty": self.repetition_penalty
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Pixa HTTP API error: {response.status} - {error_text}")
                        return None

                    # Response is WAV audio (PCM16 at 32kHz)
                    wav_audio = await response.read()

                    # Skip WAV header (44 bytes) to get raw PCM
                    pcm_audio = wav_audio[44:] if len(wav_audio) > 44 else wav_audio

                    # Resample from 32kHz to 8kHz
                    resampled, _ = audioop.ratecv(
                        pcm_audio, 2, 1,
                        self.native_sampling_rate,
                        self.target_sampling_rate,
                        None
                    )
                    return resampled

        except Exception as e:
            logger.error(f"Error in synthesize: {e}")
            traceback.print_exc()
            return None

    def get_synthesized_characters(self):
        return self.synthesized_characters

    async def generate(self):
        try:
            async for message in self.receiver():
                if len(self.text_queue) > 0:
                    self.meta_info = self.text_queue.popleft()
                    try:
                        if self.current_turn_start_time is not None:
                            first_result_latency = time.perf_counter() - self.current_turn_start_time
                            self.meta_info['synthesizer_latency'] = first_result_latency
                    except Exception:
                        pass

                # Defensive check for meta_info
                if self.meta_info is None:
                    self.meta_info = {}

                self.meta_info['format'] = 'mulaw' if self.use_mulaw else 'pcm'
                self.meta_info['sample_rate'] = self.target_sampling_rate
                audio = message

                if not self.first_chunk_generated:
                    self.meta_info["is_first_chunk"] = True
                    self.first_chunk_generated = True
                else:
                    self.meta_info["is_first_chunk"] = False

                if self.last_text_sent:
                    self.first_chunk_generated = False
                    self.last_text_sent = True

                if message == b'\x00':
                    logger.info("Received end of stream marker")
                    self.meta_info["end_of_synthesizer_stream"] = True
                    self.first_chunk_generated = False
                    try:
                        if self.current_turn_start_time is not None:
                            total_stream_duration = time.perf_counter() - self.current_turn_start_time
                            self.turn_latencies.append({
                                'turn_id': self.current_turn_id,
                                'sequence_id': self.current_turn_id,
                                'first_result_latency_ms': round((self.meta_info.get('synthesizer_latency', 0)) * 1000),
                                'total_stream_duration_ms': round(total_stream_duration * 1000)
                            })
                            self.current_turn_start_time = None
                            self.current_turn_id = None
                    except Exception:
                        pass

                yield create_ws_data_packet(audio, self.meta_info)

        except Exception as e:
            traceback.print_exc()
            logger.error(f"Error in pixa generate: {e}")

    async def establish_connection(self):
        try:
            start_time = time.perf_counter()

            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            websocket = await asyncio.wait_for(
                websockets.connect(self.ws_url, additional_headers=headers),
                timeout=10.0
            )

            # Send initial configuration
            config_message = {
                "type": "config",
                "model": self.model,
                "language": self.language,
                "voice_id": self.voice_id,
                "top_p": self.top_p,
                "repetition_penalty": self.repetition_penalty
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
            if '401' in error_msg or '403' in error_msg:
                logger.error(f"Pixa authentication failed: Invalid or expired API key - {e}")
            elif '404' in error_msg:
                logger.error(f"Pixa endpoint not found: {e}")
            else:
                logger.error(f"Pixa handshake failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to connect to Pixa: {e}")
            return None

    async def monitor_connection(self):
        consecutive_failures = 0
        max_failures = 3

        while consecutive_failures < max_failures:
            if self.websocket_holder["websocket"] is None or self.websocket_holder["websocket"].state != websockets.protocol.State.OPEN:
                logger.info("Re-establishing Pixa connection...")
                result = await self.establish_connection()
                if result is None:
                    consecutive_failures += 1
                    logger.warning(f"Pixa connection failed (attempt {consecutive_failures}/{max_failures})")
                    if consecutive_failures >= max_failures:
                        logger.error("Max connection failures reached for Pixa - stopping reconnection attempts")
                        break
                else:
                    self.websocket_holder["websocket"] = result
                    consecutive_failures = 0
            await asyncio.sleep(1)

    def update_context(self, meta_info):
        self.context_id = str(uuid.uuid4())

    async def push(self, message):
        if self.stream:
            meta_info, text, self.current_text = message.get("meta_info"), message.get("data"), message.get("data")
            self.synthesized_characters += len(text) if text is not None else 0
            end_of_llm_stream = "end_of_llm_stream" in meta_info and meta_info["end_of_llm_stream"]
            self.meta_info = copy.deepcopy(meta_info)
            meta_info["text"] = text

            if not self.context_id:
                self.update_context(meta_info)

            try:
                self.current_turn_start_time = time.perf_counter()
                self.current_turn_id = meta_info.get('turn_id') or meta_info.get('sequence_id')
            except Exception:
                pass

            self.sender_task = asyncio.create_task(self.sender(text, meta_info.get('sequence_id'), end_of_llm_stream))
            self.text_queue.append(meta_info)
        else:
            self.internal_queue.put_nowait(message)

    async def cleanup(self):
        self.conversation_ended = True
        logger.info("Cleaning up Pixa synthesizer tasks")

        if self.sender_task:
            try:
                self.sender_task.cancel()
                await self.sender_task
            except asyncio.CancelledError:
                logger.info("Sender task was successfully cancelled during WebSocket cleanup.")
            except Exception as e:
                logger.error(f"Error cancelling sender task: {e}")

        try:
            if self.websocket_holder["websocket"]:
                await self.websocket_holder["websocket"].close()
        except Exception as e:
            logger.error(f"Error closing WebSocket: {e}")
        finally:
            self.websocket_holder["websocket"] = None
        logger.info("Pixa WebSocket connection closed.")
