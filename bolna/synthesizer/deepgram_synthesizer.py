import asyncio
import copy
import time
import aiohttp
import websockets
import json
import os
import uuid
import traceback
from collections import deque
from dotenv import load_dotenv
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.utils import convert_audio_to_wav, create_ws_data_packet
from bolna.memory.cache.inmemory_scalar_cache import InmemoryScalarCache
from .base_synthesizer import BaseSynthesizer

logger = configure_logger(__name__)
load_dotenv()
DEEPGRAM_HOST = os.getenv('DEEPGRAM_HOST', 'api.deepgram.com')
DEEPGRAM_TTS_URL = "https://{}/v1/speak".format(DEEPGRAM_HOST)
DEEPGRAM_TTS_WS_URL = "wss://{}/v1/speak".format(DEEPGRAM_HOST)


class DeepgramSynthesizer(BaseSynthesizer):
    def __init__(self, voice_id, voice, audio_format="pcm", sampling_rate="8000", stream=False, buffer_size=400, caching=True,
                 model="aura-zeus-en", **kwargs):
        super().__init__(kwargs.get("task_manager_instance", None), stream, buffer_size)
        self.voice = voice
        self.voice_id = voice_id
        self.sample_rate = str(sampling_rate)
        self.model = model
        self.first_chunk_generated = False
        self.api_key = kwargs.get("transcriber_key", os.getenv('DEEPGRAM_AUTH_TOKEN'))
        
        # For telephony (8000 sample rate), use mulaw encoding
        # For WebSocket streaming, Deepgram supports: linear16, mulaw, alaw
        self.use_mulaw = kwargs.get("use_mulaw", False)
        if self.use_mulaw or audio_format in ["pcm", "wav"]:
            self.format = "mulaw"
        else:
            self.format = audio_format

        if len(self.model.split('-')) == 2:
            self.model = f"{self.model}-{self.voice_id}"
        
        self.synthesized_characters = 0
        self.caching = caching
        if caching:
            self.cache = InmemoryScalarCache()

        # WebSocket streaming support
        self.stream = stream
        self.ws_url = f"{DEEPGRAM_TTS_WS_URL}?encoding={self.format}&sample_rate={self.sample_rate}&model={self.model}"
        self.websocket_holder = {"websocket": None}
        self.text_queue = deque()
        self.meta_info = None
        self.last_text_sent = False
        self.sender_task = None
        self.conversation_ended = False
        self.current_turn_start_time = None
        self.current_turn_id = None
        self.current_text = ""
        self.ws_send_time = None
        self.current_turn_ttfb = None

    def get_synthesized_characters(self):
        return self.synthesized_characters
    
    def get_engine(self):
        return self.model

    async def __generate_http(self, text):
        headers = {
            "Authorization": "Token {}".format(self.api_key),
            "Content-Type": "application/json"
        }
        url = DEEPGRAM_TTS_URL + "?container=none&encoding={}&sample_rate={}&model={}".format(
            self.format, self.sample_rate, self.model
        )

        logger.info(f"Sending deepgram request {url}")

        payload = {
            "text": text
        }
        try:
            async with aiohttp.ClientSession() as session:
                if payload is not None:
                    async with session.post(url, headers=headers, json=payload) as response:
                        if response.status == 200:
                            chunk = await response.read()
                            logger.info(f"status for deepgram request {response.status} response {len(await response.read())}")
                            return chunk
                        else:
                            logger.info(f"status for deepgram reques {response.status} response {await response.read()}")
                            return b'\x00'
                else:
                    logger.info("Payload was null")
        except Exception as e:
            logger.error("something went wrong")

    def supports_websocket(self):
        return True

    def get_sleep_time(self):
        return 0.01 if self.stream else super().get_sleep_time()

    async def open_connection(self):
        pass

    async def handle_interruption(self):
        """Handle interruption by clearing the buffer on Deepgram's side"""
        try:
            if self.websocket_holder["websocket"] is not None and self.websocket_holder["websocket"].state is websockets.protocol.State.OPEN:
                # Send Clear message to Deepgram to discard buffered text
                clear_message = {"type": "Clear"}
                await self.websocket_holder["websocket"].send(json.dumps(clear_message))
                logger.info("Sent Clear message to Deepgram TTS WebSocket")
        except Exception as e:
            logger.error(f"Error handling interruption: {e}")

    def form_payload(self, text):
        payload = {
            "type": "Speak",
            "text": text
        }
        return payload

    async def sender(self, text, sequence_id, end_of_llm_stream=False):
        """Send text to Deepgram WebSocket for TTS generation"""
        try:
            if self.conversation_ended:
                return

            if not self.should_synthesize_response(sequence_id):
                logger.info(
                    f"Not synthesizing text as the sequence_id ({sequence_id}) is not in the list of sequence_ids present in the task manager.")
                await self.flush_synthesizer_stream()
                return

            # Wait for WebSocket connection to be established
            ws_wait_start = time.perf_counter()
            while self.websocket_holder["websocket"] is None or self.websocket_holder["websocket"].state is websockets.protocol.State.CLOSED:
                logger.info("Waiting for Deepgram TTS WebSocket connection to be established...")
                await asyncio.sleep(0.5)
            ws_wait_time = (time.perf_counter() - ws_wait_start) * 1000
            if ws_wait_time > 10:
                logger.info(f"Deepgram sender ws_wait={ws_wait_time:.0f}ms")

            if text != "":
                if not self.should_synthesize_response(sequence_id):
                    logger.info(
                        f"Not synthesizing text as the sequence_id ({sequence_id}) of it is not in the list of sequence_ids present in the task manager (inner loop).")
                    await self.flush_synthesizer_stream()
                    return
                try:
                    # Capture WebSocket send time on first send
                    if self.ws_send_time is None:
                        self.ws_send_time = time.perf_counter()
                        logger.info(f"Deepgram WS send first_text_sent")
                    speak_message = self.form_payload(text)
                    await self.websocket_holder["websocket"].send(json.dumps(speak_message))
                except Exception as e:
                    logger.error(f"Error sending chunk to Deepgram: {e}")
                    return

            # If end_of_llm_stream is True, flush the buffer and mark stream as complete
            if end_of_llm_stream:
                self.last_text_sent = True
                try:
                    # Send Flush to get all remaining audio
                    flush_message = {"type": "Flush"}
                    await self.websocket_holder["websocket"].send(json.dumps(flush_message))
                    logger.info("Sent Flush message to Deepgram TTS WebSocket")
                except Exception as e:
                    logger.error(f"Error sending Flush to Deepgram: {e}")

        except asyncio.CancelledError:
            logger.info("Deepgram sender task was cancelled.")
        except Exception as e:
            logger.error(f"Unexpected error in Deepgram sender: {e}")

    async def receiver(self):
        """Receive audio chunks from Deepgram WebSocket"""
        audio_chunk_count = 0
        while True:
            try:
                if self.conversation_ended:
                    return

                if (self.websocket_holder["websocket"] is None or
                        self.websocket_holder["websocket"].state is websockets.protocol.State.CLOSED):
                    logger.info("Deepgram WebSocket is not connected, skipping receive.")
                    await asyncio.sleep(0.10)
                    continue

                recv_start = time.perf_counter()
                response = await self.websocket_holder["websocket"].recv()
                recv_duration = (time.perf_counter() - recv_start) * 1000

                # Deepgram sends binary audio data directly, or JSON for metadata
                if isinstance(response, bytes):
                    # Binary audio data
                    audio_chunk_count += 1
                    if audio_chunk_count == 1 and self.ws_send_time is not None:
                        time_since_send = (time.perf_counter() - self.ws_send_time) * 1000
                        logger.info(f"Deepgram WS recv FIRST audio chunk recv_wait={recv_duration:.0f}ms time_since_send={time_since_send:.0f}ms")
                    yield response
                else:
                    # JSON metadata response
                    try:
                        data = json.loads(response)
                        msg_type = data.get("type", "")
                        
                        if msg_type == "Metadata":
                            logger.info(f"Deepgram TTS Metadata: request_id={data.get('request_id')}")
                        elif msg_type == "Flushed":
                            logger.info(f"Deepgram TTS Flushed: sequence_id={data.get('sequence_id')}")
                            # Flushed indicates all audio for current text has been sent
                            audio_chunk_count = 0
                            yield b'\x00'  # Signal end of stream
                        elif msg_type == "Cleared":
                            logger.info(f"Deepgram TTS Cleared: sequence_id={data.get('sequence_id')}")
                            audio_chunk_count = 0
                        elif msg_type == "Warning":
                            logger.info(f"Deepgram TTS Warning: {data.get('description')}")
                        else:
                            logger.info(f"Deepgram TTS response: {data}")
                    except json.JSONDecodeError:
                        logger.warning(f"Received unexpected non-JSON text response from Deepgram")

            except websockets.exceptions.ConnectionClosed:
                logger.info("Deepgram WebSocket connection closed")
                break
            except Exception as e:
                logger.error(f"Error in Deepgram receiver: {e}")
                traceback.print_exc()    

    async def synthesize(self, text):
        # This is used for one off synthesis mainly for use cases like voice lab and IVR
        try:
            audio = await self.__generate_http(text)
            if self.format == "mp3":
                audio = convert_audio_to_wav(audio, source_format="mp3")
            return audio
        except Exception as e:
            logger.error(f"Could not synthesize {e}")

    async def generate(self):
        try:
            if self.stream:
                # WebSocket streaming mode
                async for message in self.receiver():
                    if len(self.text_queue) > 0:
                        self.meta_info = self.text_queue.popleft()
                        # Compute TTFB on first audio chunk only
                        try:
                            if self.current_turn_ttfb is None and self.ws_send_time is not None:
                                self.current_turn_ttfb = time.perf_counter() - self.ws_send_time
                                self.meta_info['synthesizer_latency'] = self.current_turn_ttfb
                        except Exception:
                            pass

                    if self.use_mulaw:
                        self.meta_info['format'] = 'mulaw'
                    else:
                        self.meta_info['format'] = self.format
                    audio = message

                    if not self.first_chunk_generated:
                        self.meta_info["is_first_chunk"] = True
                        self.first_chunk_generated = True
                    else:
                        self.meta_info["is_first_chunk"] = False

                    if self.last_text_sent:
                        # Reset the last_text_sent and first_chunk converted to reset synth latency
                        self.first_chunk_generated = False
                        self.last_text_sent = True

                    if message == b'\x00':
                        logger.info("Deepgram received null byte and hence end of stream")
                        self.meta_info["end_of_synthesizer_stream"] = True
                        self.first_chunk_generated = False
                        # Compute total stream duration for this synthesizer turn
                        try:
                            if self.current_turn_start_time is not None:
                                total_stream_duration = time.perf_counter() - self.current_turn_start_time
                                self.turn_latencies.append({
                                    'turn_id': self.current_turn_id,
                                    'sequence_id': self.current_turn_id,
                                    'first_result_latency_ms': round((self.current_turn_ttfb or 0) * 1000),
                                    'total_stream_duration_ms': round(total_stream_duration * 1000)
                                })
                                self.current_turn_start_time = None
                                self.current_turn_id = None
                                self.ws_send_time = None  # Reset for next turn
                                self.current_turn_ttfb = None  # Reset for next turn
                        except Exception:
                            pass

                    self.meta_info["mark_id"] = str(uuid.uuid4())
                    yield create_ws_data_packet(audio, self.meta_info)
            else:
                # HTTP non-streaming mode
                while True:
                    message = await self.internal_queue.get()
                    logger.info(f"Generating TTS response for message: {message}")
                    meta_info, text = message.get("meta_info"), message.get("data")
                    # Stamp synthesizer turn start time for HTTP flow
                    try:
                        meta_info['synthesizer_start_time'] = time.perf_counter()
                    except Exception:
                        pass
                    if not self.should_synthesize_response(meta_info.get('sequence_id')):
                        logger.info(f"Not synthesizing text as the sequence_id ({meta_info.get('sequence_id')}) of it is not in the list of sequence_ids present in the task manager.")
                        return
                    if self.caching:
                        logger.info(f"Caching is on")
                        if self.cache.get(text):
                            logger.info(f"Cache hit and hence returning quickly {text}")
                            audio_message = self.cache.get(text)
                        else:
                            logger.info(f"Not a cache hit {list(self.cache.data_dict)}")
                            self.synthesized_characters += len(text)
                            audio_message = await self.__generate_http(text)
                            self.cache.set(text, audio_message)
                    else:
                        logger.info(f"No caching present")
                        self.synthesized_characters += len(text)
                        audio_message = await self.__generate_http(text)

                    if self.format == "mp3":
                        audio_message = convert_audio_to_wav(audio_message, source_format="mp3")
                    if not self.first_chunk_generated:
                        meta_info["is_first_chunk"] = True
                        self.first_chunk_generated = True
                    else:
                        meta_info["is_first_chunk"] = False
                    if "end_of_llm_stream" in meta_info and meta_info["end_of_llm_stream"]:
                        meta_info["end_of_synthesizer_stream"] = True
                        self.first_chunk_generated = False
                    meta_info['text'] = text
                    # Compute first-result latency for HTTP (first and only audio message)
                    try:
                        if 'synthesizer_start_time' in meta_info and 'synthesizer_first_result_latency' not in meta_info:
                            meta_info['synthesizer_first_result_latency'] = time.perf_counter() - meta_info['synthesizer_start_time']
                            meta_info['synthesizer_latency'] = meta_info['synthesizer_first_result_latency']
                    except Exception:
                        pass
                    if self.use_mulaw:
                        meta_info['format'] = 'mulaw'
                    else:
                        meta_info['format'] = self.format
                    meta_info["text_synthesized"] = f"{text} "
                    meta_info["mark_id"] = str(uuid.uuid4())
                    # Compute total stream duration (HTTP single-shot)
                    try:
                        if 'synthesizer_start_time' in meta_info:
                            meta_info['synthesizer_total_stream_duration'] = time.perf_counter() - meta_info['synthesizer_start_time']
                    except Exception:
                        pass
                    yield create_ws_data_packet(audio_message, meta_info)
        except Exception as e:
            traceback.print_exc()
            logger.error(f"Error in Deepgram generate: {e}")

    async def establish_connection(self):
        """Establish WebSocket connection to Deepgram TTS"""
        try:
            start_time = time.perf_counter()
            additional_headers = {
                "Authorization": f"Token {self.api_key}"
            }
            websocket = await asyncio.wait_for(
                websockets.connect(self.ws_url, additional_headers=additional_headers),
                timeout=10.0
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
                logger.error(f"Deepgram authentication failed: Invalid API key")
            elif e.status_code == 403:
                logger.error(f"Deepgram authentication failed: Access forbidden")
            else:
                logger.error(f"Deepgram WebSocket connection failed with status {e.status_code}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to connect to Deepgram TTS WebSocket: {e}")
            return None

    async def monitor_connection(self):
        """Periodically check if the connection is still alive and reconnect if needed"""
        consecutive_failures = 0
        max_failures = 3

        while consecutive_failures < max_failures:
            if self.websocket_holder["websocket"] is None or self.websocket_holder["websocket"].state is websockets.protocol.State.CLOSED:
                logger.info("Re-establishing Deepgram TTS connection...")
                result = await self.establish_connection()
                if result is None:
                    consecutive_failures += 1
                    logger.warning(f"Deepgram TTS connection failed (attempt {consecutive_failures}/{max_failures})")
                    if consecutive_failures >= max_failures:
                        logger.error("Max connection failures reached for Deepgram TTS - stopping reconnection attempts")
                        break
                else:
                    self.websocket_holder["websocket"] = result
                    consecutive_failures = 0  # Reset on success
            await asyncio.sleep(1)

    async def push(self, message):
        if self.stream:
            meta_info, text, self.current_text = message.get("meta_info"), message.get("data"), message.get("data")
            self.synthesized_characters += len(text) if text is not None else 0
            end_of_llm_stream = "end_of_llm_stream" in meta_info and meta_info["end_of_llm_stream"]
            self.meta_info = copy.deepcopy(meta_info)
            meta_info["text"] = text

            # Stamp synthesizer turn start time
            try:
                if self.current_turn_start_time is None:
                    self.current_turn_start_time = time.perf_counter()
                    self.ws_send_time = None  # Reset for new turn
                    self.current_turn_ttfb = None  # Reset for new turn
                    logger.info(f"Deepgram push new_turn text_len={len(text) if text else 0}")
                self.current_turn_id = meta_info.get('turn_id') or meta_info.get('sequence_id')
            except Exception:
                pass

            self.sender_task = asyncio.create_task(self.sender(text, meta_info.get('sequence_id'), end_of_llm_stream))
            self.text_queue.append(meta_info)
        else:
            logger.info(f"Pushed message to internal queue {message}")
            self.internal_queue.put_nowait(copy.deepcopy(message))

    async def cleanup(self):
        """Clean up WebSocket connection and tasks"""
        self.conversation_ended = True
        logger.info("Cleaning up Deepgram synthesizer tasks")
        
        if self.sender_task:
            try:
                self.sender_task.cancel()
                await self.sender_task
            except asyncio.CancelledError:
                logger.info("Deepgram sender task was successfully cancelled during cleanup.")
            except Exception as e:
                logger.error(f"Error cancelling sender task: {e}")

        if self.websocket_holder["websocket"]:
            try:
                # Send Close message for graceful shutdown
                close_message = {"type": "Close"}
                await self.websocket_holder["websocket"].send(json.dumps(close_message))
                logger.info("Sent Close message to Deepgram TTS WebSocket")
            except Exception as e:
                logger.error(f"Error sending Close message: {e}")
            try:
                await self.websocket_holder["websocket"].close()
            except Exception as e:
                logger.error(f"Error closing WebSocket: {e}")
        
        self.websocket_holder["websocket"] = None
        logger.info("Deepgram TTS WebSocket connection closed.")
