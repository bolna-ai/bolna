import asyncio
import base64
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

# Inworld TTS API URLs
INWORLD_BASE_URL = os.getenv('INWORLD_BASE_URL', "api.inworld.ai")
INWORLD_TTS_HTTP_URL = f"https://{INWORLD_BASE_URL}/tts/v1/voice:stream"
INWORLD_TTS_WS_URL = f"wss://{INWORLD_BASE_URL}/tts/v1/voice:streamBidirectional"


class InworldSynthesizer(BaseSynthesizer):
    def __init__(self, voice_id, voice, audio_format="pcm", sampling_rate="8000", stream=False, buffer_size=400, caching=True,
                 model="inworld-tts-1", temperature=1.1, **kwargs):
        super().__init__(kwargs.get("task_manager_instance", None), stream, buffer_size)
        self.voice = voice
        self.voice_id = voice_id  # Inworld voice name, e.g., "Dennis"
        self.sample_rate = int(sampling_rate)
        self.model = model  # "inworld-tts-1" or "inworld-tts-1-max"
        self.temperature = temperature
        self.first_chunk_generated = False
        
        # Inworld uses Basic auth with base64 encoded credentials
        self.api_key = kwargs.get("synthesizer_key", os.getenv('INWORLD_API_KEY'))
        
        # Map audio format to Inworld's encoding enum
        self.use_mulaw = kwargs.get("use_mulaw", False)
        if self.use_mulaw or audio_format in ["pcm", "wav"]:
            self.format = "MULAW"
            self.audio_encoding = "MULAW"
        elif audio_format == "mp3":
            self.format = "MP3"
            self.audio_encoding = "MP3"
        elif audio_format == "opus":
            self.format = "OGG_OPUS"
            self.audio_encoding = "OGG_OPUS"
        elif audio_format == "flac":
            self.format = "FLAC"
            self.audio_encoding = "FLAC"
        elif audio_format == "alaw":
            self.format = "ALAW"
            self.audio_encoding = "ALAW"
        else:
            # Default to LINEAR16 for pcm
            self.format = "LINEAR16"
            self.audio_encoding = "LINEAR16"
        
        self.synthesized_characters = 0
        self.caching = caching
        if caching:
            self.cache = InmemoryScalarCache()

        # WebSocket streaming support
        self.stream = stream
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
        
        # Inworld WebSocket context management
        self.context_id = None
        self.context_created = False

    def get_synthesized_characters(self):
        return self.synthesized_characters
    
    def get_engine(self):
        return self.model

    def _get_auth_header(self):
        """Get the Authorization header for Inworld API"""
        # Inworld expects Basic auth with base64 encoded key
        return f"Basic {self.api_key}"

    async def __generate_http(self, text):
        """Generate audio using Inworld HTTP streaming endpoint"""
        headers = {
            "Authorization": self._get_auth_header(),
            "Content-Type": "application/json"
        }

        payload = {
            "text": text,
            "voiceId": self.voice_id,
            "modelId": self.model,
            "audioConfig": {
                "audioEncoding": self.audio_encoding,
                "sampleRateHertz": self.sample_rate
            },
            "temperature": self.temperature
        }

        logger.info(f"Sending Inworld TTS request to {INWORLD_TTS_HTTP_URL}")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(INWORLD_TTS_HTTP_URL, headers=headers, json=payload) as response:
                    if response.status == 200:
                        # Inworld returns streaming JSON with audioContent (base64 encoded)
                        audio_chunks = []
                        async for line in response.content:
                            if line:
                                try:
                                    data = json.loads(line.decode('utf-8'))
                                    if 'result' in data and 'audioContent' in data['result']:
                                        # Decode base64 audio content
                                        audio_chunk = base64.b64decode(data['result']['audioContent'])
                                        audio_chunks.append(audio_chunk)
                                except json.JSONDecodeError:
                                    continue
                        
                        if audio_chunks:
                            return b''.join(audio_chunks)
                        else:
                            logger.warning("No audio content received from Inworld")
                            return b'\x00'
                    else:
                        error_body = await response.text()
                        logger.error(f"Inworld TTS request failed with status {response.status}: {error_body}")
                        return b'\x00'
        except Exception as e:
            logger.error(f"Error in Inworld HTTP synthesis: {e}")
            traceback.print_exc()
            return b'\x00'

    def supports_websocket(self):
        return True

    def get_sleep_time(self):
        return 0.01 if self.stream else super().get_sleep_time()

    async def open_connection(self):
        pass

    async def _create_context(self):
        """Create a new TTS context on the WebSocket connection"""
        if self.websocket_holder["websocket"] is None:
            logger.warning("Cannot create context - WebSocket not connected")
            return False
        
        self.context_id = f"ctx-{uuid.uuid4().hex[:8]}"
        
        create_context_message = {
            "create_context": {
                "contextId": self.context_id,
                "voiceId": self.voice_id,
                "modelId": self.model,
                "audioConfig": {
                    "audioEncoding": self.audio_encoding,
                    "sampleRateHertz": self.sample_rate
                },
                "temperature": self.temperature
            }
        }
        
        try:
            await self.websocket_holder["websocket"].send(json.dumps(create_context_message))
            logger.info(f"Sent create_context for contextId={self.context_id}")
            return True
        except Exception as e:
            logger.error(f"Error creating context: {e}")
            return False

    async def _close_context(self):
        """Close the current TTS context"""
        if self.websocket_holder["websocket"] is None or self.context_id is None:
            return
        
        close_context_message = {
            "close_context": {
                "contextId": self.context_id
            }
        }
        
        try:
            await self.websocket_holder["websocket"].send(json.dumps(close_context_message))
            logger.info(f"Sent close_context for contextId={self.context_id}")
            self.context_id = None
            self.context_created = False
        except Exception as e:
            logger.error(f"Error closing context: {e}")

    async def handle_interruption(self):
        """Handle interruption by closing and recreating the context"""
        try:
            if self.websocket_holder["websocket"] is not None and self.websocket_holder["websocket"].state is websockets.protocol.State.OPEN:
                # Close the current context to discard buffered text
                await self._close_context()
                # Create a new context for future synthesis
                await self._create_context()
                logger.info("Inworld TTS context reset for interruption handling")
        except Exception as e:
            logger.error(f"Error handling interruption: {e}")

    def form_payload(self, text):
        """Form the send_text payload for WebSocket"""
        payload = {
            "send_text": {
                "contextId": self.context_id,
                "text": text
            }
        }
        return payload

    async def sender(self, text, sequence_id, end_of_llm_stream=False):
        """Send text to Inworld WebSocket for TTS generation"""
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
                logger.info("Waiting for Inworld TTS WebSocket connection to be established...")
                await asyncio.sleep(0.5)
            ws_wait_time = (time.perf_counter() - ws_wait_start) * 1000
            if ws_wait_time > 10:
                logger.info(f"Inworld sender ws_wait={ws_wait_time:.0f}ms")

            # Ensure context is created
            if not self.context_created:
                await self._create_context()
                # Wait a bit for context_created response
                await asyncio.sleep(0.1)
                self.context_created = True

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
                        logger.info(f"Inworld WS send first_text_sent")
                    
                    send_text_message = self.form_payload(text)
                    await self.websocket_holder["websocket"].send(json.dumps(send_text_message))
                except Exception as e:
                    logger.error(f"Error sending chunk to Inworld: {e}")
                    return

            # If end_of_llm_stream is True, flush the buffer
            if end_of_llm_stream:
                self.last_text_sent = True
                try:
                    # Send flush_context to synthesize all accumulated text
                    flush_message = {
                        "flush_context": {
                            "contextId": self.context_id
                        }
                    }
                    await self.websocket_holder["websocket"].send(json.dumps(flush_message))
                    logger.info(f"Sent flush_context to Inworld TTS WebSocket for contextId={self.context_id}")
                except Exception as e:
                    logger.error(f"Error sending flush_context to Inworld: {e}")

        except asyncio.CancelledError:
            logger.info("Inworld sender task was cancelled.")
        except Exception as e:
            logger.error(f"Unexpected error in Inworld sender: {e}")

    async def receiver(self):
        """Receive audio chunks from Inworld WebSocket"""
        audio_chunk_count = 0
        while True:
            try:
                if self.conversation_ended:
                    return

                if (self.websocket_holder["websocket"] is None or
                        self.websocket_holder["websocket"].state is websockets.protocol.State.CLOSED):
                    logger.info("Inworld WebSocket is not connected, skipping receive.")
                    await asyncio.sleep(0.10)
                    continue

                recv_start = time.perf_counter()
                response = await self.websocket_holder["websocket"].recv()
                recv_duration = (time.perf_counter() - recv_start) * 1000

                # Inworld sends JSON responses
                try:
                    data = json.loads(response)
                    
                    if "context_created" in data:
                        context_info = data["context_created"]
                        logger.info(f"Inworld TTS context created: {context_info.get('contextId', 'unknown')}")
                        self.context_created = True
                    
                    elif "audio_chunk" in data:
                        # Audio data chunk - decode base64 audio content
                        chunk_data = data["audio_chunk"]
                        if "audioContent" in chunk_data:
                            audio_chunk_count += 1
                            audio_bytes = base64.b64decode(chunk_data["audioContent"])
                            
                            if audio_chunk_count == 1 and self.ws_send_time is not None:
                                time_since_send = (time.perf_counter() - self.ws_send_time) * 1000
                                logger.info(f"Inworld WS recv FIRST audio chunk recv_wait={recv_duration:.0f}ms time_since_send={time_since_send:.0f}ms")
                            
                            yield audio_bytes
                    
                    elif "flush_completed" in data:
                        logger.info(f"Inworld TTS flush completed for contextId={data.get('flush_completed', {}).get('contextId', 'unknown')}")
                        audio_chunk_count = 0
                        yield b'\x00'  # Signal end of stream
                    
                    elif "context_closed" in data:
                        logger.info(f"Inworld TTS context closed: {data.get('context_closed', {}).get('contextId', 'unknown')}")
                        audio_chunk_count = 0
                        self.context_created = False
                    
                    elif "context_updated" in data:
                        logger.info(f"Inworld TTS context updated: {data.get('context_updated', {}).get('contextId', 'unknown')}")
                    
                    elif "error" in data:
                        error_info = data["error"]
                        logger.error(f"Inworld TTS error: code={error_info.get('code')}, message={error_info.get('message')}")
                    
                    else:
                        logger.info(f"Inworld TTS response: {data}")
                
                except json.JSONDecodeError:
                    logger.warning(f"Received unexpected non-JSON response from Inworld")

            except websockets.exceptions.ConnectionClosed:
                logger.info("Inworld WebSocket connection closed")
                break
            except Exception as e:
                logger.error(f"Error in Inworld receiver: {e}")
                traceback.print_exc()

    async def synthesize(self, text):
        # This is used for one-off synthesis mainly for use cases like voice lab and IVR
        try:
            audio = await self.__generate_http(text)
            if self.format == "MP3":
                audio = convert_audio_to_wav(audio, source_format="mp3")
            return audio
        except Exception as e:
            logger.error(f"Could not synthesize: {e}")

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
                        self.meta_info['format'] = self.format.lower()
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
                        logger.info("Inworld received null byte and hence end of stream")
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

                    if self.format == "MP3":
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
                        meta_info['format'] = self.format.lower()
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
            logger.error(f"Error in Inworld generate: {e}")

    async def establish_connection(self):
        """Establish WebSocket connection to Inworld TTS"""
        try:
            start_time = time.perf_counter()
            additional_headers = {
                "Authorization": self._get_auth_header()
            }
            websocket = await asyncio.wait_for(
                websockets.connect(INWORLD_TTS_WS_URL, additional_headers=additional_headers),
                timeout=10.0
            )
            if not self.connection_time:
                self.connection_time = round((time.perf_counter() - start_time) * 1000)
            logger.info(f"Connected to Inworld TTS WebSocket: {INWORLD_TTS_WS_URL}")
            return websocket
        except asyncio.TimeoutError:
            logger.error("Timeout while connecting to Inworld TTS WebSocket")
            return None
        except websockets.exceptions.InvalidStatusCode as e:
            if e.status_code == 401:
                logger.error(f"Inworld authentication failed: Invalid API key")
            elif e.status_code == 403:
                logger.error(f"Inworld authentication failed: Access forbidden")
            else:
                logger.error(f"Inworld WebSocket connection failed with status {e.status_code}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to connect to Inworld TTS WebSocket: {e}")
            return None

    async def monitor_connection(self):
        """Periodically check if the connection is still alive and reconnect if needed"""
        consecutive_failures = 0
        max_failures = 3

        while consecutive_failures < max_failures:
            if self.websocket_holder["websocket"] is None or self.websocket_holder["websocket"].state is websockets.protocol.State.CLOSED:
                logger.info("Re-establishing Inworld TTS connection...")
                result = await self.establish_connection()
                if result is None:
                    consecutive_failures += 1
                    logger.warning(f"Inworld TTS connection failed (attempt {consecutive_failures}/{max_failures})")
                    if consecutive_failures >= max_failures:
                        logger.error("Max connection failures reached for Inworld TTS - stopping reconnection attempts")
                        break
                else:
                    self.websocket_holder["websocket"] = result
                    consecutive_failures = 0  # Reset on success
                    # Create context after successful connection
                    self.context_created = False
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
                    logger.info(f"Inworld push new_turn text_len={len(text) if text else 0}")
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
        logger.info("Cleaning up Inworld synthesizer tasks")
        
        if self.sender_task:
            try:
                self.sender_task.cancel()
                await self.sender_task
            except asyncio.CancelledError:
                logger.info("Inworld sender task was successfully cancelled during cleanup.")
            except Exception as e:
                logger.error(f"Error cancelling sender task: {e}")

        if self.websocket_holder["websocket"]:
            try:
                # Close the context before closing WebSocket
                await self._close_context()
            except Exception as e:
                logger.error(f"Error closing context: {e}")
            try:
                await self.websocket_holder["websocket"].close()
            except Exception as e:
                logger.error(f"Error closing WebSocket: {e}")
        
        self.websocket_holder["websocket"] = None
        logger.info("Inworld TTS WebSocket connection closed.")
