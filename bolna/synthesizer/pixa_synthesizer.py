import asyncio
import copy
import time
import websockets
import json
import os
import uuid
import audioop
import traceback
from collections import deque
from dotenv import load_dotenv
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.utils import create_ws_data_packet
from bolna.memory.cache.inmemory_scalar_cache import InmemoryScalarCache
from .base_synthesizer import BaseSynthesizer

logger = configure_logger(__name__)
load_dotenv()


class PixaSynthesizer(BaseSynthesizer):
    """
    Pixa (Luna) TTS Synthesizer - WebSocket Only
    
    API: wss://hindi.heypixa.ai/api/v1/ws/synthesize
    
    Audio format: PCM16 (16-bit signed integer), 32kHz, mono, little-endian
    
    Key characteristics:
    - WebSocket streaming with ~300ms latency
    - 32kHz high-quality audio
    - Primarily for Hindi language
    - No authentication required (based on provided docs)
    """

    def __init__(
        self,
        voice_id,
        voice,
        audio_format="pcm",
        sampling_rate="32000",  # Pixa native is 32kHz
        stream=True,
        buffer_size=400,
        caching=True,
        model="luna-tts",
        language="hi",
        **kwargs
    ):
        super().__init__(kwargs.get("task_manager_instance", None), stream, buffer_size)
        self.voice = voice
        self.voice_id = voice_id
        self.sample_rate = str(sampling_rate)
        self.model = model
        self.language = language
        self.first_chunk_generated = False
        self.api_host = os.getenv("PIXA_TTS_HOST", "hindi.heypixa.ai")
        self.top_p = kwargs.get("top_p", 0.95)
        self.repetition_penalty = kwargs.get("repetition_penalty", 1.3)
        
        # Pixa native format is PCM16 32kHz
        self.format = "pcm16"
        
        self.synthesized_characters = 0
        self.caching = caching
        if caching:
            self.cache = InmemoryScalarCache()

        self.stream = stream
        self.ws_url = f"wss://{self.api_host}/api/v1/ws/synthesize"
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

    def supports_websocket(self):
        return True

    def get_sleep_time(self):
        return 0.01 if self.stream else super().get_sleep_time()
    
    def resample_audio(self, audio_bytes, from_rate=32000, to_rate=8000):
        """Resample PCM16 audio from Pixa's 32kHz to telephony's 8kHz"""
        if len(audio_bytes) == 0:
            return audio_bytes
        try:
            # audioop.ratecv(fragment, width, nchannels, inrate, outrate, state)
            # width=2 for 16-bit PCM, nchannels=1 for mono
            resampled, _ = audioop.ratecv(audio_bytes, 2, 1, from_rate, to_rate, None)
            return resampled
        except Exception as e:
            logger.error(f"Error resampling audio: {e}")
            return audio_bytes

    async def open_connection(self):
        pass

    async def handle_interruption(self):
        """Handle interruption by closing and reconnecting WebSocket"""
        try:
            if self.websocket_holder["websocket"] is not None:
                logger.info("Handling interruption in Pixa TTS WebSocket")
                try:
                    await self.websocket_holder["websocket"].close()
                except Exception:
                    pass
                self.websocket_holder["websocket"] = None
        except Exception as e:
            logger.error(f"Error handling interruption: {e}")

    async def sender_ws(self, text, sequence_id, end_of_llm_stream=False):
        """Send text to Pixa WebSocket for TTS generation"""
        try:
            if self.conversation_ended:
                return

            if not self.should_synthesize_response(sequence_id):
                logger.info(
                    f"Not synthesizing text as the sequence_id ({sequence_id}) is not in the list")
                await self.flush_synthesizer_stream()
                return

            # Wait for WebSocket connection to be established
            ws_wait_start = time.perf_counter()
            while self.websocket_holder["websocket"] is None or self.websocket_holder["websocket"].state is websockets.protocol.State.CLOSED:
                logger.info("Waiting for Pixa TTS WebSocket connection...")
                await asyncio.sleep(0.5)
            ws_wait_time = (time.perf_counter() - ws_wait_start) * 1000
            if ws_wait_time > 10:
                logger.info(f"Pixa sender ws_wait={ws_wait_time:.0f}ms")

            if text != "":
                if not self.should_synthesize_response(sequence_id):
                    logger.info(f"Not synthesizing (inner loop check)")
                    await self.flush_synthesizer_stream()
                    return
                
                try:
                    if self.ws_send_time is None:
                        self.ws_send_time = time.perf_counter()
                        logger.info(f"Pixa WS: Sending first text chunk")

                    message = {
                        "type": "text",
                        "content": text,
                        "is_final": end_of_llm_stream
                    }
                    await self.websocket_holder["websocket"].send(json.dumps(message))
                    logger.info(f"Pixa TTS WS: Sent text chunk ({len(text)} chars, is_final={end_of_llm_stream})")
                except Exception as e:
                    logger.error(f"Error sending to Pixa WS: {e}")
                    return

            if end_of_llm_stream:
                self.last_text_sent = True

        except asyncio.CancelledError:
            logger.info("Pixa sender task cancelled")
        except Exception as e:
            logger.error(f"Unexpected error in Pixa sender: {e}")
            traceback.print_exc()

    async def receiver_ws(self):
        """Receive audio chunks from Pixa WebSocket"""
        audio_chunk_count = 0
        while True:
            try:
                if self.conversation_ended:
                    return

                if (self.websocket_holder["websocket"] is None or
                        self.websocket_holder["websocket"].state is websockets.protocol.State.CLOSED):
                    logger.info("Pixa WebSocket not connected, skipping receive")
                    await asyncio.sleep(0.10)
                    continue

                recv_start = time.perf_counter()
                response = await self.websocket_holder["websocket"].recv()
                recv_duration = (time.perf_counter() - recv_start) * 1000

                # Binary frames = audio data (PCM16)
                if isinstance(response, bytes):
                    audio_chunk_count += 1
                    if audio_chunk_count == 1 and self.ws_send_time is not None:
                        time_since_send = (time.perf_counter() - self.ws_send_time) * 1000
                        logger.info(f"Pixa WS: FIRST audio chunk recv_wait={recv_duration:.0f}ms time_since_send={time_since_send:.0f}ms")
                    
                    # Resample from 32kHz to 8kHz for telephony
                    resampled_audio = self.resample_audio(response, from_rate=32000, to_rate=8000)

                    yield resampled_audio
                else:
                    try:
                        data = json.loads(response)
                        msg_type = data.get("type", "")

                        if msg_type == "status":
                            logger.info(f"Pixa TTS status: {data.get('message')}")
                        elif msg_type == "done":
                            logger.info(f"Pixa TTS done: {data.get('total_audio_bytes')} bytes")
                            audio_chunk_count = 0
                            yield b'\x00'  # End of stream marker
                        elif msg_type == "error":
                            logger.error(f"Pixa TTS error: {data.get('message')}")
                            yield b'\x00'
                            break
                        else:
                            logger.info(f"Pixa TTS message: {data}")
                    except json.JSONDecodeError:
                        logger.warning(f"Received non-JSON text from Pixa WS")

            except websockets.exceptions.ConnectionClosed:
                logger.info("Pixa WebSocket connection closed")
                break
            except Exception as e:
                logger.error(f"Error in Pixa WS receiver: {e}")
                traceback.print_exc()
                break

    async def synthesize(self, text):
        """
        One-off synthesis using WebSocket.
        Opens a connection, synthesizes, and closes.
        """
        try:
            logger.info(f"Pixa TTS: One-off synthesis via WebSocket")

            ws = await websockets.connect(self.ws_url)

            config_msg = {
                "type": "config",
                "top_p": self.top_p,
                "repetition_penalty": self.repetition_penalty
            }
            await ws.send(json.dumps(config_msg))
            
            # Send text
            text_msg = {
                "type": "text",
                "content": text,
                "is_final": True
            }
            await ws.send(json.dumps(text_msg))
            logger.info(f"Pixa TTS: Sent text for synthesis")
            
            # Collect audio
            audio_data = b""
            async for message in ws:
                if isinstance(message, bytes):
                    audio_data += message
                else:
                    data = json.loads(message)
                    if data.get("type") == "done":
                        logger.info(f"Pixa TTS: Synthesis complete")
                        break
                    elif data.get("type") == "error":
                        logger.error(f"Pixa TTS error: {data.get('message')}")
                        break
            
            await ws.close()
            
            # Resample from 32kHz to 8kHz
            if audio_data and audio_data != b'\x00':
                audio_data = self.resample_audio(audio_data, from_rate=32000, to_rate=8000)
            
            return audio_data
            
        except Exception as e:
            logger.error(f"Could not synthesize: {e}")
            traceback.print_exc()
            return b'\x00'

    async def generate(self):
        """Generate audio stream"""
        try:
            if self.stream:
                # WebSocket streaming mode
                async for message in self.receiver_ws():
                    if len(self.text_queue) > 0:
                        self.meta_info = self.text_queue.popleft()
                        try:
                            if self.current_turn_ttfb is None and self.ws_send_time is not None:
                                self.current_turn_ttfb = time.perf_counter() - self.ws_send_time
                                self.meta_info['synthesizer_latency'] = self.current_turn_ttfb
                        except Exception:
                            pass

                    # Set format (Pixa native is PCM16 32kHz, resampled to 8kHz)
                    # Format is 'pcm' (not 'pcm16') for Twilio compatibility
                    self.meta_info['format'] = 'pcm'
                    self.meta_info['sample_rate'] = 8000  # After resampling
                    audio = message

                    if not self.first_chunk_generated:
                        self.meta_info["is_first_chunk"] = True
                        self.first_chunk_generated = True
                    else:
                        self.meta_info["is_first_chunk"] = False

                    if self.last_text_sent:
                        self.first_chunk_generated = False
                        self.last_text_sent = False

                    if message == b'\x00':
                        logger.info("Pixa WS: End of stream")
                        self.meta_info["end_of_synthesizer_stream"] = True
                        self.first_chunk_generated = False

                        try:
                            if self.current_turn_start_time is not None:
                                total_duration = time.perf_counter() - self.current_turn_start_time
                                self.turn_latencies.append({
                                    'turn_id': self.current_turn_id,
                                    'sequence_id': self.current_turn_id,
                                    'first_result_latency_ms': round((self.current_turn_ttfb or 0) * 1000),
                                    'total_stream_duration_ms': round(total_duration * 1000)
                                })
                                self.current_turn_start_time = None
                                self.current_turn_id = None
                                self.ws_send_time = None
                                self.current_turn_ttfb = None
                        except Exception:
                            pass

                    self.meta_info["mark_id"] = str(uuid.uuid4())
                    yield create_ws_data_packet(audio, self.meta_info)
                    
            else:
                while True:
                    message = await self.internal_queue.get()
                    logger.info(f"Pixa TTS: Processing message")
                    meta_info, text = message.get("meta_info"), message.get("data")
                    
                    try:
                        meta_info['synthesizer_start_time'] = time.perf_counter()
                    except Exception:
                        pass
                    
                    if not self.should_synthesize_response(meta_info.get('sequence_id')):
                        logger.info(f"Not synthesizing (sequence_id check)")
                        return
                    
                    if self.caching:
                        if self.cache.get(text):
                            logger.info(f"Pixa TTS: Cache hit")
                            audio_message = self.cache.get(text)
                        else:
                            logger.info(f"Pixa TTS: Cache miss, generating")
                            self.synthesized_characters += len(text)
                            audio_message = await self.synthesize(text)
                            self.cache.set(text, audio_message)
                    else:
                        self.synthesized_characters += len(text)
                        audio_message = await self.synthesize(text)
                    
                    # Resample from 32kHz to 8kHz
                    if audio_message and audio_message != b'\x00':
                        audio_message = self.resample_audio(audio_message, from_rate=32000, to_rate=8000)

                    if not self.first_chunk_generated:
                        meta_info["is_first_chunk"] = True
                        self.first_chunk_generated = True
                    else:
                        meta_info["is_first_chunk"] = False
                    
                    if "end_of_llm_stream" in meta_info and meta_info["end_of_llm_stream"]:
                        meta_info["end_of_synthesizer_stream"] = True
                        self.first_chunk_generated = False
                    
                    meta_info['text'] = text
                    
                    try:
                        if 'synthesizer_start_time' in meta_info:
                            latency = time.perf_counter() - meta_info['synthesizer_start_time']
                            meta_info['synthesizer_first_result_latency'] = latency
                            meta_info['synthesizer_latency'] = latency
                            meta_info['synthesizer_total_stream_duration'] = latency
                    except Exception:
                        pass
                    
                    # Format metadata (after resampling to 8kHz)
                    meta_info['format'] = 'pcm'
                    meta_info['sample_rate'] = 8000  # After resampling
                    meta_info["text_synthesized"] = f"{text} "
                    meta_info["mark_id"] = str(uuid.uuid4())
                    
                    yield create_ws_data_packet(audio_message, meta_info)
                    
        except Exception as e:
            traceback.print_exc()
            logger.error(f"Error in Pixa generate: {e}")

    async def establish_connection(self):
        """Establish WebSocket connection to Pixa TTS"""
        try:
            start_time = time.perf_counter()
            
            websocket = await asyncio.wait_for(
                websockets.connect(self.ws_url),
                timeout=10.0
            )
            
            if not self.connection_time:
                self.connection_time = round((time.perf_counter() - start_time) * 1000)
            
            logger.info(f"Connected to Pixa TTS WebSocket: {self.ws_url}")
            
            # Send configuration
            config_msg = {
                "type": "config",
                "top_p": self.top_p,
                "repetition_penalty": self.repetition_penalty
            }
            await websocket.send(json.dumps(config_msg))
            logger.info(f"Sent Pixa TTS config: {config_msg}")
            
            return websocket
        except asyncio.TimeoutError:
            logger.error("Timeout connecting to Pixa TTS WebSocket")
            return None
        except Exception as e:
            logger.error(f"Failed to connect to Pixa TTS WebSocket: {e}")
            traceback.print_exc()
            return None

    async def monitor_connection(self):
        """Monitor WebSocket connection and reconnect if needed"""
        consecutive_failures = 0
        max_failures = 3

        while consecutive_failures < max_failures:
            if self.websocket_holder["websocket"] is None or self.websocket_holder["websocket"].state is websockets.protocol.State.CLOSED:
                logger.info("Re-establishing Pixa TTS WebSocket...")
                result = await self.establish_connection()
                if result is None:
                    consecutive_failures += 1
                    logger.warning(f"Pixa TTS connection failed ({consecutive_failures}/{max_failures})")
                    if consecutive_failures >= max_failures:
                        logger.error("Max failures reached, stopping reconnection")
                        break
                else:
                    self.websocket_holder["websocket"] = result
                    consecutive_failures = 0
            await asyncio.sleep(1)

    async def push(self, message):
        """Push message to synthesizer"""
        if self.stream:
            # WebSocket streaming mode
            meta_info, text, self.current_text = message.get("meta_info"), message.get("data"), message.get("data")
            self.synthesized_characters += len(text) if text is not None else 0
            end_of_llm_stream = "end_of_llm_stream" in meta_info and meta_info["end_of_llm_stream"]
            self.meta_info = copy.deepcopy(meta_info)
            meta_info["text"] = text

            try:
                if self.current_turn_start_time is None:
                    self.current_turn_start_time = time.perf_counter()
                    self.ws_send_time = None
                    self.current_turn_ttfb = None
                    logger.info(f"Pixa push: new turn, text_len={len(text) if text else 0}")
                self.current_turn_id = meta_info.get('turn_id') or meta_info.get('sequence_id')
            except Exception:
                pass

            self.sender_task = asyncio.create_task(self.sender_ws(text, meta_info.get('sequence_id'), end_of_llm_stream))
            self.text_queue.append(meta_info)
        else:
            # Non-streaming mode
            logger.info(f"Pixa push: Non-streaming mode")
            self.internal_queue.put_nowait(copy.deepcopy(message))

    async def cleanup(self):
        """Clean up resources"""
        self.conversation_ended = True
        logger.info("Cleaning up Pixa synthesizer")

        if self.sender_task:
            try:
                self.sender_task.cancel()
                await self.sender_task
            except asyncio.CancelledError:
                logger.info("Pixa sender task cancelled")
            except Exception as e:
                logger.error(f"Error cancelling sender: {e}")

        if self.websocket_holder["websocket"]:
            try:
                await self.websocket_holder["websocket"].close()
            except Exception as e:
                logger.error(f"Error closing WebSocket: {e}")

        self.websocket_holder["websocket"] = None
        logger.info("Pixa TTS cleanup complete")
