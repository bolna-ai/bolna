import asyncio
import json
import os
import time
import uuid
from websockets.exceptions import InvalidHandshake
import base64

import aiohttp
import websockets

from .stream_synthesizer import StreamSynthesizer
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.utils import convert_audio_to_wav, create_ws_data_packet, resample
from bolna.memory.cache.inmemory_scalar_cache import InmemoryScalarCache

logger = configure_logger(__name__)


class ElevenlabsSynthesizer(StreamSynthesizer):
    def __init__(
        self,
        voice,
        voice_id,
        model="eleven_turbo_v2_5",
        audio_format="mp3",
        sampling_rate="16000",
        stream=False,
        buffer_size=400,
        temperature=0.5,
        similarity_boost=0.75,
        speed=1.0,
        style=0,
        synthesizer_key=None,
        caching=True,
        **kwargs,
    ):
        super().__init__(
            stream=True,  # ElevenLabs always streams
            provider_name="elevenlabs",
            buffer_size=buffer_size,
            **kwargs,
        )
        self.api_key = os.environ["ELEVENLABS_API_KEY"] if synthesizer_key is None else synthesizer_key
        self.voice = voice_id
        self.model = model
        self.stream = True
        self.sampling_rate = sampling_rate
        self.speed = speed
        self.style = style
        self.audio_format = "mp3"
        self.use_mulaw = kwargs.get("use_mulaw", True)
        self.temperature = temperature
        self.similarity_boost = similarity_boost
        self.caching = caching
        if self.caching:
            self.cache = InmemoryScalarCache()

        self.elevenlabs_host = os.getenv("ELEVENLABS_API_HOST", "api.elevenlabs.io")
        self.wire_format = "ulaw_8000" if self.use_mulaw else "mp3_44100_128"
        self.ws_url = (
            f"wss://{self.elevenlabs_host}/v1/text-to-speech/{self.voice}/multi-stream-input"
            f"?model_id={self.model}&output_format={self.wire_format}"
            f"&inactivity_timeout=170&sync_alignment=true&optimize_streaming_latency=4"
        )
        self.api_url = f"https://{self.elevenlabs_host}/v1/text-to-speech/{self.voice}/stream?optimize_streaming_latency=2&output_format="

        self.context_id = None
        self.ws_send_time = None
        self.ws_trace_id = None
        self.current_turn_ttfb = None

    # ------------------------------------------------------------------
    # StreamSynthesizer hooks
    # ------------------------------------------------------------------

    def _get_audio_format(self):
        return "mulaw" if self.wire_format == "ulaw_8000" else "wav"

    def _process_audio_chunk(self, chunk):
        # ulaw_8000 arrives ready to use; mp3 needs conversion + resampling
        if self.wire_format == "ulaw_8000":
            return chunk
        return resample(convert_audio_to_wav(chunk, source_format="mp3"), int(self.sampling_rate), format="wav")

    def _unpack_receiver_message(self, item):
        """ElevenLabs receiver yields (audio, text_synthesized) tuples."""
        audio, text_synthesized = item
        return audio, {"text_synthesized": text_synthesized}

    def _on_push(self, meta_info, text):
        if not self.context_id:
            self.context_id = str(uuid.uuid4())

    # ------------------------------------------------------------------
    # Format helper
    # ------------------------------------------------------------------

    def _get_output_format(self):
        return self.wire_format

    # ------------------------------------------------------------------
    # Interruption
    # ------------------------------------------------------------------

    async def handle_interruption(self):
        try:
            if self.context_id:
                interrupt_message = {"context_id": self.context_id, "close_context": True}
                self.context_id = str(uuid.uuid4())
                await self.websocket.send(json.dumps(interrupt_message))
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
                await self.flush_synthesizer_stream()
                return

            await self._wait_for_ws()

            if text != "":
                for text_chunk in self.text_chunker(text):
                    if not self.should_synthesize_response(sequence_id):
                        logger.info(f"Not synthesizing (inner): sequence_id {sequence_id} not current")
                        await self.flush_synthesizer_stream()
                        return
                    try:
                        if self.ws_send_time is None:
                            self.ws_send_time = time.perf_counter()
                            logger.info(f"WS send trace_id={self.ws_trace_id} first_text_sent")
                        await self.websocket.send(json.dumps({"text": text_chunk}))
                    except Exception as e:
                        logger.info(f"Error sending chunk: {e}")
                        self.connection_error = str(e)
                        return

            if end_of_llm_stream:
                self.last_text_sent = True
                self.context_id = str(uuid.uuid4())

            try:
                await self.websocket.send(json.dumps({"text": "", "flush": True}))
            except Exception as e:
                logger.info(f"Error sending end-of-stream signal: {e}")
                self.connection_error = str(e)

        except asyncio.CancelledError:
            logger.info("Sender task was cancelled.")
        except Exception as e:
            logger.error(f"Unexpected error in sender: {e}")

    async def receiver(self):
        """Yields (audio_chunk, text_spoken) tuples, or (b'\\x00', '') for end-of-stream."""
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
                        logger.error("ElevenLabs receiver: WebSocket never connected after 30s, giving up.")
                        self.connection_error = self.connection_error or "WebSocket never connected"
                        return
                    logger.info("WebSocket is not connected, skipping receive.")
                    await asyncio.sleep(0.10)
                    continue
                else:
                    not_connected_since = None

                recv_start = time.perf_counter()
                response = await self.websocket.recv()
                recv_duration = (time.perf_counter() - recv_start) * 1000
                data = json.loads(response)

                if "audio" in data and data["audio"] and self.ws_send_time is not None:
                    audio_chunk_count += 1
                    if audio_chunk_count == 1:
                        time_since_send = (time.perf_counter() - self.ws_send_time) * 1000
                        logger.info(
                            f"WS recv FIRST trace_id={self.ws_trace_id} recv_wait={recv_duration:.0f}ms time_since_send={time_since_send:.0f}ms"
                        )
                    elif recv_duration > 200:
                        gap = (recv_start - last_recv_time) * 1000 if last_recv_time else 0
                        logger.info(
                            f"WS recv SLOW chunk={audio_chunk_count} trace_id={self.ws_trace_id} recv_wait={recv_duration:.0f}ms gap={gap:.0f}ms"
                        )
                    last_recv_time = time.perf_counter()

                logger.info("response for isFinal: {}".format(data.get("isFinal", False)))

                if "audio" in data and data["audio"]:
                    chunk = base64.b64decode(data["audio"])
                    try:
                        text_spoken = "".join(data.get("alignment", {}).get("chars", []))
                    except Exception:
                        text_spoken = ""
                    yield chunk, text_spoken

                if "isFinal" in data and data["isFinal"]:
                    logger.info(f"WS recv isFinal trace_id={self.ws_trace_id}")
                    audio_chunk_count = 0
                    last_recv_time = None
                    yield b"\x00", ""

                elif self.last_text_sent:
                    try:
                        response_chars = data.get("alignment", {}).get("chars", [])
                        response_text = "".join(response_chars)
                        last_four = " ".join(response_text.split(" ")[-4:]).replace('"', "").strip()
                        current_norm = self.normalize_text(self.current_text.strip()).replace('"', "").strip()
                        logger.info(f"Last four char - {last_four} | current text - {current_norm}")

                        if current_norm.endswith(last_four):
                            logger.info("send end_of_synthesizer_stream")
                            yield b"\x00", ""
                        elif (
                            current_norm.replace('"', "").replace(" ", "").strip().endswith(last_four.replace(" ", ""))
                        ):
                            logger.info("send end_of_synthesizer_stream on fallback")
                            yield b"\x00", ""
                    except Exception as e:
                        logger.error(f"Error getting chars from response - {e}")
                        yield b"\x00", ""
                else:
                    logger.info("No audio data in the response")

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
            websocket = await asyncio.wait_for(websockets.connect(self.ws_url), timeout=10.0)
            if hasattr(websocket, "response") and hasattr(websocket.response, "headers"):
                self.ws_trace_id = websocket.response.headers.get("x-trace-id")
                logger.info(f"Elevenlabs WebSocket connected trace_id={self.ws_trace_id}")
            bos_message = {
                "text": " ",
                "voice_settings": {
                    "stability": self.temperature,
                    "similarity_boost": self.similarity_boost,
                    "speed": self.speed,
                    "style": self.style,
                },
                "generation_config": {
                    "chunk_length_schedule": [50, 80, 120, 150],
                },
                "xi_api_key": self.api_key,
            }
            await websocket.send(json.dumps(bos_message))
            if not self.connection_time:
                self.connection_time = round((time.perf_counter() - start_time) * 1000)
            logger.info(f"Connected to {self.ws_url}")
            return websocket
        except asyncio.TimeoutError:
            logger.error("Timeout while connecting to ElevenLabs websocket")
            return None
        except InvalidHandshake as e:
            error_msg = str(e)
            if "401" in error_msg or "403" in error_msg:
                logger.error(f"ElevenLabs authentication failed: Invalid or expired API key - {e}")
            else:
                logger.error(f"ElevenLabs handshake failed: {e}")
            self.connection_error = str(e)
            return None
        except Exception as e:
            logger.error(f"Failed to connect to ElevenLabs: {e}")
            return None

    # ------------------------------------------------------------------
    # HTTP fallback
    # ------------------------------------------------------------------

    async def synthesize(self, text):
        return await self._generate_http(text, format="mp3_44100_128")

    async def _generate_http(self, text, format=None):
        payload = {
            "text": text,
            "model_id": self.model,
            "voice_settings": {
                "stability": self.temperature,
                "similarity_boost": self.similarity_boost,
                "optimize_streaming_latency": 3,
                "speed": self.speed,
                "style": self.style,
            },
        }
        headers = {"xi-api-key": self.api_key}
        fmt = format or self._get_output_format()
        url = f"{self.api_url}{fmt}"
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    return await response.read()
                else:
                    logger.error(f"Error: {response.status} - {await response.text()}")
                    return None
