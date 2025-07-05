import asyncio
import copy
import uuid
import time
import websockets
import base64
import json
import aiohttp
import os
import traceback
from collections import deque

from bolna.memory.cache.inmemory_scalar_cache import InmemoryScalarCache
from .base_synthesizer import BaseSynthesizer
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.utils import convert_audio_to_wav, create_ws_data_packet, resample

logger = configure_logger(__name__)


class ElevenlabsSynthesizer(BaseSynthesizer):
    def __init__(self, voice, voice_id, model="eleven_turbo_v2_5", audio_format="mp3", sampling_rate="16000",
                 stream=False, buffer_size=400, temperature=0.5, similarity_boost=0.8, synthesizer_key=None,
                 caching=True, **kwargs):
        super().__init__(kwargs.get("task_manager_instance", None), stream)
        self.api_key = os.environ["ELEVENLABS_API_KEY"] if synthesizer_key is None else synthesizer_key
        self.voice = voice_id
        self.model = model
        self.stream = True  # Issue with elevenlabs streaming that we need to always send the text quickly
        self.sampling_rate = sampling_rate
        self.audio_format = "mp3"
        self.use_mulaw = kwargs.get("use_mulaw", True)
        self.ws_url = f"wss://api.elevenlabs.io/v1/text-to-speech/{self.voice}/multi-stream-input?model_id={self.model}&output_format={'ulaw_8000' if self.use_mulaw else 'mp3_44100_128'}&inactivity_timeout=170&sync_alignment=true"
        self.api_url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice}?optimize_streaming_latency=2&output_format="
        self.temperature = temperature
        self.similarity_boost = similarity_boost
        self.caching = caching
        if self.caching:
            self.cache = InmemoryScalarCache()
        self.previous_request_ids = []
        self.context_id = None

    # Ensuring we only do wav output for now
    def get_format(self, format, sampling_rate):
        # Eleven labs only allow mp3_44100_64, mp3_44100_96, mp3_44100_128, mp3_44100_192, pcm_16000, pcm_22050,
        # pcm_24000, ulaw_8000
        if self.use_mulaw:
            return "ulaw_8000"
        return f"mp3_44100_128"

    def get_engine(self):
        return self.model

    async def handle_interruption(self):
        try:
            if self.context_id:
                interrupt_message = {
                    "context_id": self.context_id,
                    "close_context": True
                }

                self.context_id = str(uuid.uuid4())
                websocket = self.websocket_holder.get("websocket")
                if websocket and hasattr(websocket, 'send'):
                    await websocket.send(json.dumps(interrupt_message))
        except Exception as e:
            pass

    async def sender(self, text, sequence_id, end_of_llm_stream=False):
        try:
            if self.conversation_ended:
                return

            if not self.should_synthesize_response(sequence_id):
                logger.info(
                    f"Not synthesizing text as the sequence_id ({sequence_id}) of it is not in the list of sequence_ids present in the task manager.")
                await self.flush_synthesizer_stream()
                return

            # Ensure the WebSocket connection is established
            while (self.websocket_holder["websocket"] is None or 
                   (hasattr(self.websocket_holder["websocket"], 'state') and 
                    hasattr(self.websocket_holder["websocket"].state, 'name') and
                    self.websocket_holder["websocket"].state.name == 'CLOSED')):
                logger.info("Waiting for elevenlabs ws connection to be established...")
                await asyncio.sleep(1)

            if text != "":
                logger.info(f"Sending text: {text}")
                for text_chunk in self.text_chunker(text):
                    if not self.should_synthesize_response(sequence_id):
                        logger.info(
                            f"Not synthesizing text as the sequence_id ({sequence_id}) of it is not in the list of sequence_ids present in the task manager (inner loop).")
                        await self.flush_synthesizer_stream()
                        return
                    try:
                        websocket = self.websocket_holder.get("websocket")
                        if websocket and hasattr(websocket, 'send'):
                            await websocket.send(json.dumps({"text": text_chunk}))
                    except Exception as e:
                        logger.info(f"Error sending chunk: {e}")
                        return

            # If end_of_llm_stream is True, mark the last chunk and send an empty message
            if end_of_llm_stream:
                self.last_text_sent = True
                self.context_id = str(uuid.uuid4())

            # Send the end-of-stream signal with an empty string as text
            try:
                websocket = self.websocket_holder.get("websocket")
                if websocket and hasattr(websocket, 'send'):
                    await websocket.send(json.dumps({"text": "", "flush": True}))
            except Exception as e:
                logger.info(f"Error sending end-of-stream signal: {e}")

        except asyncio.CancelledError:
            logger.info("Sender task was cancelled.")
        except Exception as e:
            logger.error(f"Unexpected error in sender: {e}")

    async def receiver(self):
        while True:
            try:
                if self.conversation_ended:
                    return

                if (self.websocket_holder["websocket"] is None or
                        (hasattr(self.websocket_holder["websocket"], 'state') and 
                         hasattr(self.websocket_holder["websocket"].state, 'name') and
                         self.websocket_holder["websocket"].state.name == 'CLOSED')):
                    logger.info("WebSocket is not connected, skipping receive.")
                    await asyncio.sleep(5)
                    continue

                websocket = self.websocket_holder.get("websocket")
                if websocket and hasattr(websocket, 'recv'):
                    response = await websocket.recv()
                else:
                    continue
                data = json.loads(response)
                logger.info("response for isFinal: {}".format(data.get('isFinal', False)))
                # logger.info(f"Response from elevenlabs - {data}")

                if "audio" in data and data["audio"]:
                    chunk = base64.b64decode(data["audio"])
                    try:
                        text_spoken = ''.join(data.get('alignment', {}).get('chars', []))
                    except Exception as e:
                        text_spoken = ""
                    yield chunk, text_spoken

                if "isFinal" in data and data["isFinal"]:
                    yield b'\x00', ""

                elif self.last_text_sent:
                    try:
                        response_chars = data.get('alignment', {}).get('chars', [])
                        response_text = ''.join(response_chars)
                        last_four_words_text = ' '.join(response_text.split(" ")[-4:]).strip()
                        last_four_words_text = last_four_words_text.replace('"', "").strip()
                        logger.info(f'Last four char - {last_four_words_text} | current text - {self.current_text.strip()}')
                        if self.current_text.replace('"', "").strip().endswith(last_four_words_text):
                            logger.info('send end_of_synthesizer_stream')
                            yield b'\x00', ""
                        elif self.current_text.replace('"', "").replace(' ', '').strip().endswith(last_four_words_text.replace(' ', '')):
                            logger.info('send end_of_synthesizer_stream on fallback')
                            yield b'\x00', ""
                    except Exception as e:
                        logger.error(f"Error occurred while getting chars from response - {e}")
                        yield b'\x00', ""

                else:
                    logger.info("No audio data in the response")

            except websockets.exceptions.ConnectionClosed:
                break
            except Exception as e:
                logger.error(f"Error occurred in receiver - {e}")

    async def __send_payload(self, payload, format=None):
        headers = {
            'xi-api-key': self.api_key
        }
        url = f"{self.api_url}{self.get_format(self.audio_format, self.sampling_rate)}" if format is None else f"{self.api_url}{format}"
        return await self.send_http_request(url, headers, payload)

    async def synthesize(self, text):
        audio = await self.__generate_http(text, format="mp3_44100_128")
        return audio

    async def __generate_http(self, text, format=None):
        payload = None
        logger.info(f"text {text}")
        payload = {
            "text": text,
            "model_id": self.model,
            "voice_settings": {
                "stability": self.temperature,
                "similarity_boost": self.similarity_boost,
                "optimize_streaming_latency": 3
            }
        }
        response = await self.__send_payload(payload, format=format)
        return response

    def get_synthesized_characters(self):
        return self.synthesized_characters

    # Currently we are only supporting wav output but soon we will incorporate conver
    async def generate(self):
        try:
            if self.stream:
                async for message, text_synthesized in self.receiver():
                    logger.info(f"Received message from server")

                    if len(self.text_queue) > 0:
                        self.meta_info = self.text_queue.popleft()
                    audio = ""

                    if self.use_mulaw:
                        self.meta_info['format'] = 'mulaw'
                        audio = message
                    else:
                        self.meta_info['format'] = "wav"
                        audio = message
                        if message != b'\x00':
                            audio = resample(convert_audio_to_wav(message, source_format="mp3"), int(self.sampling_rate),
                                             format="wav")

                    self.set_first_chunk_metadata(self.meta_info)

                    if self.last_text_sent:
                        # Reset the last_text_sent and first_chunk converted to reset synth latency
                        self.first_chunk_generated = False
                        self.last_text_sent = True

                    if message == b'\x00':
                        logger.info("received null byte and hence end of stream")
                        self.meta_info["end_of_synthesizer_stream"] = True
                        self.first_chunk_generated = False

                    yield self.create_audio_packet(audio, self.meta_info, text_synthesized)
            else:
                while True:
                    message = await self.internal_queue.get()
                    logger.info(f"Generating TTS response for message: {message}, using mulaw {self.use_mulaw}")
                    meta_info, text = message.get("meta_info"), message.get("data")
                    audio = None
                    if self.caching:
                        if self.cache.get(text):
                            logger.info(f"Cache hit and hence returning quickly {text}")
                            audio = self.cache.get(text)
                            meta_info['is_cached'] = True
                        else:
                            c = len(text)
                            self.synthesized_characters += c
                            logger.info(
                                f"Not a cache hit {list(self.cache.data_dict)} and hence increasing characters by {c}")
                            meta_info['is_cached'] = False
                            audio = await self.__generate_http(text)
                            self.cache.set(text, audio)
                    else:
                        meta_info['is_cached'] = False
                        audio = await self.__generate_http(text)

                    meta_info['text'] = text
                    self.set_first_chunk_metadata(meta_info)
                    self.set_end_of_stream_metadata(meta_info)

                    if self.use_mulaw:
                        meta_info['format'] = "mulaw"
                    else:
                        meta_info['format'] = "wav"
                        wav_bytes = convert_audio_to_wav(audio, source_format="mp3")
                        logger.info(f"self.sampling_rate {self.sampling_rate}")
                        audio = resample(wav_bytes, int(self.sampling_rate), format="wav")
                    yield self.create_audio_packet(audio, meta_info)

        except Exception as e:
            traceback.print_exc()
            logger.info(f"Error in eleven labs generate {e}")

    def supports_websocket(self):
        return True

    async def establish_connection(self):
        try:
            start_time = time.perf_counter()
            websocket = await websockets.connect(self.ws_url)
            bos_message = {
                "text": " ",
                "voice_settings": {
                    "stability": self.temperature,
                    "similarity_boost": self.similarity_boost
                },
                "xi_api_key": self.api_key
            }
            await websocket.send(json.dumps(bos_message))
            if not self.connection_time:
                self.connection_time = round((time.perf_counter() - start_time) * 1000)

            logger.info(f"Connected to {self.ws_url}")
            return websocket
        except Exception as e:
            logger.info(f"Failed to connect: {e}")
            return None


    async def get_sender_task(self):
        return self.sender_task

    def update_streaming_context(self, meta_info):
        """ElevenLabs-specific context management"""
        if not self.context_id:
            self.context_id = str(uuid.uuid4())
        logger.info(f"context_id: {self.context_id}")
