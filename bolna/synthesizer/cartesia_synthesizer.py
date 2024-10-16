import asyncio
import copy


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


class CartesiaSynthesizer(BaseSynthesizer):
    def __init__(self, voice_id, voice, model="sonic-english", audio_format="mp3", sampling_rate="16000",
                 stream=False, buffer_size=400, synthesier_key=None, caching=True, **kwargs):
        super().__init__(stream)
        self.api_key = os.environ["CARTESIA_API_KEY"] if synthesier_key is None else synthesier_key
        self.version = '2024-06-10'
        self.voice_id = voice_id
        self.model = model
        self.stream = True
        self.websocket_connection = None
        self.connection_open = False
        self.sampling_rate = sampling_rate
        self.use_mulaw = True
        self.first_chunk_generated = False
        self.last_text_sent = False
        self.text_queue = deque()
        self.meta_info = None
        self.caching = caching
        if self.caching:
            self.cache = InmemoryScalarCache()
        self.synthesized_characters = 0
        self.previous_request_ids = []
        self.websocket_holder = {"websocket": None}
        self.context_id = None

        self.ws_url = f"wss://api.cartesia.ai/tts/websocket?api_key={self.api_key}&cartesia_version=2024-06-10"
        self.api_url = "https://api.cartesia.ai/tts/bytes"

    def get_engine(self):
        return self.model

    async def sender(self, text, end_of_llm_stream=False):
        while self.websocket_holder["websocket"] is None or self.websocket_holder["websocket"].closed:
            logger.info("Waiting for webSocket connection to be established...")
            await asyncio.sleep(1)

        if text != "":
            logger.info(f"Sending text_chunk: {text}")
            try:
                input_message = {
                    "context_id": self.context_id,
                    "model_id": self.model,
                    "transcript": text,
                    "voice": {
                        "mode": "id",
                        "id": self.voice_id
                    },
                    "continue": True,
                    "output_format": {
                        "container": "raw",
                        "encoding": "pcm_mulaw",
                        "sample_rate": 8000
                    },
                    "add_timestamps": True
                }

                await self.websocket_holder["websocket"].send(json.dumps(input_message))
            except Exception as e:
                logger.error(f"Error sending chunk: {e}")
                return

        # If end_of_llm_stream is True, mark the last chunk and send an empty message
        if end_of_llm_stream:
            self.last_text_sent = True

        # Send the end-of-stream signal with an empty string as text
        try:
            input_message = {
                "context_id": self.context_id,
                "model_id": self.model,
                "transcript": "",
                "voice": {
                    "mode": "id",
                    "id": self.voice_id
                },
                "continue": False,
                "output_format": {
                    "container": "raw",
                    "encoding": "pcm_mulaw",
                    "sample_rate": 8000
                }
            }

            await self.websocket_holder["websocket"].send(json.dumps(input_message))
            logger.info("Sent end-of-stream signal.")
        except Exception as e:
            logger.error(f"Error sending end-of-stream signal: {e}")

    async def receiver(self):
        while True:
            try:
                if self.websocket_holder["websocket"] is None or self.websocket_holder["websocket"].closed:
                    logger.info("WebSocket is not connected, skipping receive.")
                    await asyncio.sleep(5)
                    continue

                response = await self.websocket_holder["websocket"].recv()
                data = json.loads(response)

                if "data" in data and data["data"]:
                    chunk = base64.b64decode(data["data"])
                    yield chunk

                elif "done" in data and data["done"]:
                    yield b'\x00'
                else:
                    logger.info("No audio data in the response")
            except websockets.exceptions.ConnectionClosed:
                break

    async def __send_payload(self, payload):
        headers = {
            'X-API-Key': self.api_key,
            'Cartesia-Version': self.version
        }

        async with aiohttp.ClientSession() as session:
            if payload is not None:
                async with session.post(self.api_url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        data = await response.read()
                        return data
                    else:
                        logger.error(f"Error: {response.status} - {await response.text()}")
            else:
                logger.info("Payload was null")

    async def synthesize(self, text):
        audio = await self.__generate_http(text)
        return audio

    async def __generate_http(self, text,):
        payload = None
        logger.info(f"text {text}")
        payload = {
            "model_id": self.model,
            "transcript": text,
            "voice": {
                "mode": "id",
                "id": self.voice_id
            },
            "output_format": {
                "container": "mp3",
                "encoding": "mp3",
                "sample_rate": 44100
            }
        }
        response = await self.__send_payload(payload)
        return response

    def get_synthesized_characters(self):
        return self.synthesized_characters

    # Currently we are only supporting wav output but soon we will incorporate conver
    async def generate(self):
        try:
            if self.stream:
                async for message in self.receiver():
                    logger.info(f"Received message from server")

                    if len(self.text_queue) > 0:
                        self.meta_info = self.text_queue.popleft()
                    audio = ""

                    if self.use_mulaw:
                        self.meta_info['format'] = 'mulaw'
                        audio = message
                    else:
                        self.meta_info['format'] = "wav"
                        audio = resample(convert_audio_to_wav(message, source_format="mp3"), int(self.sampling_rate),
                                         format="wav")

                    yield create_ws_data_packet(audio, self.meta_info)
                    if not self.first_chunk_generated:
                        self.meta_info["is_first_chunk"] = True
                        self.first_chunk_generated = True

                    if self.last_text_sent:
                        # Reset the last_text_sent and first_chunk converted to reset synth latency
                        self.first_chunk_generated = False
                        self.last_text_sent = True

                    if message == b'\x00':
                        logger.info("received null byte and hence end of stream")
                        self.meta_info["end_of_synthesizer_stream"] = True
                        #yield create_ws_data_packet(resample(message, int(self.sampling_rate)), self.meta_info)
                        self.first_chunk_generated = False

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
                    if not self.first_chunk_generated:
                        meta_info["is_first_chunk"] = True
                        self.first_chunk_generated = True

                    if "end_of_llm_stream" in meta_info and meta_info["end_of_llm_stream"]:
                        meta_info["end_of_synthesizer_stream"] = True
                        self.first_chunk_generated = False

                    if self.use_mulaw:
                        meta_info['format'] = "mulaw"
                    else:
                        meta_info['format'] = "wav"
                        wav_bytes = convert_audio_to_wav(audio, source_format="mp3")
                        logger.info(f"self.sampling_rate {self.sampling_rate}")
                        audio = resample(wav_bytes, int(self.sampling_rate), format="wav")
                    yield create_ws_data_packet(audio, meta_info)

        except Exception as e:
            traceback.print_exc()
            logger.error(f"Error in cartesia generate {e}")

    async def establish_connection(self):
        try:
            websocket = await websockets.connect(self.ws_url)
            logger.info(f"Connected to {self.ws_url}")
            return websocket
        except Exception as e:
            logger.info(f"Failed to connect: {e}")
            return None

    async def monitor_connection(self):
        # Periodically check if the connection is still alive
        while True:
            if self.websocket_holder["websocket"] is None or self.websocket_holder["websocket"].closed:
                logger.info("Re-establishing connection...")
                self.websocket_holder["websocket"] = await self.establish_connection()
            await asyncio.sleep(50)

    async def push(self, message):
        logger.info(f"Pushed message to internal queue {message}")
        if self.stream:
            meta_info, text = message.get("meta_info"), message.get("data")
            self.synthesized_characters += len(text) if text is not None else 0
            end_of_llm_stream = "end_of_llm_stream" in meta_info and meta_info["end_of_llm_stream"]
            self.meta_info = copy.deepcopy(meta_info)
            meta_info["text"] = text
            self.context_id = meta_info["request_id"]
            self.sender_task = asyncio.create_task(self.sender(text, end_of_llm_stream))
            self.text_queue.append(meta_info)
        else:
            self.internal_queue.put_nowait(message)
