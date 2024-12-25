import asyncio
import copy
import uuid

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
                 stream=False, buffer_size=400, synthesizer_key=None, caching=True, **kwargs):
        super().__init__(stream)
        self.api_key = os.environ["CARTESIA_API_KEY"] if synthesizer_key is None else synthesizer_key
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
        self.sender_task = None

        self.ws_url = f"wss://api.cartesia.ai/tts/websocket?api_key={self.api_key}&cartesia_version=2024-06-10"
        self.api_url = "https://api.cartesia.ai/tts/bytes"
        self.turn_id = 0
        self.sequence_id = 0
        self.context_ids_to_ignore = set()
        self.conversation_ended = False

    def get_engine(self):
        return self.model

    async def handle_interruption(self):
        try:
            if self.context_id:
                self.context_ids_to_ignore.add(self.context_id)
                interrupt_message = {
                    "context_id": self.context_id,
                    "cancel": True
                }

                logger.info('handle_interruption: {}'.format(interrupt_message))
                await self.websocket_holder["websocket"].send(json.dumps(interrupt_message))
        except Exception as e:
            pass

    def form_payload(self, text):
        payload = {
            "context_id": self.context_id,
            "model_id": self.model,
            "transcript": text,
            "voice": {
                "mode": "id",
                "id": self.voice_id
            },
            "output_format": {
                "container": "raw",
                "encoding": "pcm_mulaw",
                "sample_rate": 8000
            }
        }

        if text:
            payload["continue"] = True

        return payload

    async def sender(self, text, end_of_llm_stream=False):
        try:
            if self.conversation_ended:
                return

            while self.websocket_holder["websocket"] is None or self.websocket_holder["websocket"].closed:
                logger.info("Waiting for webSocket connection to be established...")
                await asyncio.sleep(1)

            if text != "":
                logger.info(f"Sending text_chunk: {text}")
                try:
                    input_message = self.form_payload(text)
                    await self.websocket_holder["websocket"].send(json.dumps(input_message))
                except Exception as e:
                    logger.error(f"Error sending chunk: {e}")
                    return

            # If end_of_llm_stream is True, mark the last chunk and send an empty message
            if end_of_llm_stream:
                self.last_text_sent = True

            # Send the end-of-stream signal with an empty string as text
            try:
                input_message = self.form_payload("")
                await self.websocket_holder["websocket"].send(json.dumps(input_message))
                logger.info("Sent end-of-stream signal.")
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

                if self.websocket_holder["websocket"] is None or self.websocket_holder["websocket"].closed:
                    logger.info("WebSocket is not connected, skipping receive.")
                    await asyncio.sleep(5)
                    continue

                response = await self.websocket_holder["websocket"].recv()
                data = json.loads(response)

                # ignore all future generations of audio
                if data.get('context_id', None) in self.context_ids_to_ignore:
                    continue

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

    async def generate(self):
        try:
            async for message in self.receiver():
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
                    logger.info("received null byte and hence end of stream")
                    self.meta_info["end_of_synthesizer_stream"] = True
                    self.first_chunk_generated = False

                yield create_ws_data_packet(audio, self.meta_info)

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

    def update_context(self, meta_info):
        self.context_id = str(uuid.uuid4())
        self.turn_id = meta_info.get('turn_id', 0)
        self.sequence_id = meta_info.get('sequence_id', 0)

    async def push(self, message):
        logger.info(f"Pushed message to internal queue {message}")
        if self.stream:
            meta_info, text = message.get("meta_info"), message.get("data")
            self.synthesized_characters += len(text) if text is not None else 0
            end_of_llm_stream = "end_of_llm_stream" in meta_info and meta_info["end_of_llm_stream"]
            self.meta_info = copy.deepcopy(meta_info)
            meta_info["text"] = text
            if not self.context_id:
                self.update_context(meta_info)
            else:
                if self.turn_id != meta_info.get('turn_id', 0) or self.sequence_id != meta_info.get('sequence_id', 0):
                    self.update_context(meta_info)

            self.sender_task = asyncio.create_task(self.sender(text, end_of_llm_stream))
            self.text_queue.append(meta_info)
        else:
            self.internal_queue.put_nowait(message)

    async def cleanup(self):
        self.conversation_ended = True
        logger.info("cleaning cartesia synthesizer tasks")
        if self.sender_task:
            try:
                self.sender_task.cancel()
                await self.sender_task
            except asyncio.CancelledError:
                logger.info("Sender task was successfully cancelled during WebSocket cleanup.")

        if self.websocket_holder["websocket"]:
            await self.websocket_holder["websocket"].close()
        self.websocket_holder["websocket"] = None
        logger.info("WebSocket connection closed.")
