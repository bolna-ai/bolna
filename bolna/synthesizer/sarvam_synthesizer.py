import aiohttp
import asyncio
import os
import websockets
import copy
import time
import uuid
import traceback
import json
import base64
from collections import deque
from .base_synthesizer import BaseSynthesizer
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.utils import create_ws_data_packet, resample, wav_bytes_to_pcm

logger = configure_logger(__name__)


class SarvamSynthesizer(BaseSynthesizer):
    def __init__(self, voice_id, model, language, sampling_rate="8000", stream=False, buffer_size=400, speed=1.0, synthesizer_key=None, **kwargs):
        super().__init__(kwargs.get("task_manager_instance", None), stream)
        self.api_key = os.environ["SARVAM_API_KEY"] if synthesizer_key is None else synthesizer_key
        self.voice_id = voice_id
        self.model = model
        self.stream = stream
        self.buffer_size = buffer_size
        if self.buffer_size < 30 or self.buffer_size > 200:
            self.buffer_size = 200

        self.sampling_rate = int(sampling_rate)
        self.api_url = f"https://api.sarvam.ai/text-to-speech"
        self.ws_url = f"wss://api.sarvam.ai/text-to-speech/ws?model={model}"

        self.language = language
        self.loudness = 1.0
        self.pitch = 0.0
        self.pace = speed
        self.enable_preprocessing = True

        self.first_chunk_generated = False
        self.last_text_sent = False
        self.meta_info = None
        self.synthesized_characters = 0
        self.previous_request_ids = []
        self.websocket_holder = {"websocket": None}
        self.sender_task = None
        self.conversation_ended = False
        self.text_queue = deque()
        self.current_text = ""

    def get_engine(self):
        return self.model

    async def __send_payload(self, payload):
        headers = {
            'api-subscription-key': self.api_key,
            'Content-Type': 'application/json'
        }

        async with aiohttp.ClientSession() as session:
            if payload is not None:
                async with session.post(self.api_url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data and data.get('audios', []) and isinstance(data.get('audios', []), list):
                            return data.get('audios')[0]
                    else:
                        logger.error(f"Error: {response.status} - {await response.text()}")
            else:
                logger.info("Payload was null")

    async def synthesize(self, text):
        audio = await self.__generate_http(text)
        return audio

    def supports_websocket(self):
        return True

    async def __generate_http(self, text):
        payload = {
            "target_language_code": self.language,
            "text": text,
            "speaker": self.voice_id,
            "pitch": self.pitch,
            "loudness": self.loudness,
            "speech_sample_rate": self.sampling_rate,
            "enable_preprocessing": self.enable_preprocessing,
            "model": self.model
        }
        response = await self.__send_payload(payload)
        return response

    async def sender(self, text, sequence_id, end_of_llm_stream=False):
        try:
            if self.conversation_ended:
                return

            if not self.should_synthesize_response(sequence_id):
                logger.info(
                    f"Not synthesizing text as the sequence_id ({sequence_id}) of it is not in the list of sequence_ids present in the task manager.")
                return

            # Ensure the WebSocket connection is established
            while self.websocket_holder["websocket"] is None or self.websocket_holder["websocket"].state is websockets.protocol.State.CLOSED:
                logger.info("Waiting for sarvam ws connection to be established...")
                await asyncio.sleep(1)

            if text != "":
                try:
                    await self.websocket_holder["websocket"].send(json.dumps({"type": "text", "data": {"text": text}}))
                except Exception as e:
                    logger.error(f"Error sending chunk: {e}")
                    return

            # If end_of_llm_stream is True, mark the last chunk and send an empty message
            if end_of_llm_stream:
                self.last_text_sent = True

            try:
                await self.websocket_holder["websocket"].send(json.dumps({"type": "flush"}))
            except Exception as e:
                logger.info(f"Error sending end-of-stream signal: {e}")
        except asyncio.CancelledError:
            logger.info("Sender task was cancelled.")
        except Exception as e:
            logger.error(f"Unexpected error in sender: {e}")

    def form_payload(self, text):
        payload = {
            "target_language_code": self.language,
            "text": text,
            "speaker": self.voice_id,
            "pitch": self.pitch,
            "loudness": self.loudness,
            "speech_sample_rate": self.sampling_rate,
            "enable_preprocessing": self.enable_preprocessing,
            "model": self.model
        }

        return payload

    async def receiver(self):
        while True:
            try:
                if self.conversation_ended:
                    return

                if (self.websocket_holder["websocket"] is None or
                        self.websocket_holder["websocket"].state is websockets.protocol.State.CLOSED):
                    logger.info("WebSocket is not connected, skipping receive.")
                    await asyncio.sleep(5)
                    continue

                response = await self.websocket_holder["websocket"].recv()
                data = json.loads(response)

                if "type" in data and data["type"] == 'audio':
                    chunk = base64.b64decode(data["data"]["audio"])
                    yield chunk

                if self.last_text_sent:
                    yield b'\x00'

            except websockets.exceptions.ConnectionClosed:
                break
            except Exception as e:
                logger.error(f"Error occurred in receiver - {e}")

    async def establish_connection(self):
        try:
            start_time = time.perf_counter()
            websocket_url = self.ws_url
            additional_headers = {
                'api-subscription-key': self.api_key,
            }
            websocket = await websockets.connect(self.ws_url, additional_headers=additional_headers)
            bos_message = {
                "type": "config",
                "data": {
                    "target_language_code": self.language,
                    "speaker": self.voice_id,
                    "pitch": self.pitch,
                    "pace": self.pace,
                    "loudness": self.loudness,
                    "enable_preprocessing": self.enable_preprocessing,
                    "output_audio_codec": "wav",
                    "output_audio_bitrate": "32k",
                    "max_chunk_length": 250,
                    "min_buffer_size": self.buffer_size
                }
            }
            await websocket.send(json.dumps(bos_message))
            if not self.connection_time:
                self.connection_time = round((time.perf_counter() - start_time) * 1000)

            logger.info(f"Connected to {self.ws_url}")
            return websocket
        except Exception as e:
            logger.info(f"Failed to connect: {e}")
            return None

    async def monitor_connection(self):
        # Periodically check if the connection is still alive
        while True:
            if self.websocket_holder["websocket"] is None or self.websocket_holder["websocket"].state is websockets.protocol.State.CLOSED:
                logger.info("Re-establishing sarvam connection...")
                self.websocket_holder["websocket"] = await self.establish_connection()
            await asyncio.sleep(1)

    def get_synthesized_characters(self):
        return self.synthesized_characters

    async def get_sender_task(self):
        return self.sender_task

    async def generate(self):
        try:
            if self.stream:
                async for message in self.receiver():
                    logger.info(f"Received message from server")

                    if len(self.text_queue) > 0:
                        self.meta_info = self.text_queue.popleft()

                    self.meta_info['format'] = 'wav'
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
                        logger.info("received null byte and hence end of stream")
                        self.meta_info["end_of_synthesizer_stream"] = True
                        self.first_chunk_generated = False
                    else:
                        resampled_audio = resample(audio, int(self.sampling_rate), format="wav")
                        audio = wav_bytes_to_pcm(resampled_audio)

                    self.meta_info["mark_id"] = str(uuid.uuid4())
                    yield create_ws_data_packet(audio, self.meta_info)

        except Exception as e:
            traceback.print_exc()
            logger.info(f"Error in sarvam generate {e}")

    async def push(self, message):
        if self.stream:
            meta_info, text, self.current_text = message.get("meta_info"), message.get("data"), message.get("data")
            self.synthesized_characters += len(text) if text is not None else 0
            end_of_llm_stream = "end_of_llm_stream" in meta_info and meta_info["end_of_llm_stream"]
            self.meta_info = copy.deepcopy(meta_info)
            meta_info["text"] = text
            self.sender_task = asyncio.create_task(self.sender(text, meta_info.get("sequence_id"), end_of_llm_stream))
            self.text_queue.append(meta_info)
        else:
            self.internal_queue.put_nowait(message)

    async def cleanup(self):
        self.conversation_ended = True
        logger.info("cleaning sarvam synthesizer tasks")
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
