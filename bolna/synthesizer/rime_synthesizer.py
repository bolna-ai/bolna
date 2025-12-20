import copy
import aiohttp
import os
import uuid
import asyncio
import base64
import json
import traceback
import time
import websockets
from collections import deque
from dotenv import load_dotenv


from bolna.helpers.logger_config import configure_logger
from bolna.helpers.utils import convert_audio_to_wav, create_ws_data_packet
from bolna.memory.cache.inmemory_scalar_cache import InmemoryScalarCache
from .base_synthesizer import BaseSynthesizer

logger = configure_logger(__name__)
load_dotenv()


class RimeSynthesizer(BaseSynthesizer):
    def __init__(self, voice_id, voice, audio_format="wav", sampling_rate="8000", stream=False, buffer_size=400,
                 caching=True, model="arcana", synthesizer_key=None, **kwargs):
        super().__init__(kwargs.get("task_manager_instance", None), stream, buffer_size)
        self.format = 'mp3' if model == 'mistv2' and audio_format == 'wav' else audio_format
        self.voice = voice
        self.voice_id = voice_id
        self.buffer_size = buffer_size
        self.sample_rate = str(sampling_rate)
        self.model = model
        self.first_chunk_generated = False
        self.api_key = os.environ["RIME_API_KEY"] if synthesizer_key is None else synthesizer_key

        self.use_mulaw = True
        self.synthesized_characters = 0
        self.caching = caching
        self.ws_url = f"wss://users.rime.ai/ws2?speaker={self.voice_id}&modelId={self.model}&audioFormat=mulaw&samplingRate={self.sample_rate}"
        self.api_url = "https://users.rime.ai/v1/rime-tts"
        if self.model == 'arcana':
            self.stream = False

        self.websocket_holder = {"websocket": None}
        self.sender_task = None
        self.last_text_sent = False
        self.conversation_ended = False
        self.current_text = ""
        self.context_id = None
        self.text_queue = deque()
        self.meta_info = None
        self.audio_data = b''

        if caching:
            self.cache = InmemoryScalarCache()

    def get_synthesized_characters(self):
        return self.synthesized_characters

    def get_engine(self):
        return self.model

    async def handle_interruption(self):
        if self.stream:
            try:
                if self.context_id:
                    self.context_id = str(uuid.uuid4())
                    await self.websocket_holder["websocket"].send(json.dumps({"operation": "clear"}))
            except Exception as e:
                pass

    async def __generate_http(self, text):
        headers = {
            "Authorization": "Bearer {}".format(self.api_key),
            "Content-Type": "application/json",
            "Accept": f"audio/{self.format}"
        }

        payload = {
            "speaker": self.voice_id,
            "text": text,
            "modelId": self.model,
            "repetition_penalty": 1.5,
            "temperature": 0.5,
            "top_p": 0.5,
            "samplingRate": int(self.sample_rate),
            "max_tokens": 5000
        }

        try:
            async with aiohttp.ClientSession() as session:
                if payload is not None:
                    async with session.post(self.api_url, headers=headers, json=payload) as response:
                        if response.status == 200:
                            chunk = await response.read()
                            return chunk
                        else:
                            return b'\x00'
                else:
                    logger.info("Payload was null")
        except Exception as e:
            logger.error("something went wrong")

    def supports_websocket(self):
        return False

    async def open_connection(self):
        pass

    async def synthesize(self, text):
        # This is used for one off synthesis mainly for use cases like voice lab and IVR
        try:
            audio = await self.__generate_http(text)
            if self.format == "mp3":
                audio = convert_audio_to_wav(audio, source_format="mp3")
            return audio
        except Exception as e:
            logger.error(f"Could not synthesize {e}")

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
            while self.websocket_holder["websocket"] is None or self.websocket_holder["websocket"].state is websockets.protocol.State.CLOSED:
                logger.info("Waiting for elevenlabs ws connection to be established...")
                await asyncio.sleep(1)

            if text != "":
                for text_chunk in self.text_chunker(text):
                    if not self.should_synthesize_response(sequence_id):
                        logger.info(
                            f"Not synthesizing text as the sequence_id ({sequence_id}) of it is not in the list of sequence_ids present in the task manager (inner loop).")
                        await self.flush_synthesizer_stream()
                        return
                    try:
                        await self.websocket_holder["websocket"].send(json.dumps({"text": text_chunk, "contextId": self.context_id}))
                    except Exception as e:
                        logger.info(f"Error sending chunk: {e}")
                        return

            # If end_of_llm_stream is True, mark the last chunk and send an empty message
            if end_of_llm_stream:
                self.last_text_sent = True
                self.context_id = str(uuid.uuid4())

            # Send the end-of-stream signal with an empty string as text
            try:
                await self.websocket_holder["websocket"].send(json.dumps({"operation": "flush"}))
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
                        self.websocket_holder["websocket"].state is websockets.protocol.State.CLOSED):
                    logger.info("WebSocket is not connected, skipping receive.")
                    await asyncio.sleep(0.1)
                    continue

                response = await self.websocket_holder["websocket"].recv()
                data = json.loads(response)

                if data.get('type', '') == 'chunk':
                    self.audio_data += base64.b64decode(data["data"])

                if data['type'] == 'timestamps':
                    yield self.audio_data
                    self.audio_data = b''

                chunk_context_id = data.get('contextId', None)
                if chunk_context_id != self.context_id:
                    yield b'\x00'
                else:
                    logger.info("No audio data in the response")

            except websockets.exceptions.ConnectionClosed:
                break
            except Exception as e:
                logger.error(f"Error occurred in receiver - {e}")

    async def generate(self):
        try:
            if self.stream:
                async for message in self.receiver():
                    logger.info(f"Received message from server")

                    if len(self.text_queue) > 0:
                        self.meta_info = self.text_queue.popleft()
                        # Compute first-result latency on first audio chunk
                        try:
                            if self.meta_info and 'synthesizer_start_time' in self.meta_info and 'synthesizer_first_result_latency' not in self.meta_info:
                                self.meta_info['synthesizer_first_result_latency'] = time.perf_counter() - self.meta_info['synthesizer_start_time']
                                self.meta_info['synthesizer_latency'] = self.meta_info['synthesizer_first_result_latency']
                        except Exception:
                            pass
                    audio = ""

                    if self.use_mulaw:
                        self.meta_info['format'] = 'mulaw'
                        audio = message
                    else:
                        self.meta_info['format'] = "wav"
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
                        # Compute total stream duration for this synthesizer turn
                        try:
                            if self.meta_info and 'synthesizer_start_time' in self.meta_info:
                                self.meta_info['synthesizer_total_stream_duration'] = time.perf_counter() - self.meta_info['synthesizer_start_time']
                        except Exception:
                            pass

                    self.meta_info["mark_id"] = str(uuid.uuid4())
                    yield create_ws_data_packet(audio, self.meta_info)
            else:
                while True:
                    message = await self.internal_queue.get()
                    logger.info(f"Generating TTS response for message: {message}")
                    meta_info, text = message.get("meta_info"), message.get("data")
                    if not self.should_synthesize_response(meta_info.get('sequence_id')):
                        logger.info(
                            f"Not synthesizing text as the sequence_id ({meta_info.get('sequence_id')}) of it is not in the list of sequence_ids present in the task manager.")
                        return
                    if self.caching:
                        logger.info(f"Caching is on")
                        if self.cache.get(text):
                            logger.info(f"Cache hit and hence returning quickly {text}")
                            message = self.cache.get(text)
                        else:
                            logger.info(f"Not a cache hit {list(self.cache.data_dict)}")
                            self.synthesized_characters += len(text)
                            message = await self.__generate_http(text)
                            self.cache.set(text, message)
                    else:
                        logger.info(f"No caching present")
                        self.synthesized_characters += len(text)
                        message = await self.__generate_http(text)

                    if self.format == "mp3":
                        message = convert_audio_to_wav(message, source_format="mp3")
                    if not self.first_chunk_generated:
                        meta_info["is_first_chunk"] = True
                        self.first_chunk_generated = True
                    else:
                        meta_info["is_first_chunk"] = False
                    if "end_of_llm_stream" in meta_info and meta_info["end_of_llm_stream"]:
                        meta_info["end_of_synthesizer_stream"] = True
                        self.first_chunk_generated = False
                    meta_info['text'] = text
                    meta_info['format'] = 'wav'
                    meta_info["text_synthesized"] = f"{text} "
                    meta_info["mark_id"] = str(uuid.uuid4())
                    yield create_ws_data_packet(message, meta_info)

        except Exception as e:
            traceback.print_exc()
            logger.info(f"Error in eleven labs generate {e}")

    async def establish_connection(self):
        try:
            start_time = time.perf_counter()
            websocket_url = self.ws_url
            additional_headers = {
                'Authorization': 'Bearer {}'.format(self.api_key)
            }
            websocket = await websockets.connect(websocket_url, additional_headers=additional_headers)
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
                logger.info("Re-establishing rime connection...")
                self.websocket_holder["websocket"] = await self.establish_connection()
            await asyncio.sleep(1)

    async def get_sender_task(self):
        return self.sender_task

    async def push(self, message):
        if self.stream:
            meta_info, text, self.current_text = message.get("meta_info"), message.get("data"), message.get("data")
            self.synthesized_characters += len(text) if text is not None else 0
            end_of_llm_stream = "end_of_llm_stream" in meta_info and meta_info["end_of_llm_stream"]
            self.meta_info = copy.deepcopy(meta_info)
            meta_info["text"] = text
            # Stamp synthesizer turn start time
            try:
                meta_info['synthesizer_start_time'] = time.perf_counter()
            except Exception:
                pass
            if not self.context_id:
                self.context_id = str(uuid.uuid4())
            self.sender_task = asyncio.create_task(self.sender(text, meta_info.get("sequence_id"), end_of_llm_stream))
            self.text_queue.append(meta_info)
        else:
            self.internal_queue.put_nowait(message)

    async def cleanup(self):
        self.conversation_ended = True
        logger.info("cleaning rime synthesizer tasks")
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
