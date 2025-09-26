import copy
import time
import aiohttp
import os
import uuid
from dotenv import load_dotenv
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.utils import convert_audio_to_wav, create_ws_data_packet
from bolna.memory.cache.inmemory_scalar_cache import InmemoryScalarCache
from .base_synthesizer import BaseSynthesizer

logger = configure_logger(__name__)
load_dotenv()
DEEPGRAM_HOST = os.getenv('DEEPGRAM_HOST', 'api.deepgram.com')
DEEPGRAM_TTS_URL = "https://{}/v1/speak".format(DEEPGRAM_HOST)


class DeepgramSynthesizer(BaseSynthesizer):
    def __init__(self, voice_id, voice, audio_format="pcm", sampling_rate="8000", stream=False, buffer_size=400, caching=True,
                 model="aura-zeus-en", **kwargs):
        super().__init__(kwargs.get("task_manager_instance", None), stream, buffer_size)
        self.format = "mulaw" if audio_format in ["pcm", 'wav'] else audio_format
        self.voice = voice
        self.voice_id = voice_id
        self.sample_rate = str(sampling_rate)
        self.model = model
        self.first_chunk_generated = False
        self.api_key = kwargs.get("transcriber_key", os.getenv('DEEPGRAM_AUTH_TOKEN'))

        if len(self.model.split('-')) == 2:
            self.model = f"{self.model}-{self.voice_id}"
        
        self.synthesized_characters = 0
        self.caching = caching
        if caching:
            self.cache = InmemoryScalarCache()

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

    async def generate(self):
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
            # Compute first-result latency for HTTP (first and only audio message)
            try:
                if 'synthesizer_start_time' in meta_info and 'synthesizer_first_result_latency' not in meta_info:
                    meta_info['synthesizer_first_result_latency'] = time.perf_counter() - meta_info['synthesizer_start_time']
                    meta_info['synthesizer_latency'] = meta_info['synthesizer_first_result_latency']
            except Exception:
                pass
            meta_info['format'] = 'mulaw'
            meta_info["text_synthesized"] = f"{text} "
            meta_info["mark_id"] = str(uuid.uuid4())
            # Compute total stream duration (HTTP single-shot)
            try:
                if 'synthesizer_start_time' in meta_info:
                    meta_info['synthesizer_total_stream_duration'] = time.perf_counter() - meta_info['synthesizer_start_time']
            except Exception:
                pass
            yield create_ws_data_packet(message, meta_info)

    async def push(self, message):
        logger.info(f"Pushed message to internal queue {message}")
        self.internal_queue.put_nowait(copy.deepcopy(message))
