import asyncio
import copy
import io
import os
import time
import aiohttp
from dotenv import load_dotenv
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.utils import create_ws_data_packet
from .base_synthesizer import BaseSynthesizer

logger = configure_logger(__name__)
load_dotenv()

CAMB_TTS_URL = "https://client.camb.ai/apis/tts-stream"


class CambSynthesizer(BaseSynthesizer):
    def __init__(
        self,
        voice_id,
        voice,
        model="mars-flash",
        language="en-us",
        audio_format="pcm",
        sampling_rate=24000,
        stream=False,
        buffer_size=400,
        **kwargs
    ):
        super().__init__(kwargs.get("task_manager_instance", None), stream, buffer_size)
        self.voice_id = int(voice_id)
        self.voice = voice
        self.model = model
        self.language = language
        self.sample_rate = int(sampling_rate) if isinstance(sampling_rate, str) else sampling_rate
        self.first_chunk_generated = False
        self.synthesized_characters = 0

        api_key = kwargs.get("synthesizer_key", os.getenv("CAMB_API_KEY"))
        if not api_key:
            raise ValueError(
                "Camb.ai API key not found. Set CAMB_API_KEY environment variable "
                "or pass synthesizer_key parameter."
            )
        self.api_key = api_key
        self.user_instructions = kwargs.get("user_instructions")
        self.session = None

    async def synthesize(self, text):
        """One-off synthesis for use cases like voice lab and IVR."""
        audio = await self.__generate_http(text)
        return audio

    async def __generate_http(self, text):
        """Call Camb.ai TTS API and return audio bytes."""
        if len(text) < 3:
            logger.warning(f"Text too short for Camb.ai TTS (min 3 chars), skipping: '{text}'")
            return b""

        if len(text) > 3000:
            logger.warning(f"Text too long for Camb.ai TTS (max 3000 chars), truncating")
            text = text[:3000]

        payload = {
            "text": text,
            "voice_id": self.voice_id,
            "language": self.language,
            "speech_model": self.model,
            "output_configuration": {
                "format": "pcm_s16le",
                "sample_rate": self.sample_rate
            },
        }

        if self.user_instructions and self.model == "mars-instruct":
            payload["user_instructions"] = self.user_instructions

        headers = {
            "x-api-key": self.api_key,
            "Accept": "application/json",
            "Content-Type": "application/json"
        }

        try:
            if self.session is None:
                self.session = aiohttp.ClientSession()

            async with self.session.post(
                CAMB_TTS_URL,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=60.0)
            ) as response:
                if response.status == 401:
                    raise ValueError(
                        "Invalid Camb.ai API key. Set CAMB_API_KEY environment variable "
                        "with your API key from https://camb.ai"
                    )
                elif response.status == 403:
                    raise ValueError(
                        f"Voice ID {self.voice_id} is not accessible with your API key. "
                        "Check available voices at https://camb.ai"
                    )
                elif response.status == 404:
                    raise ValueError(f"Invalid voice ID: {self.voice_id}")
                elif response.status == 429:
                    raise ValueError("Camb.ai rate limit exceeded. Please wait before making more requests.")
                elif response.status != 200:
                    error_body = await response.text()
                    raise ValueError(f"Camb.ai API error {response.status}: {error_body}")

                # Read all audio chunks
                audio_buffer = io.BytesIO()
                async for chunk in response.content.iter_chunked(8192):
                    if chunk:
                        audio_buffer.write(chunk)

                wav_data = audio_buffer.getvalue()
                if not wav_data:
                    logger.warning(f"Camb.ai returned empty audio for text: '{text[:50]}'")
                    return b""

                self.synthesized_characters += len(text)

                return wav_data

        except asyncio.TimeoutError:
            logger.error(f"Camb.ai TTS request timed out for text: '{text[:50]}...'")
            raise
        except aiohttp.ClientError as e:
            logger.error(f"Camb.ai HTTP error: {e}")
            raise

    async def __generate_http_stream(self, text):
        """Call Camb.ai TTS API and yield audio chunks as they arrive."""
        if len(text) < 3:
            logger.warning(f"Text too short for Camb.ai TTS (min 3 chars), skipping: '{text}'")
            return

        if len(text) > 3000:
            logger.warning(f"Text too long for Camb.ai TTS (max 3000 chars), truncating")
            text = text[:3000]

        payload = {
            "text": text,
            "voice_id": self.voice_id,
            "language": self.language,
            "speech_model": self.model,
            "output_configuration": {
                "format": "pcm_s16le",
                "sample_rate": self.sample_rate
            },
        }

        if self.user_instructions and self.model == "mars-instruct":
            payload["user_instructions"] = self.user_instructions

        headers = {
            "x-api-key": self.api_key,
            "Accept": "application/json",
            "Content-Type": "application/json"
        }

        try:
            if self.session is None:
                self.session = aiohttp.ClientSession()

            async with self.session.post(
                CAMB_TTS_URL,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=60.0)
            ) as response:
                if response.status == 401:
                    raise ValueError(
                        "Invalid Camb.ai API key. Set CAMB_API_KEY environment variable "
                        "with your API key from https://camb.ai"
                    )
                elif response.status == 403:
                    raise ValueError(
                        f"Voice ID {self.voice_id} is not accessible with your API key. "
                        "Check available voices at https://camb.ai"
                    )
                elif response.status == 404:
                    raise ValueError(f"Invalid voice ID: {self.voice_id}")
                elif response.status == 429:
                    raise ValueError("Camb.ai rate limit exceeded. Please wait before making more requests.")
                elif response.status != 200:
                    error_body = await response.text()
                    raise ValueError(f"Camb.ai API error {response.status}: {error_body}")

                remainder = b""
                async for chunk in response.content.iter_chunked(8192):
                    if not chunk:
                        continue
                    data = remainder + chunk
                    # PCM s16le is 2 bytes per sample; keep alignment
                    usable = len(data) - (len(data) % 2)
                    if usable:
                        yield data[:usable]
                    remainder = data[usable:]

                if remainder:
                    yield remainder

                self.synthesized_characters += len(text)

        except asyncio.TimeoutError:
            logger.error(f"Camb.ai TTS request timed out for text: '{text[:50]}...'")
            raise
        except aiohttp.ClientError as e:
            logger.error(f"Camb.ai HTTP error: {e}")
            raise

    async def generate(self):
        """Main async generator that yields audio packets."""
        try:
            while True:
                message = await self.internal_queue.get()
                logger.info(f"Generating Camb.ai TTS response for message: {message}")
                meta_info, text = message.get("meta_info"), message.get("data")
                meta_info["text"] = text

                if not self.should_synthesize_response(meta_info.get('sequence_id')):
                    logger.info(
                        f"Not synthesizing text as the sequence_id ({meta_info.get('sequence_id')}) "
                        "is not in the list of sequence_ids present in the task manager."
                    )
                    return

                logger.info(f"Generating Camb.ai TTS for: {text}")

                if self.stream:
                    t_start = time.perf_counter()
                    async for chunk in self.__generate_http_stream(text):
                        chunk_meta = copy.deepcopy(meta_info)
                        if not self.first_chunk_generated:
                            chunk_meta["is_first_chunk"] = True
                            self.first_chunk_generated = True
                            logger.info(
                                f"Camb.ai TTFB: {time.perf_counter() - t_start:.3f}s"
                            )
                        yield create_ws_data_packet(chunk, chunk_meta)

                    if meta_info.get("end_of_llm_stream"):
                        meta_info["end_of_synthesizer_stream"] = True
                        self.first_chunk_generated = False
                        yield create_ws_data_packet(b"\x00", meta_info)
                else:
                    audio = await self.__generate_http(text)

                    if not self.first_chunk_generated:
                        meta_info["is_first_chunk"] = True
                        self.first_chunk_generated = True

                    if meta_info.get("end_of_llm_stream"):
                        meta_info["end_of_synthesizer_stream"] = True
                        self.first_chunk_generated = False

                    yield create_ws_data_packet(audio, meta_info)

        except Exception as e:
            logger.error(f"Error in Camb.ai generate: {e}")

    async def push(self, message):
        """Queue a message for synthesis."""
        logger.info(f"Pushed message to Camb.ai internal queue: {message}")
        self.internal_queue.put_nowait(message)

    async def open_connection(self):
        """No persistent connection needed for HTTP-based TTS."""
        pass

    def supports_websocket(self):
        return False

    def get_sleep_time(self):
        return 0.01 if self.stream else 0.2

    def get_synthesized_characters(self):
        """Return the total number of characters synthesized."""
        return self.synthesized_characters

    async def cleanup(self):
        """Close the aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None
