import os
import io
import aiohttp
from dotenv import load_dotenv
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.utils import create_ws_data_packet, pcm_to_wav_bytes, resample
from .base_synthesizer import BaseSynthesizer

logger = configure_logger(__name__)
load_dotenv()

CAMB_TTS_URL = "https://client.camb.ai/apis/tts-stream"


class CambSynthesizer(BaseSynthesizer):
    def __init__(
        self,
        voice_id,
        voice,
        model="mars-8-flash",
        language="en-us",
        audio_format="pcm",
        sampling_rate=8000,
        stream=False,
        buffer_size=400,
        speed=1.0,
        **kwargs
    ):
        super().__init__(kwargs.get("task_manager_instance", None), stream, buffer_size)
        self.voice_id = int(voice_id)
        self.voice = voice
        self.model = model
        self.language = language
        self.speed = speed
        self.sample_rate = int(sampling_rate) if isinstance(sampling_rate, str) else sampling_rate
        self.stream = False  # Camb.ai uses HTTP chunked streaming, not WebSocket
        self.first_chunk_generated = False
        self.synthesized_characters = 0

        api_key = kwargs.get("synthesizer_key", os.getenv("CAMB_API_KEY"))
        if not api_key:
            raise ValueError(
                "Camb.ai API key not found. Set CAMB_API_KEY environment variable "
                "or pass synthesizer_key parameter."
            )
        self.api_key = api_key

    async def synthesize(self, text):
        """One-off synthesis for use cases like voice lab and IVR."""
        audio = await self.__generate_http(text)
        return audio

    async def __generate_http(self, text):
        """Call Camb.ai TTS API and return audio bytes."""
        if len(text) < 3:
            logger.warning(f"Text too short for Camb.ai TTS (min 3 chars): '{text}'")
            text = text + "..."  # Pad to minimum length

        if len(text) > 3000:
            logger.warning(f"Text too long for Camb.ai TTS (max 3000 chars), truncating")
            text = text[:3000]

        payload = {
            "text": text,
            "voice_id": self.voice_id,
            "language": self.language,
            "speech_model": self.model,
            "output_configuration": {
                "format": "pcm_s16le"
            },
            "voice_settings": {
                "speed": self.speed
            }
        }

        headers = {
            "x-api-key": self.api_key,
            "Accept": "application/json",
            "Content-Type": "application/json"
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
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

                    pcm_data = audio_buffer.getvalue()
                    self.synthesized_characters += len(text)

                    # Convert PCM (24kHz) to WAV and resample to target sample rate
                    wav_data = pcm_to_wav_bytes(pcm_data, sample_rate=24000)
                    resampled_audio = resample(wav_data, self.sample_rate, format="wav")

                    return resampled_audio

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
                audio = await self.__generate_http(text)

                if not self.first_chunk_generated:
                    meta_info["is_first_chunk"] = True
                    self.first_chunk_generated = True

                if "end_of_llm_stream" in meta_info and meta_info["end_of_llm_stream"]:
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

    def get_synthesized_characters(self):
        """Return the total number of characters synthesized."""
        return self.synthesized_characters
