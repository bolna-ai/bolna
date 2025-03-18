from collections import deque
import os
from dotenv import load_dotenv
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.utils import convert_audio_to_wav, create_ws_data_packet, pcm_to_wav_bytes, resample
from .base_synthesizer import BaseSynthesizer
from openai import AsyncOpenAI
import io

logger = configure_logger(__name__)
load_dotenv()

class OPENAISynthesizer(BaseSynthesizer):
    def __init__(self, voice, audio_format="mp3", model = "tts-1", stream=False, sampling_rate=8000, buffer_size=400, **kwargs):
        super().__init__(kwargs.get("task_manager_instance", None), stream, buffer_size, is_web_based_call=kwargs.get("is_web_based_call", False),
                         is_precise_transcript_generation_enabled=kwargs.get("is_precise_transcript_generation_enabled"))
        self.format = self.get_format(audio_format.lower())
        self.voice = voice
        self.sample_rate = sampling_rate
        api_key = kwargs.get("synthesizer_key", os.getenv("OPENAI_API_KEY"))
        self.async_client = AsyncOpenAI(api_key= api_key)
        self.model = model
        self.first_chunk_generated = False 
        self.text_queue = deque()
        self.stream = False
        if type(self.sample_rate) is str:
            self.sample_rate = int(self.sample_rate)
        
    # Ensuring we can only do wav outputs becasue mulaw conversion for others messes up twilio
    def get_format(self, format):
        return "mp3"
    
    async def synthesize(self, text):
        #This is used for one off synthesis mainly for use cases like voice lab and IVR
        audio = await self.__generate_http(text)
        return audio

    async def __generate_http(self, text):
        spoken_response = await self.async_client.audio.speech.create(
            model=self.model,
            voice=self.voice,
            response_format=self.format,
            input=text
            )

        buffer = io.BytesIO()
        for chunk in spoken_response.iter_bytes(chunk_size=4096):
            buffer.write(chunk)
        buffer.seek(0)
        return buffer.getvalue()
    
    async def __generate_stream(self, text):
        spoken_response = await self.async_client.audio.speech.create(
            model=self.model,
            voice=self.voice,
            response_format="mp3",
            input=text
            )

        for chunk in spoken_response.iter_bytes(chunk_size=4096):
            yield chunk

    async def generate(self):
        try:
            while True:
                message = await self.internal_queue.get()
                logger.info(f"Generating TTS response for message: {message}")
                meta_info, text = message.get("meta_info"), message.get("data")
                meta_info["text"] = text

                if not self.should_synthesize_response(meta_info.get('sequence_id')):
                    logger.info(
                        f"Not synthesizing text as the sequence_id ({meta_info.get('sequence_id')}) of it is not in the list of sequence_ids present in the task manager.")
                    return

                if self.stream:
                    async for chunk in self.__generate_stream(text):
                        if not self.first_chunk_generated:
                            meta_info["is_first_chunk"] = True
                            self.first_chunk_generated = True
                        yield create_ws_data_packet(resample(convert_audio_to_wav(chunk, 'mp3'), self.sample_rate, format="wav"), meta_info)
                        
                    if "end_of_llm_stream" in meta_info and meta_info["end_of_llm_stream"]:
                        meta_info["end_of_synthesizer_stream"] = True
                        self.first_chunk_generated = False
                        yield create_ws_data_packet(b"\x00", meta_info)

                else:
                    logger.info(f"Generating without a stream")
                    audio = await self.__generate_http(text)
                    if not self.first_chunk_generated:
                        meta_info["is_first_chunk"] = True
                        self.first_chunk_generated = True
                    
                    if "end_of_llm_stream" in meta_info and meta_info["end_of_llm_stream"]:
                        meta_info["end_of_synthesizer_stream"] = True
                        self.first_chunk_generated = False 
                    yield create_ws_data_packet(resample(convert_audio_to_wav(audio, 'mp3'), self.sample_rate, format="wav"), meta_info)

        except Exception as e:
                logger.error(f"Error in openai generate {e}")

    async def open_connection(self):
        pass

    async def push(self, message):
        logger.info(f"Pushed message to internal queue {message}")
        self.internal_queue.put_nowait(message)
