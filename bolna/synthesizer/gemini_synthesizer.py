from collections import deque
import os
from dotenv import load_dotenv
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.utils import convert_audio_to_wav, create_ws_data_packet, resample
from .base_synthesizer import BaseSynthesizer
from google import genai
from google.genai import types
import wave
import io

logger = configure_logger(__name__)
load_dotenv()


class GeminiSynthesizer(BaseSynthesizer):
    def __init__(self, voice, voice_name, model="gemini-2.5-flash-preview-tts", audio_format="wav", 
                 sampling_rate=8000, buffer_size=400, language="en", **kwargs):
        super().__init__(kwargs.get("task_manager_instance", None), False, buffer_size)  # stream=False for Gemini
        self.voice = voice  # Human-readable voice name
        self.voice_name = voice_name  # API voice name (e.g., "Aoede", "Kore")
        self.model = model
        self.language = language
        self.audio_format = audio_format
        self.sample_rate = sampling_rate
        api_key = kwargs.get("synthesizer_key", os.getenv("GOOGLE_API_KEY"))
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is required for Gemini TTS")
        self.client = genai.Client(api_key=api_key)
        self.first_chunk_generated = False
        self.text_queue = deque()
        if type(self.sample_rate) is str:
            self.sample_rate = int(self.sample_rate)
        logger.info(f"Initialized GeminiSynthesizer with model={self.model}, voice_name={self.voice_name}")
    
    async def synthesize(self, text):
        """One-off synthesis for use cases like voice lab and IVR"""
        audio = await self.__generate_http(text)
        return audio

    async def __generate_http(self, text):
        """Core HTTP synthesis using Gemini API"""
        try:
            logger.info(f"Generating Gemini TTS for text: {text[:100]}...")
            
            # Call Gemini API with speech configuration
            response = self.client.models.generate_content(
                model=self.model,
                contents=text,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=self.voice_name
                            )
                        )
                    )
                )
            )
            
            # Extract PCM audio data from response
            audio_data = response.candidates[0].content.parts[0].inline_data.data
            
            # Create WAV file with proper headers (24kHz, 16-bit, mono)
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, "wb") as wf:
                wf.setnchannels(1)  # Mono
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(24000)  # 24kHz sample rate (Gemini output)
                wf.writeframes(audio_data)
            
            wav_buffer.seek(0)
            return wav_buffer.getvalue()
            
        except Exception as e:
            logger.error(f"Error generating Gemini TTS: {e}")
            raise

    async def generate(self):
        """Async generator for streaming to task manager"""
        try:
            while True:
                message = await self.internal_queue.get()
                logger.info(f"Generating Gemini TTS response for message: {message}")
                meta_info, text = message.get("meta_info"), message.get("data")
                meta_info["text"] = text

                if not self.should_synthesize_response(meta_info.get('sequence_id')):
                    logger.info(
                        f"Not synthesizing text as the sequence_id ({meta_info.get('sequence_id')}) "
                        f"is not in the list of sequence_ids present in the task manager.")
                    return

                # Gemini TTS is HTTP-only (no streaming)
                logger.info(f"Generating Gemini TTS without streaming")
                audio = await self.__generate_http(text)
                
                if not self.first_chunk_generated:
                    meta_info["is_first_chunk"] = True
                    self.first_chunk_generated = True
                
                if "end_of_llm_stream" in meta_info and meta_info["end_of_llm_stream"]:
                    meta_info["end_of_synthesizer_stream"] = True
                    self.first_chunk_generated = False
                
                # Resample from 24kHz (Gemini output) to target sample rate (usually 8kHz for telephony)
                yield create_ws_data_packet(resample(audio, self.sample_rate, format="wav"), meta_info)

        except Exception as e:
            logger.error(f"Error in Gemini TTS generate: {e}")

    async def open_connection(self):
        """Placeholder - Gemini doesn't use WebSocket"""
        pass

    async def push(self, message):
        """Add message to internal queue for processing"""
        logger.info(f"Pushed message to Gemini TTS internal queue: {message}")
        self.internal_queue.put_nowait(message)
