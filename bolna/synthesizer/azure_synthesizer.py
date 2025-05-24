import os
import uuid
from dotenv import load_dotenv
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.utils import convert_audio_to_wav, create_ws_data_packet, pcm_to_wav_bytes, resample, wav_bytes_to_pcm
from bolna.memory.cache.inmemory_scalar_cache import InmemoryScalarCache
from .base_synthesizer import BaseSynthesizer
import azure.cognitiveservices.speech as speechsdk

logger = configure_logger(__name__)
load_dotenv()


class AzureSynthesizer(BaseSynthesizer):
    def __init__(self, voice, language, model="neural", stream=False, sampling_rate=16000, buffer_size=400, caching=True, **kwargs):
        super().__init__(kwargs.get("task_manager_instance", None), stream, buffer_size)
        self.model = model
        self.language = language
        self.voice = f"{language}-{voice}{model}" #hard-code for testing to self.voice = "en-US-JennyNeural"
        logger.synt(f"{self.voice} initialized")
        self.sample_rate = str(sampling_rate)
        self.first_chunk_generated = False
        self.stream = stream
        self.synthesized_characters = 0
        self.caching = caching
        if caching:
            self.cache = InmemoryScalarCache()

        # Initialize Azure Speech Config
        self.subscription_key = kwargs.get("synthesizer_key", os.getenv("AZURE_SPEECH_KEY"))
        self.region = kwargs.get("region", os.getenv("AZURE_SPEECH_REGION"))
        self.speech_config = speechsdk.SpeechConfig(subscription=self.subscription_key, region=self.region)
        if self.stream:
            self.speech_config.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Riff8Khz16BitMonoPcm)
        else:
            self.speech_config.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Riff16Khz16BitMonoPcm)
        self.speech_config.speech_synthesis_voice_name = self.voice

    def get_synthesized_characters(self):
        return self.synthesized_characters

    def get_engine(self):
        return self.model

    def supports_websocket(self):
        return False

    async def synthesize(self, text):
        audio = await self.__generate_http(text)
        return audio

    async def __generate_http(self, text):
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=self.speech_config, audio_config=None)
        result = synthesizer.speak_text_async(text).get()
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            return result.audio_data
        else:
            logger.error(f"Speech synthesis failed: {result.reason}")
            return None

    async def generate(self):
        while True:
            logger.synt("Generating TTS response")
            message = await self.internal_queue.get()
            logger.synt(f"Generating TTS response for message: {message}")
            meta_info, text = message.get("meta_info"), message.get("data")

            if not self.should_synthesize_response(meta_info.get('sequence_id')):
                logger.synt(f"Not synthesizing text as the sequence_id ({meta_info.get('sequence_id')}) of it is not in the list of sequence_ids present in the task manager.")
                return

            if self.caching:
                logger.synt(f"Caching is on")
                if self.cache.get(text):
                    logger.synt(f"Cache hit and hence returning quickly {text}")
                    message = self.cache.get(text)
                else:
                    logger.synt(f"Not a cache hit {list(self.cache.data_dict)}")
                    self.synthesized_characters += len(text)
                    message = await self.__generate_http(text)
                    self.cache.set(text, message)
            else:
                logger.synt(f"No caching present")
                self.synthesized_characters += len(text)
                message = await self.__generate_http(text)

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

    async def open_connection(self):
        pass

    async def push(self, message):
        logger.synt(f"Pushed message to internal queue {message}")
        self.internal_queue.put_nowait(message)