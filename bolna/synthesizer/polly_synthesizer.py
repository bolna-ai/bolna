import os
import uuid
import unicodedata
from dotenv import load_dotenv
from botocore.exceptions import BotoCoreError, ClientError
from aiobotocore.session import AioSession
from contextlib import AsyncExitStack
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.utils import convert_audio_to_wav, create_ws_data_packet, pcm_to_wav_bytes, resample
from bolna.memory.cache.inmemory_scalar_cache import InmemoryScalarCache
from .base_synthesizer import BaseSynthesizer

logger = configure_logger(__name__)
load_dotenv()


class PollySynthesizer(BaseSynthesizer):
    def __init__(self, voice, language, audio_format="pcm", sampling_rate=8000, stream=False, engine="neural",
                 buffer_size=400, speaking_rate = "100%", volume = "0dB", caching= True, **kwargs):
        super().__init__(kwargs.get("task_manager_instance", None), stream, buffer_size, is_web_based_call=kwargs.get("is_web_based_call", False),
                         is_precise_transcript_generation_enabled=kwargs.get("is_precise_transcript_generation_enabled"))
        self.engine = engine
        self.format = self.get_format(audio_format.lower())
        self.voice = self.resolve_voice(voice)
        self.language = language
        self.sample_rate = str(sampling_rate)
        self.client = None
        self.first_chunk_generated = False
        self.speaking_rate = speaking_rate
        self.volume = volume
        self.synthesized_characters = 0
        self.caching = caching
        self.slicing_range = int(sampling_rate / 2)
        if caching:
            self.cache = InmemoryScalarCache()

    def get_synthesized_characters(self):
        return self.synthesized_characters
    
    def get_engine(self):
        return self.engine

    def resolve_voice(self, text):
        return ''.join(c for c in unicodedata.normalize('NFKD', text) if not unicodedata.combining(c))

    def supports_websocket(self):
        return False

    def get_format(self, audio_format):
        if audio_format == "pcm":
            return "pcm"
        else:
            return "mp3"

    @staticmethod
    async def create_client(service: str, session: AioSession, exit_stack: AsyncExitStack):
        if os.getenv('AWS_ACCESS_KEY_ID'):
            return await exit_stack.enter_async_context(session.create_client(
                service,
                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                region_name=os.getenv('AWS_REGION')
            ))
        else:
            return await exit_stack.enter_async_context(session.create_client(service))

    async def __generate_http(self, text):
        session = AioSession()
        async with AsyncExitStack() as exit_stack:
            polly = await self.create_client("polly", session, exit_stack)
            logger.info(f"Generating TTS response for text: {text}, SampleRate {self.sample_rate} format {self.format}")
            # input = f"""
            # <speak> 
            #     <amazon:auto-breaths volume= "x-loud" frequency="x-high" duration="x-long"> 
            #         <prosody volume="{self.volume}" rate="{self.speaking_rate}"> {text} 
            #         </prosody> 
            #     </amazon:auto-breaths>
            # </speak>
            # """

            try:
                response = await polly.synthesize_speech(
                    Engine=self.engine,
                    Text=text,
                    OutputFormat=self.format,
                    VoiceId=self.voice,
                    LanguageCode=self.language,
                    SampleRate=self.sample_rate
                )
            except (BotoCoreError, ClientError) as error:
                logger.error(error)
            else:
                return await response["AudioStream"].read()

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
            logger.info("Generating TTS response")
            message = await self.internal_queue.get()
            logger.info(f"Generating TTS response for message: {message}")
            meta_info, text = message.get("meta_info"), message.get("data")

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
            meta_info['format'] = 'wav'
            meta_info["text_synthesized"] = f"{text} "
            if not self.is_web_based_call and self.is_precise_transcript_generation_enabled:
                async for chunk in self.break_audio_into_chunks(message, self.slicing_range, meta_info,
                                                                override_end_of_synthesizer_stream=True):
                    yield chunk
            else:
                meta_info["mark_id"] = str(uuid.uuid4())
                yield create_ws_data_packet(message, meta_info)

    async def push(self, message):
        logger.info("Pushed message to internal queue")
        self.internal_queue.put_nowait(message)
