import os
import unicodedata

from aiobotocore.session import AioSession
from botocore.exceptions import BotoCoreError, ClientError
from contextlib import AsyncExitStack
from dotenv import load_dotenv

from .base_synthesizer import BaseSynthesizer
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.utils import convert_audio_to_wav, create_ws_data_packet
from bolna.memory.cache.inmemory_scalar_cache import InmemoryScalarCache

logger = configure_logger(__name__)
load_dotenv()


class PollySynthesizer(BaseSynthesizer):
    def __init__(self, voice, language, audio_format="pcm", sampling_rate=8000, stream=False,
                 engine="neural", buffer_size=400, speaking_rate="100%", volume="0dB",
                 caching=True, **kwargs):
        super().__init__(kwargs.get("task_manager_instance"), stream, buffer_size)
        self.engine = engine
        self.format = "pcm" if audio_format == "pcm" else "mp3"
        self.voice = self._resolve_voice(voice)
        self.language = language
        self.sample_rate = str(sampling_rate)
        self.model = engine
        self.speaking_rate = speaking_rate
        self.volume = volume
        self.caching = caching
        if caching:
            self.cache = InmemoryScalarCache()

    def supports_websocket(self):
        return False

    @staticmethod
    def _resolve_voice(text):
        return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))

    @staticmethod
    async def _create_client(service, session, exit_stack):
        if os.getenv("AWS_ACCESS_KEY_ID"):
            return await exit_stack.enter_async_context(session.create_client(
                service,
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                region_name=os.getenv("AWS_REGION"),
            ))
        return await exit_stack.enter_async_context(session.create_client(service))

    # ------------------------------------------------------------------
    # HTTP synthesis
    # ------------------------------------------------------------------

    async def _generate_http(self, text):
        session = AioSession()
        async with AsyncExitStack() as exit_stack:
            polly = await self._create_client("polly", session, exit_stack)
            logger.info(f"Generating TTS for text: {text}, SampleRate {self.sample_rate} format {self.format}")
            try:
                response = await polly.synthesize_speech(
                    Engine=self.engine, Text=text, OutputFormat=self.format,
                    VoiceId=self.voice, LanguageCode=self.language, SampleRate=self.sample_rate,
                )
            except (BotoCoreError, ClientError) as error:
                logger.error(error)
                return None
            else:
                return await response["AudioStream"].read()

    async def synthesize(self, text):
        try:
            audio = await self._generate_http(text)
            if self.format == "mp3":
                audio = convert_audio_to_wav(audio, source_format="mp3")
            return audio
        except Exception as e:
            logger.error(f"Could not synthesize {e}")

    # ------------------------------------------------------------------
    # generate / push
    # ------------------------------------------------------------------

    async def generate(self):
        while True:
            logger.info("Generating TTS response")
            message = await self.internal_queue.get()
            logger.info(f"Generating TTS response for message: {message}")
            meta_info, text = message.get("meta_info"), message.get("data")

            if not self.should_synthesize_response(meta_info.get("sequence_id")):
                logger.info(f"Not synthesizing: sequence_id {meta_info.get('sequence_id')} not current")
                return

            if self.caching:
                logger.info("Caching is on")
                if self.cache.get(text):
                    logger.info(f"Cache hit: {text}")
                    audio = self.cache.get(text)
                else:
                    logger.info(f"Not a cache hit {list(self.cache.data_dict)}")
                    self.synthesized_characters += len(text)
                    audio = await self._generate_http(text)
                    self.cache.set(text, audio)
            else:
                self.synthesized_characters += len(text)
                audio = await self._generate_http(text)

            if self.format == "mp3":
                audio = convert_audio_to_wav(audio, source_format="mp3")

            self._stamp_first_chunk(meta_info)
            self._stamp_end_of_stream(meta_info)

            meta_info["text"] = text
            meta_info["format"] = "wav"
            meta_info["text_synthesized"] = f"{text} "
            self._stamp_mark_id(meta_info)
            yield create_ws_data_packet(audio, meta_info)

    async def push(self, message):
        logger.info("Pushed message to internal queue")
        self.internal_queue.put_nowait(message)
