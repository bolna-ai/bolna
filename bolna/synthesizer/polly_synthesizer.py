import os
import uuid
import unicodedata
import xml.sax.saxutils as sax
from dotenv import load_dotenv
from botocore.exceptions import BotoCoreError, ClientError
from aiobotocore.session import AioSession
from contextlib import AsyncExitStack
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.utils import convert_audio_to_wav, create_ws_data_packet
from bolna.memory.cache.inmemory_scalar_cache import InmemoryScalarCache
from .base_synthesizer import BaseSynthesizer

logger = configure_logger(__name__)
load_dotenv()


class PollySynthesizer(BaseSynthesizer):
    def __init__(self, voice, language, audio_format="pcm", sampling_rate=8000, stream=False, engine="neural",
                 buffer_size=400, speaking_rate=None, volume=None, pitch=None, emphasis=None,
                 auto_breaths=False, newscaster=False,
                 whispered=False, drc=False, vocal_tract_length=None, soft_phonation=False,
                 conversational=False, lang=None, caching=True, **kwargs):
        super().__init__(kwargs.get("task_manager_instance", None), stream, buffer_size)
        self.engine = engine
        self.format = self.get_format(audio_format.lower())
        self.voice = self.resolve_voice(voice)
        self.language = language
        self.sample_rate = str(sampling_rate)
        self.client = None
        self.first_chunk_generated = False
        # Prosody — rate and volume work on all engines
        self.speaking_rate = speaking_rate   # e.g. "slow", "85%", "120%"
        self.volume = volume                 # e.g. "loud", "+6dB"
        # Standard-engine-only features
        self.pitch = pitch                   # e.g. "high", "+5%"
        self.emphasis = emphasis             # "strong" | "moderate" | "reduced"
        self.auto_breaths = auto_breaths     # wrap in <amazon:auto-breaths>
        # Standard-engine-only voice effects
        self.whispered = whispered           # whispered voice effect
        self.drc = drc                       # dynamic range compression
        self.vocal_tract_length = vocal_tract_length  # e.g. "+15%", "-10%"
        self.soft_phonation = soft_phonation # soft/breathy phonation
        # Neural-voice-only features
        self.newscaster = newscaster         # wrap in <amazon:domain name="news">
        self.conversational = conversational # wrap in <amazon:domain name="conversational">
        # Language switching
        self.lang = lang                     # e.g. "hi-IN"
        self.synthesized_characters = 0
        self.caching = caching
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

    def _build_ssml(self, text: str):
        """
        Build a Polly SSML document from plain text using the agent's config.

        Engine-aware rules (from AWS docs):
          - rate, volume          → all engines
          - pitch, emphasis,
            auto_breaths, drc,
            vocal_tract_length,
            soft_phonation,
            whispered             → Standard engine ONLY
          - newscaster,
            conversational        → select Neural voices ONLY
          - lang                  → all engines

        Returns None when no SSML config is active, so the caller falls back
        to plain-text synthesis (TextType='text').
        """
        # Pass-through: text is already a full SSML document
        if self.is_ssml(text):
            return text

        is_standard = self.engine.lower() == "standard"
        is_neural = self.engine.lower() == "neural"

        has_prosody = any([self.speaking_rate, self.volume,
                           self.pitch and is_standard])
        has_standard_only = is_standard and any([
            self.emphasis, self.auto_breaths, self.whispered,
            self.drc, self.vocal_tract_length, self.soft_phonation
        ])
        has_neural_only = is_neural and any([self.newscaster, self.conversational])
        has_lang = bool(self.lang)
        has_inline = self.has_inline_ssml(text)

        if not any([has_prosody, has_standard_only, has_neural_only, has_lang, has_inline]):
            return None  # nothing to wrap — caller uses TextType='text'

        body = text if has_inline else sax.escape(text)

        # --- Language switch ---
        if has_lang:
            body = f'<lang xml:lang="{self.lang}">{body}</lang>'

        # --- Standard-only: emphasis ---
        if is_standard and self.emphasis:
            body = f'<emphasis level="{self.emphasis}">{body}</emphasis>'

        # --- Prosody: rate + volume (all engines); pitch (standard only) ---
        prosody_attrs = {}
        if self.speaking_rate:
            prosody_attrs["rate"] = self.speaking_rate
        if self.volume:
            prosody_attrs["volume"] = self.volume
        if self.pitch and is_standard:
            prosody_attrs["pitch"] = self.pitch
        if prosody_attrs:
            attr_str = " ".join(f'{k}="{v}"' for k, v in prosody_attrs.items())
            body = f'<prosody {attr_str}>{body}</prosody>'

        # --- Standard-only: automatic breathing sounds ---
        if is_standard and self.auto_breaths:
            body = f'<amazon:auto-breaths>{body}</amazon:auto-breaths>'

        # --- Standard-only: vocal tract length (changes voice character) ---
        if is_standard and self.vocal_tract_length:
            body = f'<amazon:effect vocal-tract-length="{self.vocal_tract_length}">{body}</amazon:effect>'

        # --- Standard-only: dynamic range compression ---
        if is_standard and self.drc:
            body = f'<amazon:effect name="drc">{body}</amazon:effect>'

        # --- Standard-only: soft phonation (breathy voice) ---
        if is_standard and self.soft_phonation:
            body = f'<amazon:effect phonation="soft">{body}</amazon:effect>'

        # --- Standard-only: whispered voice ---
        if is_standard and self.whispered:
            body = f'<amazon:effect name="whispered">{body}</amazon:effect>'

        # --- Neural-only: domain styles ---
        if is_neural and self.newscaster:
            body = f'<amazon:domain name="news">{body}</amazon:domain>'
        elif is_neural and self.conversational:
            body = f'<amazon:domain name="conversational">{body}</amazon:domain>'

        return f'<speak>{body}</speak>'

    async def __generate_http(self, text):
        session = AioSession()
        async with AsyncExitStack() as exit_stack:
            polly = await self.create_client("polly", session, exit_stack)
            ssml = self._build_ssml(text)
            logger.info(
                f"Generating TTS response for text: {text}, "
                f"SampleRate {self.sample_rate} format {self.format} "
                f"ssml={'yes' if ssml else 'no'}"
            )
            try:
                if ssml:
                    response = await polly.synthesize_speech(
                        Engine=self.engine,
                        Text=ssml,
                        TextType="ssml",
                        OutputFormat=self.format,
                        VoiceId=self.voice,
                        LanguageCode=self.language,
                        SampleRate=self.sample_rate
                    )
                else:
                    response = await polly.synthesize_speech(
                        Engine=self.engine,
                        Text=text,
                        TextType="text",
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
            meta_info["mark_id"] = str(uuid.uuid4())
            yield create_ws_data_packet(message, meta_info)

    async def push(self, message):
        logger.info("Pushed message to internal queue")
        self.internal_queue.put_nowait(message)
