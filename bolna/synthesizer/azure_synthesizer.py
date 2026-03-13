import os
import re
import uuid
import asyncio
import time
import xml.sax.saxutils as sax
from dotenv import load_dotenv
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.utils import create_ws_data_packet
from bolna.memory.cache.inmemory_scalar_cache import InmemoryScalarCache
from .base_synthesizer import BaseSynthesizer
import azure.cognitiveservices.speech as speechsdk
from azure.cognitiveservices.speech import CancellationErrorCode

logger = configure_logger(__name__)
load_dotenv()


class AzureSynthesizer(BaseSynthesizer):
    def __init__(self, voice, language, model="neural", stream=False, sampling_rate=8000, buffer_size=150, caching=True, speed=None, **kwargs):
        super().__init__(kwargs.get("task_manager_instance", None), stream, buffer_size)
        self.model = model
        self.language = language
        self.voice = f"{language}-{voice}{model}"
        logger.info(f"{self.voice} initialized")
        self.sample_rate = str(sampling_rate)
        self.first_chunk_generated = False
        self.stream = stream
        self.synthesized_characters = 0
        self.caching = False
        self.speed = speed
        if caching:
            self.cache = InmemoryScalarCache()
        self.loop = asyncio.get_event_loop()

        # Initialize Azure Speech Config
        self.subscription_key = kwargs.get("synthesizer_key", os.getenv("AZURE_SPEECH_KEY"))
        self.region = kwargs.get("region", os.getenv("AZURE_SPEECH_REGION"))
        self.speech_config = speechsdk.SpeechConfig(subscription=self.subscription_key, region=self.region)
        self.speech_config.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Raw8Khz16BitMonoPcm)
        self.speech_config.speech_synthesis_voice_name = self.voice

        # Add comprehensive latency tracking
        self.latency_stats = {
            "request_count": 0,
            "total_first_byte_latency": 0,
            "min_latency": float('inf'),
            "max_latency": 0.0
        }
        self.connection_requested_at = None

        # Implement connection pooling with a synthesizer factory
        self.synthesizer_pool = []
        self.max_pool_size = 5  # Tune based on your traffic

        # KALLABOT: Initialize Kallabot-specific voice features
        self._init_kallabot_features(voice, kwargs)

    # ------------------------------------------------------------------ #
    # KALLABOT: Feature initialization — Dragon HD, Personal Voice,
    #           temperature, auto-locale, HD performance settings.
    # ------------------------------------------------------------------ #

    def _init_kallabot_features(self, original_voice_name, kwargs):  # KALLABOT
        """Detect and configure Kallabot-specific voice capabilities.

        Sets flags that later control whether _build_ssml() produces
        enhanced SSML (Personal Voice, Dragon HD, temperature, auto-locale)
        instead of the stock prosody-only SSML or plain-text path.
        """

        # --- Locale --------------------------------------------------- #
        # Default to self.language; callers may override via kwargs.
        self.locale = kwargs.get("locale", self.language)

        # --- Temperature control for Dragon voices -------------------- #
        self.temperature = kwargs.get("temperature", None)
        if self.temperature is not None:
            try:
                self.temperature = float(self.temperature)
                if not (0 <= self.temperature <= 1):
                    logger.warning(f"KALLABOT: Temperature {self.temperature} out of range [0,1], resetting to None")
                    self.temperature = None
            except (ValueError, TypeError):
                logger.warning(f"KALLABOT: Invalid temperature value {self.temperature}, resetting to None")
                self.temperature = None

        # --- Personal Voice detection --------------------------------- #
        # Formats accepted:
        #   "UUID:model"  e.g. "12e8d4b0-…:DragonLatestNeural"
        #   Explicit kwarg  speakerProfileId / personal_voice_id
        #   Voice name containing a known personal-voice base model
        uuid_colon_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}:'
        personal_voice_model_patterns = [
            'DragonLatestNeural',
            'PheonixLatestNeural',   # legacy spelling
            'PhoenixLatestNeural',
        ]

        self.speaker_profile_id = kwargs.get(
            "speakerProfileId",
            kwargs.get("personal_voice_id", None),
        )
        self.personal_voice_model = None

        # Detect from the fully-qualified voice string
        self.is_personal_voice = bool(re.match(uuid_colon_pattern, self.voice.lower()))
        if not self.is_personal_voice and self.speaker_profile_id:
            self.is_personal_voice = True
        if not self.is_personal_voice:
            for pat in personal_voice_model_patterns:
                if pat in self.voice:
                    self.is_personal_voice = True
                    break

        # Parse speaker-profile-id and model out of "UUID:Model" voice string
        if self.is_personal_voice:
            if ':' in self.voice:
                parts = self.voice.split(':', 1)
                if len(parts) == 2:
                    if not self.speaker_profile_id:
                        self.speaker_profile_id = parts[0]
                    self.personal_voice_model = parts[1]
                    logger.info(
                        f"KALLABOT: Personal voice detected — "
                        f"Speaker ID: {self.speaker_profile_id}, Model: {self.personal_voice_model}"
                    )
            else:
                self.personal_voice_model = self.voice
                logger.info(f"KALLABOT: Personal voice base model detected: {self.personal_voice_model}")

        # --- Dragon HD Voice detection -------------------------------- #
        multilingual_patterns = [
            'DragonHDLatestNeural',
            'DragonHDFlashLatestNeural',
            'MultilingualNeural',
        ]
        self.is_dragon_hd_voice = False
        for pat in multilingual_patterns:
            if pat in self.voice:
                self.is_dragon_hd_voice = True
                break

        self.is_dragon_voice = "Dragon" in self.voice
        self.is_multilingual_voice = self.is_dragon_hd_voice or self.is_personal_voice

        # --- Master flag: should enhanced SSML be used? --------------- #
        # Enhanced SSML activates for Personal Voice, Dragon HD, Dragon
        # voices with temperature, or when auto-locale is requested.
        self.use_kallabot_ssml = (
            self.is_personal_voice
            or self.is_dragon_hd_voice
            or (self.is_dragon_voice and self.temperature is not None)
            or (self.locale and str(self.locale).lower() == "auto")
        )

        if self.use_kallabot_ssml:
            logger.info(
                f"KALLABOT: Enhanced SSML mode ACTIVE — "
                f"personal_voice={self.is_personal_voice}, "
                f"dragon_hd={self.is_dragon_hd_voice}, "
                f"dragon={self.is_dragon_voice}, "
                f"multilingual={self.is_multilingual_voice}, "
                f"locale={self.locale}, "
                f"temperature={self.temperature}"
            )
            # Apply HD performance tuning for Dragon HD / Personal voices
            if self.is_dragon_hd_voice or self.is_personal_voice:
                self._configure_hd_performance_settings()
        else:
            logger.info("KALLABOT: Enhanced SSML mode inactive — using standard code path")

    def _configure_hd_performance_settings(self):  # KALLABOT
        """Apply buffer / latency / WebSocket-v2 tuning for HD and Personal voices."""
        try:
            # WebSocket V2 endpoint for lower latency
            ws_v2 = f"wss://{self.region}.tts.speech.microsoft.com/cognitiveservices/websocket/v2"
            self.speech_config.set_property(
                speechsdk.PropertyId.SpeechServiceConnection_Url, ws_v2
            )

            # Minimal silence timeouts
            self.speech_config.set_property(
                speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs, "1"
            )
            self.speech_config.set_property(
                speechsdk.PropertyId.SpeechServiceConnection_EndSilenceTimeoutMs, "1"
            )

            # Post-processing and compression
            self.speech_config.set_property(
                speechsdk.PropertyId.SpeechServiceResponse_PostProcessingOption, "TrueText"
            )
            self.speech_config.set_property(
                speechsdk.PropertyId.SpeechServiceConnection_SynthEnableCompressedAudioTransmission,
                "true",
            )

            # Streaming buffer / timeout tuning
            self.speech_config.set_property(
                speechsdk.PropertyId.Speech_SegmentationSilenceTimeoutMs, "500"
            )
            self.speech_config.set_property(
                speechsdk.PropertyId.SpeechSynthesis_FrameTimeoutInterval, "5000000"
            )
            self.speech_config.set_property(
                speechsdk.PropertyId.SpeechSynthesis_RtfTimeoutThreshold, "20"
            )

            logger.info("KALLABOT: HD performance settings applied")
        except Exception as exc:
            logger.warning(f"KALLABOT: Could not apply HD performance settings: {exc}")

    # ------------------------------------------------------------------ #
    # KALLABOT: Enhanced SSML builders
    # ------------------------------------------------------------------ #

    def _build_personal_voice_ssml(self, text):  # KALLABOT
        """Build SSML for Azure Personal Voice with speaker-profile-id.

        Handles both UUID:Model format and bare personal-voice model names.
        Supports auto-locale (``locale="auto"``) and explicit locale switching
        via ``<lang xml:lang="…">``.
        """
        escaped = sax.escape(text)
        creation_locale = "en-US"  # root hint; Azure docs recommend voice creation locale

        if self.speaker_profile_id:
            if self.locale and str(self.locale).lower() == "auto":
                ssml = (
                    f"<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' "
                    f"xmlns:mstts='http://www.w3.org/2001/mstts' xml:lang='{creation_locale}'>"
                    f"<voice name='{self.personal_voice_model}'>"
                    f"<mstts:ttsembedding speakerProfileId='{self.speaker_profile_id}'>"
                    f"{escaped}"
                    f"</mstts:ttsembedding>"
                    f"</voice>"
                    f"</speak>"
                )
            else:
                target = self.locale or creation_locale
                ssml = (
                    f"<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' "
                    f"xmlns:mstts='http://www.w3.org/2001/mstts' xml:lang='{creation_locale}'>"
                    f"<voice name='{self.personal_voice_model}'>"
                    f"<mstts:ttsembedding speakerProfileId='{self.speaker_profile_id}'>"
                    f"<lang xml:lang='{target}'>{escaped}</lang>"
                    f"</mstts:ttsembedding>"
                    f"</voice>"
                    f"</speak>"
                )
        else:
            # Bare personal-voice model (e.g. "DragonLatestNeural") without speaker profile
            if self.locale and str(self.locale).lower() == "auto":
                ssml = (
                    f"<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' "
                    f"xmlns:mstts='http://www.w3.org/2001/mstts' xml:lang='{creation_locale}'>"
                    f"<voice name='{self.personal_voice_model}'>"
                    f"{escaped}"
                    f"</voice>"
                    f"</speak>"
                )
            else:
                target = self.locale or creation_locale
                ssml = (
                    f"<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' "
                    f"xmlns:mstts='http://www.w3.org/2001/mstts' xml:lang='{creation_locale}'>"
                    f"<voice name='{self.personal_voice_model}'>"
                    f"<lang xml:lang='{target}'>{escaped}</lang>"
                    f"</voice>"
                    f"</speak>"
                )

        logger.info(
            f"KALLABOT: Personal-voice SSML — "
            f"Speaker: {self.speaker_profile_id}, Model: {self.personal_voice_model}, Locale: {self.locale}"
        )
        return ssml

    def _build_dragon_hd_ssml(self, text):  # KALLABOT
        """Build SSML for Dragon HD / multilingual / temperature-enabled voices.

        Supports:
        - Dragon HD and MultilingualNeural voice families
        - ``temperature`` attribute on the ``<voice>`` tag (Dragon voices)
        - ``<lang xml:lang="…">`` for locale / accent switching
        - ``locale="auto"`` for multilingual auto-detection
        """
        escaped = sax.escape(text)

        # Derive base locale from voice name (e.g. "en-US-Adam:DragonHDLatestNeural" → "en-US")
        if ':' in self.voice:
            base_voice = self.voice.split(':', 1)[0]
            voice_parts = base_voice.split('-')
        else:
            voice_parts = self.voice.split('-')

        base_locale = (
            f"{voice_parts[0]}-{voice_parts[1]}" if len(voice_parts) >= 2 else "en-US"
        )

        # Temperature attribute (Dragon voices only)
        has_temp = self.is_dragon_voice and self.temperature is not None
        temp_attr = f" parameters='temperature={self.temperature}'" if has_temp else ""

        # Build <voice> inner content with optional <lang> wrapper
        if self.is_multilingual_voice:
            if self.locale and str(self.locale).lower() == "auto":
                voice_inner = escaped
            else:
                target = self.locale or base_locale
                voice_inner = f"<lang xml:lang='{target}'>{escaped}</lang>"
        else:
            voice_inner = escaped

        ssml = (
            f"<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' "
            f"xmlns:mstts='http://www.w3.org/2001/mstts' xml:lang='{base_locale}'>"
            f"<voice name='{self.voice}'{temp_attr}>"
            f"{voice_inner}"
            f"</voice>"
            f"</speak>"
        )

        logger.info(
            f"KALLABOT: Dragon/HD SSML — Voice: {self.voice}, "
            f"Locale: {self.locale}, Base: {base_locale}, Temperature: {self.temperature}"
        )
        return ssml

    # ------------------------------------------------------------------ #
    # Standard helpers (unchanged except for _build_ssml delegation)
    # ------------------------------------------------------------------ #

    def get_synthesized_characters(self):
        return self.synthesized_characters

    def get_engine(self):
        return self.model

    def supports_websocket(self):
        return False

    def get_sleep_time(self):
        return 0.01

    async def synthesize(self, text):
        # For one-off synthesis without streaming
        audio = await self.__generate_http(text)
        return audio

    def _build_ssml(self, text: str):
        """Build SSML for synthesis.

        When Kallabot enhanced features are active (Personal Voice, Dragon HD,
        temperature, auto-locale) this delegates to the specialised builders.
        Otherwise the original prosody-only / plain-text path is preserved.
        """
        # KALLABOT: delegate to enhanced builders when Kallabot features are active
        if getattr(self, 'use_kallabot_ssml', False):
            if self.is_personal_voice:
                return self._build_personal_voice_ssml(text)
            else:
                return self._build_dragon_hd_ssml(text)

        # --- Original code path (speed / prosody only) ---
        body = sax.escape(text)
        # If nothing to tweak, return None to signal plain-text mode
        if not any([self.speed]) or self.speed == 1:
            return None

        prosody_attrs = []
        if self.speed is not None:
            prosody_attrs.append(f'rate="{self.speed}"')
        prosody_attr_str = " ".join(prosody_attrs)

        # NOTE: avoid <lang xml:lang> with <prosody>; docs state they are incompatible. :contentReference[oaicite:1]{index=1}
        return f'''<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="{self.language}">
          <voice name="{self.voice}">
            <prosody {prosody_attr_str}>{body}</prosody>
          </voice>
        </speak>'''


    async def __generate_http(self, text):
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=self.speech_config, audio_config=None)
        ssml = self._build_ssml(text)
        if ssml:
            result = synthesizer.speak_ssml_async(ssml).get()
        else:
            result = synthesizer.speak_text_async(text).get()

        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            return result.audio_data
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation = result.cancellation_details
            logger.error(f"Azure TTS canceled: {cancellation.reason}")
            if cancellation.reason == speechsdk.CancellationReason.Error:
                error_code = cancellation.error_code
                error_details = cancellation.error_details
                logger.error(f"Azure TTS error details: {error_details}")

                # Check specific error codes using SDK enums
                if error_code == CancellationErrorCode.AuthenticationFailure:
                    logger.error(f"Azure TTS authentication failed: Invalid subscription key - Region: {self.region}")
                    raise Exception(f"Azure TTS authentication failed: Invalid subscription key. Details: {error_details}")
                elif error_code == CancellationErrorCode.Forbidden:
                    logger.error(f"Azure TTS forbidden: Insufficient permissions or invalid region - Region: {self.region}")
                    raise Exception(f"Azure TTS forbidden: Insufficient permissions. Details: {error_details}")
                elif error_code == CancellationErrorCode.BadRequest:
                    logger.error(f"Azure TTS bad request: Invalid configuration - Region: {self.region}")
                    raise Exception(f"Azure TTS bad request: Invalid configuration. Details: {error_details}")
                elif error_code == CancellationErrorCode.ConnectionFailure:
                    logger.error(f"Azure TTS connection failure: Network issue - Region: {self.region}")
                    raise Exception(f"Azure TTS connection failure. Details: {error_details}")
                else:
                    logger.error(f"Azure TTS error (code: {error_code}): {error_details}")
                    raise Exception(f"Azure TTS error: {error_details}")
            return None
        else:
            logger.error(f"Azure TTS synthesis failed with reason: {result.reason}")
            return None

    async def generate(self):
        try:
            while True:
                message = await self.internal_queue.get()
                logger.info(f"Generating TTS response for message: {message}")
                meta_info, text = message.get("meta_info"), message.get("data")

                if not self.should_synthesize_response(meta_info.get('sequence_id')):
                    logger.info(f"Not synthesizing text as the sequence_id ({meta_info.get('sequence_id')}) of it is not in the list of sequence_ids present in the task manager.")
                    return

                # Check cache if enabled
                if self.caching and self.cache.get(text):
                    logger.info(f"Cache hit and hence returning quickly {text}")
                    audio_data = self.cache.get(text)

                    # Set metadata and yield the cached audio
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
                    yield create_ws_data_packet(audio_data, meta_info)
                    continue

                # Create synthesizer for each request to avoid blocking
                try:
                    synthesizer = speechsdk.SpeechSynthesizer(speech_config=self.speech_config, audio_config=None)
                except Exception as e:
                    logger.error(f"Failed to create Azure TTS synthesizer: {e}")
                    # SpeechSynthesizer creation typically fails due to config issues
                    logger.error(f"Check subscription key and region configuration - Region: {self.region}")
                    continue

                # Set up streaming events
                chunk_queue = asyncio.Queue()
                done_event = asyncio.Event()
                synthesis_error = None
                start_time = time.perf_counter()

                def speech_synthesizer_synthesizing_handler(evt):
                    try:
                        if self.connection_time is None:
                            self.connection_time = round((time.perf_counter() - start_time) * 1000)

                        # Use run_coroutine_threadsafe to safely put data from another thread
                        asyncio.run_coroutine_threadsafe(
                            chunk_queue.put(evt.result.audio_data),
                            self.loop
                        )
                    except Exception as e:
                        logger.error(f"Error in synthesizing handler: {e}")

                def speech_synthesizer_completed_handler(evt):
                    nonlocal synthesis_error
                    async def set_done_event():
                        done_event.set()

                    # Check if synthesis was canceled due to error
                    if evt.result.reason == speechsdk.ResultReason.Canceled:
                        cancellation = evt.result.cancellation_details
                        if cancellation.reason == speechsdk.CancellationReason.Error:
                            error_code = cancellation.error_code
                            error_details = cancellation.error_details
                            synthesis_error = error_details
                            logger.error(f"Azure TTS synthesis canceled with error: {error_details}")

                            # Check specific error codes using SDK enums
                            if error_code == CancellationErrorCode.AuthenticationFailure:
                                logger.error(f"Azure TTS authentication failed: Invalid subscription key - Region: {self.region}")
                            elif error_code == CancellationErrorCode.Forbidden:
                                logger.error(f"Azure TTS forbidden: Insufficient permissions - Region: {self.region}")
                            elif error_code == CancellationErrorCode.BadRequest:
                                logger.error(f"Azure TTS bad request: Invalid configuration - Region: {self.region}")
                            elif error_code == CancellationErrorCode.ConnectionFailure:
                                logger.error(f"Azure TTS connection failure: Network issue - Region: {self.region}")

                    asyncio.run_coroutine_threadsafe(set_done_event(), self.loop)

                synthesizer.synthesizing.connect(speech_synthesizer_synthesizing_handler)
                synthesizer.synthesis_completed.connect(speech_synthesizer_completed_handler)

                # Stamp synthesizer turn start time
                try:
                    meta_info['synthesizer_start_time'] = time.perf_counter()
                except Exception:
                    pass

                ssml = self._build_ssml(text)
                start_time = time.perf_counter()
                try:
                    if ssml:
                        synthesizer.speak_ssml_async(ssml)
                    else:
                        synthesizer.speak_text_async(text)
                except Exception as e:
                    logger.error(f"Failed to start Azure TTS synthesis: {e}")
                    # Synthesis start failures will be caught by the completed_handler
                    continue
                logger.info(f"Azure TTS request sent for {len(text)} chars")
                full_audio = bytearray()

                # Process chunks as they arrive
                while not done_event.is_set() or not chunk_queue.empty():
                    try:
                        # Get available chunk or wait briefly
                        chunk = await asyncio.wait_for(chunk_queue.get(), timeout=0.01)

                        # Collect full audio for caching if enabled
                        if self.caching:
                            full_audio.extend(chunk)

                        # Log first chunk latency
                        if not self.first_chunk_generated:
                            first_chunk_time = round((time.perf_counter() - start_time) * 1000)
                            self.latency_stats["request_count"] += 1
                            self.latency_stats["total_first_byte_latency"] += first_chunk_time
                            self.latency_stats["min_latency"] = min(self.latency_stats["min_latency"], first_chunk_time)
                            self.latency_stats["max_latency"] = max(self.latency_stats["max_latency"], first_chunk_time)
                            # Expose first-result latency via meta_info
                            try:
                                if 'synthesizer_first_result_latency' not in meta_info:
                                    meta_info['synthesizer_first_result_latency'] = (time.perf_counter() - meta_info.get('synthesizer_start_time', start_time))
                                    meta_info['synthesizer_latency'] = meta_info['synthesizer_first_result_latency']
                            except Exception:
                                pass

                        # Process chunk
                        if not self.first_chunk_generated:
                            meta_info["is_first_chunk"] = True
                            self.first_chunk_generated = True
                        else:
                            meta_info["is_first_chunk"] = False

                        # Track if this is the end
                        if done_event.is_set() and chunk_queue.empty():
                            if "end_of_llm_stream" in meta_info and meta_info["end_of_llm_stream"]:
                                meta_info["end_of_synthesizer_stream"] = True
                                self.first_chunk_generated = False

                        meta_info['text'] = text
                        meta_info['format'] = 'wav'
                        meta_info["text_synthesized"] = f"{text} "
                        meta_info["mark_id"] = str(uuid.uuid4())
                        yield create_ws_data_packet(chunk, meta_info)

                    except asyncio.TimeoutError:
                        # No chunk ready, just continue and check done_event again
                        continue

                # Cache the complete audio if enabled
                if self.caching and full_audio:
                    logger.info(f"Caching audio for text: {text}")
                    self.cache.set(text, bytes(full_audio))

                self.synthesized_characters += len(text)
                # Compute total stream duration
                try:
                    meta_info['synthesizer_total_stream_duration'] = (time.perf_counter() - meta_info.get('synthesizer_start_time', start_time))
                except Exception:
                    pass
        except asyncio.CancelledError:
            logger.info("Azure synthesizer task was cancelled - shutting down cleanly")
            raise
        except Exception as e:
            logger.error(f"Error in Azure TTS generate method: {e}")
            raise

    async def open_connection(self):
        pass

    async def push(self, message):
        logger.info(f"Pushed message to internal queue {message}")
        self.internal_queue.put_nowait(message)
