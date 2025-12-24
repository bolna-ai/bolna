import os
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
