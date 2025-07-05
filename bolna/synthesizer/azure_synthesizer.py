import os
import uuid
import asyncio
import time
from dotenv import load_dotenv
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.utils import create_ws_data_packet
from bolna.memory.cache.inmemory_scalar_cache import InmemoryScalarCache
from .base_synthesizer import BaseSynthesizer
import azure.cognitiveservices.speech as speechsdk

logger = configure_logger(__name__)
load_dotenv()


class AzureSynthesizer(BaseSynthesizer):
    def __init__(self, voice, language, model="neural", stream=False, sampling_rate=8000, buffer_size=150, caching=True, **kwargs):
        super().__init__(kwargs.get("task_manager_instance", None), stream, buffer_size)
        self.model = model
        self.language = language
        self.voice = f"{language}-{voice}{model}"
        logger.debug(f"{self.voice} initialized")
        self.sample_rate = str(sampling_rate)
        self.stream = stream
        self.caching = caching
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

    async def synthesize(self, text):
        # For one-off synthesis without streaming
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
        try:
            while True:
                message = await self.internal_queue.get()
                logger.debug(f"Generating TTS response for message: {message}")
                meta_info, text = message.get("meta_info"), message.get("data")

                if not self.should_synthesize_response(meta_info.get('sequence_id')):
                    logger.debug(f"Not synthesizing text as the sequence_id ({meta_info.get('sequence_id')}) of it is not in the list of sequence_ids present in the task manager.")
                    return

                # Check cache if enabled
                if self.caching and self.cache.get(text):
                    logger.debug(f"Cache hit and hence returning quickly {text}")
                    audio_data = self.cache.get(text)
                    
                    # Set metadata and yield the cached audio
                    self.set_first_chunk_metadata(meta_info)
                    self.set_end_of_stream_metadata(meta_info)
                    
                    meta_info['text'] = text
                    meta_info['format'] = 'wav'
                    yield self.create_audio_packet(audio_data, meta_info, f"{text} ")
                    continue

                # Create synthesizer for each request to avoid blocking
                synthesizer = speechsdk.SpeechSynthesizer(speech_config=self.speech_config, audio_config=None)

                # Set up streaming events
                chunk_queue = asyncio.Queue()
                done_event = asyncio.Event()
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
                    async def set_done_event():
                        done_event.set()

                    asyncio.run_coroutine_threadsafe(set_done_event(), self.loop)

                synthesizer.synthesizing.connect(speech_synthesizer_synthesizing_handler)
                synthesizer.synthesis_completed.connect(speech_synthesizer_completed_handler)
                
                # Start the synthesis (non-blocking)
                start_time = time.perf_counter()
                synthesizer.speak_text_async(text)
                logger.debug(f"Azure TTS request sent for {len(text)} chars")
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

                        # Process chunk
                        self.set_first_chunk_metadata(meta_info)
                        
                        # Track if this is the end
                        if done_event.is_set() and chunk_queue.empty():
                            self.set_end_of_stream_metadata(meta_info)
                        
                        meta_info['text'] = text
                        meta_info['format'] = 'wav'
                        yield self.create_audio_packet(chunk, meta_info, f"{text} ")
                        
                    except asyncio.TimeoutError:
                        # No chunk ready, just continue and check done_event again
                        continue
                
                # Cache the complete audio if enabled
                if self.caching and full_audio:
                    logger.debug(f"Caching audio for text: {text}")
                    self.cache.set(text, bytes(full_audio))
                
                self.synthesized_characters += len(text)
        except asyncio.CancelledError:
            logger.debug("Azure synthesizer task was cancelled - shutting down cleanly")
            raise
        except Exception as e:
            logger.error(f"Error in Azure TTS generate method: {e}")
            raise

    async def open_connection(self):
        pass
