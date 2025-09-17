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

logger = configure_logger(__name__)
load_dotenv()


class AzureSynthesizer(BaseSynthesizer):
    def __init__(self, voice, language, model="neural", stream=False, sampling_rate=8000, buffer_size=150, caching=True, speed=None, **kwargs):
        super().__init__(kwargs.get("task_manager_instance", None), stream, buffer_size)
        self.model = model
        self.language = language
        self.voice = f"{language}-{voice}{model}"
        logger.debug(f"{self.voice} initialized")
        self.sample_rate = str(sampling_rate)
        self.first_chunk_generated = False
        self.stream = stream
        self.synthesized_characters = 0
        self.caching = caching
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

        # Implement connection pooling with actual synthesizer management
        self.synthesizer_pool = []
        self.max_pool_size = 5  # Tune based on your traffic
        self.pool_lock = asyncio.Lock()
        self._initialize_pool()

    def _initialize_pool(self):
        """Initialize the synthesizer pool with pre-created instances"""
        for _ in range(min(2, self.max_pool_size)):  # Start with 2 instances
            synthesizer = speechsdk.SpeechSynthesizer(speech_config=self.speech_config, audio_config=None)
            self.synthesizer_pool.append(synthesizer)

    async def get_synthesizer(self):
        """Get a synthesizer from the pool or create a new one"""
        async with self.pool_lock:
            if self.synthesizer_pool:
                return self.synthesizer_pool.pop()
            else:
                logger.debug("Creating new synthesizer instance (pool empty)")
                return speechsdk.SpeechSynthesizer(speech_config=self.speech_config, audio_config=None)

    async def return_synthesizer(self, synthesizer):
        """Return a synthesizer to the pool for reuse"""
        async with self.pool_lock:
            if len(self.synthesizer_pool) < self.max_pool_size:
                self.synthesizer_pool.append(synthesizer)
            else:
                pass

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

                synthesizer = await self.get_synthesizer()

                try:
                    # Set up streaming events
                    chunk_queue = asyncio.Queue()
                    done_event = asyncio.Event()
                    start_time = time.perf_counter()
                    full_audio = bytearray() if self.caching else None

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

                    ssml = self._build_ssml(text)
                    start_time = time.perf_counter()
                    if ssml:
                        synthesizer.speak_ssml_async(ssml)
                    else:
                        synthesizer.speak_text_async(text)
                    logger.debug(f"Azure TTS request sent for {len(text)} chars")
                    
                    # Process chunks as they arrive - optimized for streaming
                    while not done_event.is_set() or not chunk_queue.empty():
                        try:
                            if not chunk_queue.empty():
                                chunk = chunk_queue.get_nowait()
                            elif not done_event.is_set():
                                chunk = await chunk_queue.get()
                            else:
                                break
                            
                            # Collect full audio for caching only if enabled and not streaming
                            if self.caching and full_audio is not None:
                                full_audio.extend(chunk)
                            
                            # Log first chunk latency
                            if not self.first_chunk_generated:
                                first_chunk_time = round((time.perf_counter() - start_time) * 1000)
                                self.latency_stats["request_count"] += 1
                                self.latency_stats["total_first_byte_latency"] += first_chunk_time
                                self.latency_stats["min_latency"] = min(self.latency_stats["min_latency"], first_chunk_time)
                                self.latency_stats["max_latency"] = max(self.latency_stats["max_latency"], first_chunk_time)

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
                            
                        except asyncio.QueueEmpty:
                            if done_event.is_set():
                                break
                            await asyncio.sleep(0)
                    
                    # Cache the complete audio if enabled
                    if self.caching and full_audio:
                        logger.debug(f"Caching audio for text: {text}")
                        self.cache.set(text, bytes(full_audio))
                
                finally:
                    await self.return_synthesizer(synthesizer)
                
                self.synthesized_characters += len(text)
        except asyncio.CancelledError:
            logger.debug("Azure synthesizer task was cancelled - shutting down cleanly")
            raise
        except Exception as e:
            logger.error(f"Error in Azure TTS generate method: {e}")
            raise

    async def open_connection(self):
        """Initialize connection - Azure SDK handles connections internally"""
        pass

    async def cleanup(self):
        """Clean up synthesizer pool and resources"""
        async with self.pool_lock:
            self.synthesizer_pool.clear()
        logger.debug("Azure synthesizer pool cleaned up")

    async def monitor_connection(self):
        """Monitor connection health - Azure SDK handles reconnection internally"""
        pass

    async def push(self, message):
        logger.debug(f"Pushed message to internal queue {message}")
        self.internal_queue.put_nowait(message)
