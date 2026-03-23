import os
import asyncio
import time
import xml.sax.saxutils as sax

from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk
from azure.cognitiveservices.speech import CancellationErrorCode

from .base_synthesizer import BaseSynthesizer
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.utils import create_ws_data_packet
from bolna.memory.cache.inmemory_scalar_cache import InmemoryScalarCache

logger = configure_logger(__name__)
load_dotenv()


class AzureSynthesizer(BaseSynthesizer):
    def __init__(self, voice, language, model="neural", stream=False, sampling_rate=8000,
                 buffer_size=150, caching=True, speed=None, **kwargs):
        super().__init__(kwargs.get("task_manager_instance"), stream, buffer_size)
        self.model = model
        self.language = language
        self.voice = f"{language}-{voice}{model}"
        logger.info(f"{self.voice} initialized")
        self.sample_rate = str(sampling_rate)
        self.stream = stream
        self.caching = False
        self.speed = speed
        if caching:
            self.cache = InmemoryScalarCache()
        self.loop = asyncio.get_event_loop()

        self.subscription_key = kwargs.get("synthesizer_key", os.getenv("AZURE_SPEECH_KEY"))
        self.region = kwargs.get("region", os.getenv("AZURE_SPEECH_REGION"))
        self.speech_config = speechsdk.SpeechConfig(subscription=self.subscription_key, region=self.region)
        self.speech_config.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Raw8Khz16BitMonoPcm)
        self.speech_config.speech_synthesis_voice_name = self.voice

        self.latency_stats = {
            "request_count": 0, "total_first_byte_latency": 0,
            "min_latency": float("inf"), "max_latency": 0.0,
        }
        self.connection_requested_at = None

    def supports_websocket(self):
        return False

    def get_sleep_time(self):
        return 0.01

    # ------------------------------------------------------------------
    # SSML
    # ------------------------------------------------------------------

    def _build_ssml(self, text):
        body = sax.escape(text)
        if not any([self.speed]) or self.speed == 1:
            return None
        prosody_attrs = []
        if self.speed is not None:
            prosody_attrs.append(f'rate="{self.speed}"')
        prosody_attr_str = " ".join(prosody_attrs)
        return (
            f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="{self.language}">'
            f'<voice name="{self.voice}">'
            f"<prosody {prosody_attr_str}>{body}</prosody>"
            f"</voice></speak>"
        )

    # ------------------------------------------------------------------
    # HTTP synthesis (blocking SDK call)
    # ------------------------------------------------------------------

    async def _generate_http(self, text):
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
                self._log_cancellation_error(cancellation, raise_exception=True)
            return None
        else:
            logger.error(f"Azure TTS synthesis failed with reason: {result.reason}")
            return None

    def _log_cancellation_error(self, cancellation, raise_exception=False):
        error_code = cancellation.error_code
        error_details = cancellation.error_details
        logger.error(f"Azure TTS error details: {error_details}")
        error_map = {
            CancellationErrorCode.AuthenticationFailure: "authentication failed: Invalid subscription key",
            CancellationErrorCode.Forbidden: "forbidden: Insufficient permissions or invalid region",
            CancellationErrorCode.BadRequest: "bad request: Invalid configuration",
            CancellationErrorCode.ConnectionFailure: "connection failure: Network issue",
        }
        msg = error_map.get(error_code, f"error (code: {error_code})")
        logger.error(f"Azure TTS {msg} - Region: {self.region}")
        if raise_exception:
            raise Exception(f"Azure TTS {msg}. Details: {error_details}")

    async def synthesize(self, text):
        return await self._generate_http(text)

    # ------------------------------------------------------------------
    # generate / push
    # ------------------------------------------------------------------

    async def generate(self):
        try:
            while True:
                message = await self.internal_queue.get()
                logger.info(f"Generating TTS response for message: {message}")
                meta_info, text = message.get("meta_info"), message.get("data")

                if not self.should_synthesize_response(meta_info.get("sequence_id")):
                    logger.info(f"Not synthesizing: sequence_id {meta_info.get('sequence_id')} not current")
                    return

                # Cache path
                if self.caching and self.cache.get(text):
                    logger.info(f"Cache hit: {text}")
                    audio_data = self.cache.get(text)
                    self._stamp_first_chunk(meta_info)
                    self._stamp_end_of_stream(meta_info)
                    meta_info["text"] = text
                    meta_info["format"] = "wav"
                    meta_info["text_synthesized"] = f"{text} "
                    self._stamp_mark_id(meta_info)
                    yield create_ws_data_packet(audio_data, meta_info)
                    continue

                # Streaming via Azure SDK callbacks
                try:
                    synthesizer = speechsdk.SpeechSynthesizer(speech_config=self.speech_config, audio_config=None)
                except Exception as e:
                    logger.error(f"Failed to create Azure TTS synthesizer: {e}")
                    logger.error(f"Check subscription key and region configuration - Region: {self.region}")
                    continue

                chunk_queue = asyncio.Queue()
                done_event = asyncio.Event()
                start_time = time.perf_counter()

                try:
                    meta_info["synthesizer_start_time"] = time.perf_counter()
                except Exception:
                    pass

                def on_synthesizing(evt):
                    try:
                        if self.connection_time is None:
                            self.connection_time = round((time.perf_counter() - start_time) * 1000)
                        asyncio.run_coroutine_threadsafe(chunk_queue.put(evt.result.audio_data), self.loop)
                    except Exception as e:
                        logger.error(f"Error in synthesizing handler: {e}")

                def on_completed(evt):
                    if evt.result.reason == speechsdk.ResultReason.Canceled:
                        cancellation = evt.result.cancellation_details
                        if cancellation.reason == speechsdk.CancellationReason.Error:
                            self._log_cancellation_error(cancellation)
                    asyncio.run_coroutine_threadsafe(_set_done(), self.loop)

                async def _set_done():
                    done_event.set()

                synthesizer.synthesizing.connect(on_synthesizing)
                synthesizer.synthesis_completed.connect(on_completed)

                ssml = self._build_ssml(text)
                start_time = time.perf_counter()
                try:
                    if ssml:
                        synthesizer.speak_ssml_async(ssml)
                    else:
                        synthesizer.speak_text_async(text)
                except Exception as e:
                    logger.error(f"Failed to start Azure TTS synthesis: {e}")
                    continue

                logger.info(f"Azure TTS request sent for {len(text)} chars")
                full_audio = bytearray()

                while not done_event.is_set() or not chunk_queue.empty():
                    try:
                        chunk = await asyncio.wait_for(chunk_queue.get(), timeout=0.01)

                        if self.caching:
                            full_audio.extend(chunk)

                        if not self.first_chunk_generated:
                            first_chunk_time = round((time.perf_counter() - start_time) * 1000)
                            self.latency_stats["request_count"] += 1
                            self.latency_stats["total_first_byte_latency"] += first_chunk_time
                            self.latency_stats["min_latency"] = min(self.latency_stats["min_latency"], first_chunk_time)
                            self.latency_stats["max_latency"] = max(self.latency_stats["max_latency"], first_chunk_time)
                            try:
                                if "synthesizer_first_result_latency" not in meta_info:
                                    meta_info["synthesizer_first_result_latency"] = time.perf_counter() - meta_info.get("synthesizer_start_time", start_time)
                                    meta_info["synthesizer_latency"] = meta_info["synthesizer_first_result_latency"]
                            except Exception:
                                pass

                        self._stamp_first_chunk(meta_info)

                        if done_event.is_set() and chunk_queue.empty():
                            self._stamp_end_of_stream(meta_info)

                        meta_info["text"] = text
                        meta_info["format"] = "wav"
                        meta_info["text_synthesized"] = f"{text} "
                        self._stamp_mark_id(meta_info)
                        yield create_ws_data_packet(chunk, meta_info)

                    except asyncio.TimeoutError:
                        continue

                if self.caching and full_audio:
                    logger.info(f"Caching audio for text: {text}")
                    self.cache.set(text, bytes(full_audio))

                self.synthesized_characters += len(text)
                try:
                    meta_info["synthesizer_total_stream_duration"] = time.perf_counter() - meta_info.get("synthesizer_start_time", start_time)
                except Exception:
                    pass

        except asyncio.CancelledError:
            logger.info("Azure synthesizer task was cancelled - shutting down cleanly")
            raise
        except Exception as e:
            logger.error(f"Error in Azure TTS generate method: {e}")
            raise
