import os
import time
import asyncio
import threading
import queue
import json
from dotenv import load_dotenv

from google.cloud import speech_v1p1beta1 as speech

from .base_transcriber import BaseTranscriber
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.utils import create_ws_data_packet

load_dotenv()
logger = configure_logger(__name__)


class GoogleTranscriber(BaseTranscriber):
    """
    Streaming transcriber using Google Cloud Speech-to-Text.
    Uses threading to bridge async input_queue with blocking gRPC streaming.
    """

    def __init__(self,
                 telephony_provider,
                 input_queue=None,
                 output_queue=None,
                 language="en-US",
                 encoding=None,
                 sample_rate_hertz=None,
                 model="latest_long",
                 run_id="",
                 **kwargs):
        super().__init__(input_queue)
        self.provider = telephony_provider or ""
        self.transcriber_output_queue = output_queue  # expected to be asyncio.Queue in TaskManager
        self.language = language
        self.model = model
        self.run_id = run_id or kwargs.get("run_id", "")

        # Provider-specific audio configuration
        if self.provider in ('twilio', 'exotel', 'plivo'):
            self.encoding = "MULAW" if self.provider in ("twilio") else "LINEAR16"
            self.sample_rate_hertz = 8000  
        elif self.provider == "web_based_call":
            self.encoding = "LINEAR16"
            self.sample_rate_hertz = 16000
        elif self.provider == "playground":
            self.encoding = "LINEAR16"
            self.sample_rate_hertz = 8000
        else:
            self.encoding = encoding or "LINEAR16"
            self.sample_rate_hertz = sample_rate_hertz or 16000

        # Google client using Application Default Credentials
        self.client = speech.SpeechClient()

        # Threading bridge for gRPC streaming
        self._audio_q = queue.Queue()
        self._running = False
        self._grpc_thread = None
        
        # Connection state management
        self.connection_start_time = None
        self.connection_time = None
        self.websocket_connection = None
        self.connection_authenticated = False
        self.transcription_task = None
        
        # Audio frame tracking
        self.audio_frame_duration = 0.0
        self.num_frames = 0
        if self.provider in ('twilio', 'exotel', 'plivo'):
            self.audio_frame_duration = 0.2
        elif self.provider == "web_based_call":
            self.audio_frame_duration = 0.256
        elif self.provider == "playground":
            self.audio_frame_duration = 0.0

        # Turn latency tracking
        self.turn_latencies = []
        self.current_turn_start_time = None
        self.current_turn_id = None

        # Request tracking
        self.meta_info = None
        self._request_id = None
        self.audio_submitted = False
        self.audio_submission_time = None

        # Event loop reference for thread-safe queue operations
        try:
            self.loop = asyncio.get_event_loop()
        except Exception:
            self.loop = None
    def _enqueue_output(self, data, meta=None):
        """Thread-safe enqueue to transcriber_output_queue."""
        if self.transcriber_output_queue is None:
            return

        packet = create_ws_data_packet(data, meta or self.meta_info or {})

        # Thread-safe asyncio queue operation
        try:
            if self.loop and isinstance(self.transcriber_output_queue, asyncio.Queue):
                future = asyncio.run_coroutine_threadsafe(self.transcriber_output_queue.put(packet), self.loop)
                try:
                    future.result(timeout=2.0)
                except Exception:
                    pass
                return
        except Exception:
            pass

        # Fallback to sync operations
        try:
            if hasattr(self.transcriber_output_queue, "put_nowait"):
                self.transcriber_output_queue.put_nowait(packet)
            elif hasattr(self.transcriber_output_queue, "put"):
                self.transcriber_output_queue.put(packet)
        except Exception:
            logger.exception("Failed to enqueue packet to transcriber_output_queue")

    async def google_connect(self):
        """Validate Google Speech client connection."""
        try:
            start_time = time.perf_counter()
            _ = self.client
            self.connection_authenticated = True
            
            if not self.connection_time:
                self.connection_time = round((time.perf_counter() - start_time) * 1000)
            
            logger.info("Successfully validated Google Speech client")
            return True
            
        except Exception as e:
            logger.error(f"Failed to validate Google Speech client: {e}")
            raise ConnectionError(f"Failed to validate Google Speech client: {e}")
    
    async def run(self):
        """
        Enhanced startup sequence matching Deepgram pattern.
        """
        try:
            # Connection validation
            await self.google_connect()
            
            self._running = True
            
            # Create transcription task like Deepgram
            self.transcription_task = asyncio.create_task(self._transcribe_wrapper())
            
        except Exception as e:
            logger.exception(f"Error starting GoogleTranscriber: {e}")
            await self.toggle_connection()
    
    async def _transcribe_wrapper(self):
        """Wrapper to make gRPC streaming fit async pattern better"""
        try:
            # spawn the blocking gRPC consumer in a background thread
            self._grpc_thread = threading.Thread(target=self._run_grpc_stream, daemon=True)
            self._grpc_thread.start()

            # spawn async sender to read from input_queue
            await self._send_audio_to_transcriber()
            
        except Exception as e:
            logger.error(f"Error in transcription wrapper: {e}")
            await self.toggle_connection()
        finally:
            # Ensure cleanup
            if hasattr(self, 'transcription_task') and self.transcription_task:
                try:
                    self.transcription_task.cancel()
                except Exception:
                    pass

    async def _send_audio_to_transcriber(self):
        """Reads packets from input_queue and forwards to gRPC thread via _audio_q."""
        try:
            while True:
                ws_data_packet = await self.input_queue.get()

                # Initialize metadata on first audio packet
                if not self.audio_submitted:
                    self.audio_submitted = True
                    self.audio_submission_time = time.time()
                    self._request_id = self.generate_request_id()
                    self.meta_info = ws_data_packet.get('meta_info', {}) or {}
                    self.meta_info['request_id'] = self._request_id
                    try:
                        self.meta_info['transcriber_start_time'] = time.perf_counter()
                        # start turn-level tracking
                        self.current_turn_start_time = self.meta_info['transcriber_start_time']
                        self.current_turn_id = self.meta_info.get('turn_id') or self.meta_info.get('request_id') or self._request_id
                    except Exception:
                        pass

                # check EOS
                if ws_data_packet.get('meta_info', {}).get('eos') is True:
                    # put sentinel so blocking generator ends gracefully
                    self._audio_q.put(None)
                    break

                # get raw audio bytes from packet and track frames
                data = ws_data_packet.get('data')
                if data:
                    self.num_frames += 1
                    # if data is base64 string, try to decode if needed
                    if isinstance(data, str):
                        try:
                            # if base64 encoded, decode (guardy)
                            import base64 as _b64
                            d = _b64.b64decode(data)
                            self._audio_q.put(d)
                        except Exception:
                            # fallback: push raw str bytes
                            self._audio_q.put(data.encode('utf-8'))
                    else:
                        # assume bytes-like
                        self._audio_q.put(data)
        except Exception:
            logger.exception("Error in _send_audio_to_transcriber")

    def _audio_generator(self):
        """
        Blocking generator consumed by google client.streaming_recognize.
        Yields StreamingRecognizeRequest(audio_content=...).
        """
        while True:
            chunk = self._audio_q.get()
            if chunk is None:
                # sentinel: end of stream
                return
            # ensure bytes
            if isinstance(chunk, bytes):
                yield speech.StreamingRecognizeRequest(audio_content=chunk)
            else:
                # try to coerce
                try:
                    yield speech.StreamingRecognizeRequest(audio_content=bytes(chunk))
                except Exception:
                    logger.exception("Non-bytes chunk received in google audio generator; dropping")

    def _append_turn_latency(self):
        """
        Include timing data in meta_info for TaskManager to process.
        Called when a final transcript arrives (end-of-turn semantics).
        """
        try:
            if self.current_turn_start_time:
                first_ms = int(round((self.meta_info.get('transcriber_first_result_latency', 0)) * 1000))
                total_s = (time.perf_counter() - self.current_turn_start_time) if self.current_turn_start_time else 0
                # Include timing data in meta_info for TaskManager to process
                self.meta_info['transcriber_turn_metrics'] = {
                    'first_result_latency_ms': first_ms,
                    'total_stream_duration_ms': int(round(total_s * 1000))
                }
                # reset turn tracking
                self.current_turn_start_time = None
                self.current_turn_id = None
        except Exception:
            logger.exception("Error adding turn metrics")

    def _run_grpc_stream(self):
        """
        Enhanced gRPC streaming with better error handling.
        Blocking thread target that runs google streaming_recognize and iterates responses.
        Pushes interim and final transcripts back onto transcriber_output_queue (thread-safely).
        """
        try:
            # Connection establishment timing
            if not self.connection_start_time:
                self.connection_start_time = time.time()
            # build recognition config
            # map string encodings to enum (google cloud)
            encoding_enum = speech.RecognitionConfig.AudioEncoding.LINEAR16
            enc = (self.encoding or "").upper()
            if 'MULAW' in enc or 'ULAW' in enc:
                encoding_enum = speech.RecognitionConfig.AudioEncoding.MULAW
            elif 'LINEAR' in enc or 'PCM' in enc:
                encoding_enum = speech.RecognitionConfig.AudioEncoding.LINEAR16
            else:
                # fallback: leave as unspecified (google will attempt sampling inference)
                try:
                    encoding_enum = speech.RecognitionConfig.AudioEncoding.ENCODING_UNSPECIFIED
                except Exception:
                    encoding_enum = speech.RecognitionConfig.AudioEncoding.LINEAR16

            recognition_config = speech.RecognitionConfig(
                encoding=encoding_enum,
                sample_rate_hertz=int(self.sample_rate_hertz),
                language_code=self.language,
                model=self.model,
                enable_automatic_punctuation=True,
                max_alternatives=1,
            )

            streaming_config = speech.StreamingRecognitionConfig(
                config=recognition_config,
                interim_results=True,
                single_utterance=False,
            )

            requests = self._audio_generator()
            
            try:
                responses = self.client.streaming_recognize(streaming_config, requests)
                self.connection_authenticated = True
                
                # iterate responses synchronously
                for response in responses:
                    if not self._running:
                        break
                    if not response.results:
                        continue
                    
                    result = response.results[0]
                    is_final = result.is_final
                    transcript = ""
                    if result.alternatives:
                        transcript = result.alternatives[0].transcript.strip()

                    if transcript:
                        # set first-result latency if not already set
                        try:
                            if self.meta_info and 'transcriber_start_time' in self.meta_info and 'transcriber_first_result_latency' not in self.meta_info:
                                self.meta_info['transcriber_first_result_latency'] = time.perf_counter() - self.meta_info['transcriber_start_time']
                        except Exception:
                            pass

                        # Prepare packet
                        if is_final:
                            # populate total durations and append turn latencies
                            try:
                                if self.meta_info and 'transcriber_start_time' in self.meta_info:
                                    self.meta_info['transcriber_total_stream_duration'] = time.perf_counter() - self.meta_info['transcriber_start_time']
                            except Exception:
                                pass

                            # append to turn_latencies (A)
                            self._append_turn_latency()

                            data = {"type": "transcript", "content": transcript}
                            self._enqueue_output(data, meta=self.meta_info)
                        else:
                            data = {"type": "interim_transcript_received", "content": transcript}
                            self._enqueue_output(data, meta=self.meta_info)

                # After streaming ends on Google side, send transcriber_connection_closed sentinel
                closed_meta = (self.meta_info or {}).copy()
                if 'transcriber_total_stream_duration' not in closed_meta and 'transcriber_start_time' in closed_meta:
                    try:
                        closed_meta['transcriber_total_stream_duration'] = time.perf_counter() - closed_meta['transcriber_start_time']
                    except Exception:
                        pass
                self._enqueue_output("transcriber_connection_closed", meta=closed_meta)

            except Exception as stream_error:
                # Specific gRPC error handling
                error_msg = f"Google streaming error: {stream_error}"
                logger.error(error_msg)
                
                # Determine if error is retryable
                if "deadline exceeded" in str(stream_error).lower():
                    logger.info("Deadline exceeded - this may be normal for long streams")
                elif "unavailable" in str(stream_error).lower():
                    logger.warning("Service unavailable - connection issue")
                
                # Send error to output
                err_meta = (self.meta_info or {}).copy()
                err_meta['error'] = str(stream_error)
                err_meta['error_type'] = 'streaming_error'
                self._enqueue_output("transcriber_connection_closed", meta=err_meta)
                return
                
        except Exception as e:
            # Configuration or setup error
            logger.exception(f"Google transcriber setup error: {e}")
            err_meta = (self.meta_info or {}).copy()
            err_meta['error'] = str(e)
            err_meta['error_type'] = 'setup_error'
            self._enqueue_output("transcriber_connection_closed", meta=err_meta)
        finally:
            self.connection_authenticated = False

    async def toggle_connection(self):
        """
        Called by TaskManager to force-close connection.
        Enhanced cleanup matching Deepgram pattern.
        """
        logger.info("toggle_connection called on GoogleTranscriber")
        self._running = False
        self.connection_authenticated = False
        
        # Cancel transcription task if running
        if hasattr(self, 'transcription_task') and self.transcription_task:
            try:
                self.transcription_task.cancel()
            except Exception:
                pass
        
        # Signal thread to stop
        try:
            self._audio_q.put(None)
        except Exception:
            pass
        
        # Wait for thread cleanup with timeout
        if hasattr(self, '_grpc_thread') and self._grpc_thread and self._grpc_thread.is_alive():
            try:
                self._grpc_thread.join(timeout=2.0)
                if self._grpc_thread.is_alive():
                    logger.warning("gRPC thread did not terminate within timeout")
            except Exception as e:
                logger.error(f"Error joining gRPC thread: {e}")
        
        logger.info("GoogleTranscriber connection toggled off")

    def cleanup(self):
        """
        Enhanced graceful shutdown matching Deepgram pattern.
        """
        try:
            self._running = False
            self.connection_authenticated = False
            
            # Signal thread to stop
            self._audio_q.put(None)
            
            # Wait for thread with timeout
            if self._grpc_thread and self._grpc_thread.is_alive():
                self._grpc_thread.join(timeout=2.0)
                if self._grpc_thread.is_alive():
                    logger.warning("gRPC thread did not terminate gracefully")
            
            self._grpc_thread = None
            
            # Reset connection state
            self.connection_start_time = None
            self.connection_time = None
            
            logger.info("GoogleTranscriber cleanup completed")
            
        except Exception:
            logger.exception("cleanup error in GoogleTranscriber")

    def get_meta_info(self):
        return self.meta_info
