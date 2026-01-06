import asyncio
import base64
import json
import os
import time
import traceback
import audioop
from typing import Optional
from dotenv import load_dotenv

import websockets
from websockets.asyncio.client import ClientConnection
from websockets.exceptions import ConnectionClosedError, InvalidHandshake, ConnectionClosed
import numpy as np
from scipy.signal import resample_poly

from .base_transcriber import BaseTranscriber
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.utils import create_ws_data_packet, timestamp_ms

load_dotenv()
logger = configure_logger(__name__)


class ShunyaTranscriber(BaseTranscriber):
    """
    API Documentation: https://www.shunyalabs.ai/documentation/livestream-transcriptions/quickstart
    """

    def __init__(
        self,
        telephony_provider: str,
        input_queue=None,
        output_queue=None,
        stream: bool = True,
        language: str = "en",
        endpointing: str = "400",
        encoding: str = "linear16",
        sampling_rate: str = "16000",
        model: str = None,
        **kwargs
    ):
        super().__init__(input_queue)

        # Provider configuration
        self.provider = telephony_provider
        self.stream = stream
        self.language = language
        self.endpointing = int(endpointing)
        self.model = model  # Currently unused but reserved for future model selection

        # API configuration
        self.api_key = kwargs.get("transcriber_key", os.getenv('SHUNYA_API_KEY'))
        self.shunya_host = os.getenv('SHUNYA_HOST', 'tl.shunyalabs.ai')
        
        # Queues
        self.transcriber_output_queue = output_queue

        self.encoding = encoding
        self.sampling_rate = int(sampling_rate)
        self.target_sampling_rate = 16000  # Shunya expects 16kHz
        self.audio_frame_duration = 0.2

        # Configure audio params based on telephony provider
        self._configure_audio_params()

        # Connection state
        self.websocket_connection: Optional[ClientConnection] = None
        self.connection_authenticated = False
        self.session_initialized = False

        # Tasks
        self.transcription_task = None
        self.sender_task = None
        self.heartbeat_task = None
        self.utterance_timeout_task = None

        # Audio tracking
        self.audio_submitted = False
        self.audio_submission_time = None
        self.num_frames = 0
        self.frame_seq = 0  # Shunya requires sequential frame numbers
        self.connection_start_time = None
        self.audio_frame_timestamps = []  # List of (frame_start, frame_end, send_timestamp)

        # Transcript state management
        self.final_transcript = ""
        self.is_transcript_sent_for_processing = False
        self.interruption_signalled = False
        
        # Segment tracking for deduplication
        self.processed_segment_ids = set()  # Track completed segment IDs to avoid duplicates

        # Turn tracking
        self.turn_counter = 0
        self.current_turn_start_time = None
        self.current_turn_id = None
        self.current_turn_interim_details = []
        self.speech_start_time = None
        self.speech_end_time = None

        # Latency tracking
        self.first_result_latency_ms = None
        self.total_stream_duration_ms = None
        self.connection_time = None

        # Timeout monitoring
        self.last_interim_time = None
        self.interim_timeout = kwargs.get("interim_timeout", 5.0)  # Default 5 seconds

        self.connected_via_dashboard = kwargs.get("enforce_streaming", True)

    def _configure_audio_params(self):
        """Configure audio parameters based on telephony provider."""
        if self.provider == "twilio":
            self.encoding = "mulaw"
            self.sampling_rate = 8000
            self.audio_frame_duration = 0.2
        elif self.provider in ("exotel", "plivo"):
            self.encoding = "linear16"
            self.sampling_rate = 8000
            self.audio_frame_duration = 0.2
        elif self.provider == "web_based_call":
            # Web calls typically use 16kHz
            self.encoding = "linear16"
            self.sampling_rate = 16000
            self.audio_frame_duration = 0.256
        elif self.provider == "playground":
            # Playground/dashboard mode
            self.encoding = "linear16"
            self.sampling_rate = 8000
            self.audio_frame_duration = 0.0  # Batch mode
        elif not self.connected_via_dashboard:
            self.encoding = "linear16"
            self.sampling_rate = 16000
        else:
            self.encoding = "linear16"
            self.sampling_rate = 16000
            self.audio_frame_duration = 0.2

    def _get_ws_url(self) -> str:
        """Build the WebSocket URL for Shunya Labs."""
        return f"wss://{self.shunya_host}/"

    def _create_init_message(self) -> dict:
        """Create the initialization message for Shunya WebSocket."""
        return {
            "action": "send",
            "type": "init",
            "config": {
                "language": self.language,
                "api_key": self.api_key
            }
        }

    def _create_frame_message(self, audio_b64: str, frame_seq: int) -> dict:
        """Create an audio frame message for Shunya WebSocket."""
        return {
            "action": "send",
            "type": "frame",
            "frame_seq": frame_seq,
            "audio_inline_b64": audio_b64,
            "dtype": "float32",
            "channels": 1,
            "sr": self.target_sampling_rate
        }

    def _create_end_message(self, frame_seq: int) -> dict:
        """Create the end-of-audio message for Shunya WebSocket."""
        end_sentinel = base64.b64encode(b'END_OF_AUDIO').decode('utf-8')
        return {
            "action": "send",
            "type": "frame",
            "frame_seq": frame_seq,
            "audio_inline_b64": end_sentinel,
            "dtype": "float32",
            "channels": 1,
            "sr": self.target_sampling_rate
        }

    def _convert_audio_to_float32_b64(self, audio_data: bytes) -> Optional[str]:
        """
        Convert audio data to float32 format and base64 encode.
        
        Shunya expects:
        - float32 audio samples
        - 16kHz sample rate
        - Base64 encoded
        """
        try:
            if isinstance(audio_data, str):
                audio_bytes = base64.b64decode(audio_data)
            else:
                audio_bytes = audio_data

            # Handle mulaw encoding (Twilio)
            if self.encoding == "mulaw":
                audio_bytes = audioop.ulaw2lin(audio_bytes, 2)

            # Convert to numpy array
            audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)

            # Convert to float32 FIRST (-1.0 to 1.0 range)
            audio_f32 = audio_int16.astype(np.float32) / 32768.0

            # Resample in float domain
            audio_f32 = self._resample_audio_float(audio_f32, self.sampling_rate, self.target_sampling_rate)

            # Encode to base64
            audio_b64 = base64.b64encode(audio_f32.tobytes()).decode('utf-8')
            
            return audio_b64

        except Exception as e:
            logger.error(f"Error converting audio: {e}")
            return None

    def _resample_audio_float(self, audio_f32: np.ndarray, in_sr: int, out_sr: int) -> np.ndarray:
        """Resample float32 audio from in_sr to out_sr."""
        if in_sr == out_sr:
            return audio_f32
        try:
            gcd = np.gcd(in_sr, out_sr)
            up = out_sr // gcd
            down = in_sr // gcd
            # resample_poly expects float for correct filtering
            return resample_poly(audio_f32, up, down).astype(np.float32)
        except Exception as e:
            logger.error(f"Error resampling audio: {e}")
            return audio_f32

    async def shunya_connect(self, retries: int = 3, timeout: float = 10.0) -> ClientConnection:
        """Establish WebSocket connection to Shunya Labs with retry logic."""
        attempt = 0
        last_err = None

        while attempt < retries:
            try:
                websocket_url = self._get_ws_url()
                
                logger.info(f"Attempting to connect to Shunya Labs WebSocket: {websocket_url}")

                ws = await asyncio.wait_for(
                    websockets.connect(websocket_url),
                    timeout=timeout
                )

                self.websocket_connection = ws
                logger.info("Successfully connected to Shunya Labs WebSocket")
                
                # Send initialization message
                init_msg = self._create_init_message()
                await ws.send(json.dumps(init_msg))
                logger.info("Sent initialization message to Shunya Labs")
                
                self.session_initialized = True
                self.connection_authenticated = True
                
                return ws

            except asyncio.TimeoutError:
                logger.error("Timeout while connecting to Shunya Labs WebSocket")
                raise ConnectionError("Timeout while connecting to Shunya Labs WebSocket")
            except InvalidHandshake as e:
                error_msg = str(e)
                if '401' in error_msg or '403' in error_msg:
                    logger.error(f"Shunya Labs authentication failed: {e}")
                    raise ConnectionError(f"Shunya Labs authentication failed: Invalid API key - {e}")
                else:
                    logger.error(f"Invalid handshake during Shunya Labs connection: {e}")
                    last_err = e
                    attempt += 1
                    if attempt < retries:
                        await asyncio.sleep(2 ** attempt)
            except ConnectionClosedError as e:
                logger.error(f"Shunya Labs WebSocket connection closed unexpectedly: {e}")
                raise ConnectionError(f"Shunya Labs WebSocket connection closed unexpectedly: {e}")
            except Exception as e:
                logger.error(f"Error connecting to Shunya Labs (attempt {attempt + 1}/{retries}): {e}")
                last_err = e
                attempt += 1
                if attempt < retries:
                    await asyncio.sleep(2 ** attempt)

        raise ConnectionError(f"Failed to connect to Shunya Labs after {retries} attempts: {last_err}")

    async def send_heartbeat(self, ws: ClientConnection):
        """Send periodic ping to keep connection alive."""
        try:
            while True:
                try:
                    await ws.ping()
                except ConnectionClosed as e:
                    rcvd_code = getattr(e.rcvd, "code", None)
                    sent_code = getattr(e.sent, "code", None)
                    if rcvd_code == 1000 or sent_code == 1000:
                        logger.info("WebSocket closed normally during heartbeat")
                    else:
                        logger.warning(f"WebSocket closed: received={rcvd_code}, sent={sent_code}")
                    break
                except Exception as e:
                    logger.error(f"Error sending heartbeat: {e}")
                    break

                await asyncio.sleep(5)
        except asyncio.CancelledError:
            logger.info("Heartbeat task cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in send_heartbeat: {e}")
            raise

    def _reset_turn_state(self):
        """Reset turn state after finalizing a transcript."""
        self.speech_start_time = None
        self.speech_end_time = None
        self.last_interim_time = None
        self.current_turn_interim_details = []
        self.current_turn_start_time = None
        self.current_turn_id = None
        self.final_transcript = ""
        self.is_transcript_sent_for_processing = False

    async def _force_finalize_utterance(self):
        """Force-finalize a stuck utterance."""
        transcript_to_send = self.final_transcript.strip()

        # Fallback: use last interim if no final results received
        if not transcript_to_send and self.current_turn_interim_details:
            transcript_to_send = self.current_turn_interim_details[-1]['transcript']
            logger.info(f"Using last interim as fallback: {transcript_to_send}")

        if not transcript_to_send:
            logger.warning("No transcript available to force-finalize")
            self._reset_turn_state()
            return

        # Build turn latencies
        try:
            self.turn_latencies.append({
                'turn_id': self.current_turn_id,
                'sequence_id': self.current_turn_id,
                'interim_details': self.current_turn_interim_details,
                'force_finalized': True
            })
        except Exception as e:
            logger.error(f"Error building turn latencies: {e}")

        data = {
            "type": "transcript",
            "content": transcript_to_send,
            "force_finalized": True
        }

        logger.info(f"Force-finalized transcript: {transcript_to_send}")
        await self.push_to_transcriber_queue(create_ws_data_packet(data, self.meta_info))
        self._reset_turn_state()

    async def monitor_utterance_timeout(self):
        """Monitor for stuck utterances."""
        try:
            while True:
                await asyncio.sleep(0.5)

                # Check for stuck utterances (longer timeout)
                if (self.last_interim_time and
                    not self.is_transcript_sent_for_processing and
                    (self.final_transcript.strip() or self.current_turn_interim_details)):

                    elapsed = time.time() - self.last_interim_time

                    if elapsed > self.interim_timeout:
                        logger.warning(
                            f"Utterance timeout: No finalization for {elapsed:.1f}s. "
                            f"Force-finalizing turn {self.current_turn_id}"
                        )
                        await self._force_finalize_utterance()

        except asyncio.CancelledError:
            logger.info("Utterance timeout monitoring cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in utterance timeout monitor: {e}")
            raise

    async def toggle_connection(self):
        """Close connection and cleanup tasks."""
        self.connection_on = False

        if self.heartbeat_task:
            self.heartbeat_task.cancel()
        if self.sender_task:
            self.sender_task.cancel()
        if self.utterance_timeout_task:
            self.utterance_timeout_task.cancel()

        if self.websocket_connection:
            try:
                # Send end signal before closing
                end_msg = self._create_end_message(self.frame_seq)
                await self.websocket_connection.send(json.dumps(end_msg))
                await self.websocket_connection.close()
                logger.info("Shunya Labs WebSocket connection closed successfully")
            except Exception as e:
                logger.error(f"Error closing WebSocket: {e}")
            finally:
                self.websocket_connection = None
                self.connection_authenticated = False
                self.session_initialized = False

    async def cleanup(self):
        """Clean up all resources including websocket."""
        logger.info("Cleaning up Shunya transcriber resources")

        # Cancel tasks properly
        for task_name, task in [
            ("heartbeat_task", getattr(self, 'heartbeat_task', None)),
            ("sender_task", getattr(self, 'sender_task', None)),
            ("utterance_timeout_task", getattr(self, 'utterance_timeout_task', None)),
            ("transcription_task", getattr(self, 'transcription_task', None))
        ]:
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    logger.info(f"Shunya {task_name} cancelled")
                except Exception as e:
                    logger.error(f"Error cancelling Shunya {task_name}: {e}")

        # Close websocket
        if self.websocket_connection is not None:
            try:
                await self.websocket_connection.close()
                logger.info("Shunya websocket connection closed")
            except Exception as e:
                logger.error(f"Error closing Shunya websocket: {e}")
            finally:
                self.websocket_connection = None
                self.connection_authenticated = False
                self.session_initialized = False

    async def _check_and_process_end_of_stream(self, ws_data_packet, ws):
        """Check for end of stream signal."""
        if ws_data_packet.get('meta_info', {}).get('eos') is True:
            # Send end signal
            end_msg = self._create_end_message(self.frame_seq)
            try:
                await ws.send(json.dumps(end_msg))
            except Exception as e:
                logger.error(f"Error sending end signal: {e}")
            return True
        return False

    def _find_audio_send_timestamp(self, audio_position: float):
        """
        Find when the audio frame containing this position was sent.

        Args:
            audio_position: Position in seconds within the audio stream

        Returns:
            Timestamp when the frame containing this position was sent, or None if not found
        """
        if not self.audio_frame_timestamps:
            return None

        for frame_start, frame_end, send_timestamp in self.audio_frame_timestamps:
            if frame_start <= audio_position <= frame_end:
                return send_timestamp

        return None

    async def sender_stream(self, ws: ClientConnection):
        """
        Send audio data to Shunya Labs WebSocket.

        Shunya expects JSON messages with base64 encoded float32 audio.
        """
        try:
            while True:
                ws_data_packet = await self.input_queue.get()

                if ws_data_packet is None:
                    continue

                # Initialize on first audio packet
                if not self.audio_submitted:
                    self.meta_info = ws_data_packet.get("meta_info", {}) or {}
                    self.audio_submitted = True
                    self.audio_submission_time = time.time()
                    self.current_request_id = self.generate_request_id()
                    self.meta_info["request_id"] = self.current_request_id

                    # Start new turn tracking
                    try:
                        if not self.current_turn_start_time:
                            self.current_turn_start_time = timestamp_ms()
                            self.current_turn_id = self.meta_info.get('turn_id') or self.meta_info.get('request_id')
                    except Exception:
                        pass

                    # Signal speech started
                    self.turn_counter += 1
                    self.current_turn_id = self.turn_counter
                    self.speech_start_time = timestamp_ms()
                    self.current_turn_interim_details = []
                    self.is_transcript_sent_for_processing = False

                    logger.info(f"Starting new turn with turn_id: {self.current_turn_id}")
                    await self.push_to_transcriber_queue(
                        create_ws_data_packet("speech_started", self.meta_info)
                    )

                # Check for end of stream
                end_of_stream = await self._check_and_process_end_of_stream(ws_data_packet, ws)
                if end_of_stream:
                    break

                # Track audio frame timing
                frame_start = self.num_frames * self.audio_frame_duration
                frame_end = (self.num_frames + 1) * self.audio_frame_duration
                send_timestamp = timestamp_ms()
                self.audio_frame_timestamps.append((frame_start, frame_end, send_timestamp))
                self.num_frames += 1

                # Get audio data and convert
                audio_data = ws_data_packet.get("data")
                if audio_data:
                    try:
                        audio_b64 = self._convert_audio_to_float32_b64(audio_data)
                        if audio_b64:
                            frame_msg = self._create_frame_message(audio_b64, self.frame_seq)
                            await ws.send(json.dumps(frame_msg))
                            self.frame_seq += 1

                    except ConnectionClosedError as e:
                        logger.error(f"Connection closed while sending audio: {e}")
                        break
                    except Exception as e:
                        logger.error(f"Error sending audio to Shunya Labs: {e}")
                        break

        except asyncio.CancelledError:
            logger.info("Sender stream task cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in sender_stream: {e}")
            raise

    async def receiver(self, ws: ClientConnection):
        """
        Receive and process messages from Shunya Labs WebSocket.

        Expected response format:
        {
            "uid": "client-identifier",
            "segments": [
                {
                    "start": "0.000",
                    "end": "2.500",
                    "text": "Hello world",
                    "completed": true,
                    "segment_id": "seg_001",
                    "rev": 1
                }
            ],
            "language": "en",
            "language_probability": 0.95
        }
        """
        logger.info("Starting to receive messages from Shunya Labs")
        async for msg in ws:
            logger.info(f"Received message from Shunya Labs: {msg}")
            try:
                data = json.loads(msg) if isinstance(msg, str) else json.loads(msg.decode())

                if self.connection_start_time is None:
                    self.connection_start_time = time.time() - (self.num_frames * self.audio_frame_duration)

                # Handle language detection
                # detected_language = data.get('language')
                # language_probability = data.get('language_probability')
                # if detected_language:
                #     self.meta_info['segment_language'] = detected_language
                #     if language_probability:
                #         self.meta_info['segment_language_probability'] = language_probability

                # Handle segments in response
                segments = data.get('segments', [])
                
                for segment in segments:
                    segment_text = segment.get('text', '').strip()
                    segment_start = segment.get('start', 0)
                    segment_end = segment.get('end', 0)
                    is_completed = segment.get('completed', False)
                    segment_id = segment.get('segment_id', '')
                    revision = segment.get('rev', 1)
                    
                    if not segment_text:
                        continue

                    now_timestamp = time.time()
                    self.last_interim_time = now_timestamp

                    # Calculate latency based on segment timing
                    latency_ms = None
                    try:
                        end_time = float(segment_end) if isinstance(segment_end, str) else segment_end
                        audio_sent_at = self._find_audio_send_timestamp(end_time)
                        if audio_sent_at:
                            result_received_at = timestamp_ms()
                            latency_ms = round(result_received_at - audio_sent_at, 5)
                    except (ValueError, TypeError):
                        pass

                    # Track first result latency
                    if self.first_result_latency_ms is None and self.audio_submission_time:
                        first_latency_seconds = now_timestamp - self.audio_submission_time
                        self.first_result_latency_ms = round(first_latency_seconds * 1000)
                        self.meta_info["transcriber_first_result_latency"] = first_latency_seconds
                        self.meta_info["transcriber_latency"] = first_latency_seconds
                        self.meta_info["first_result_latency_ms"] = self.first_result_latency_ms

                    if is_completed:
                        # Skip if we've already processed this completed segment
                        if segment_id and segment_id in self.processed_segment_ids:
                            logger.debug(f"Skipping already processed segment: {segment_id}")
                            continue
                        
                        if segment_id:
                            self.processed_segment_ids.add(segment_id)

                        logger.info(f"Completed segment [{segment_start}s - {segment_end}s]: {segment_text}")

                        # Track interim details for latency tracking
                        interim_detail = {
                            'transcript': segment_text,
                            'is_final': True,
                            'latency_ms': latency_ms,
                            'start': segment_start,
                            'end': segment_end,
                            'segment_id': segment_id,
                            'rev': revision,
                            'received_at': now_timestamp
                        }
                        self.current_turn_interim_details.append(interim_detail)

                        # Build turn latencies
                        try:
                            self.turn_latencies.append({
                                'turn_id': self.current_turn_id,
                                'sequence_id': self.current_turn_id,
                                'interim_details': self.current_turn_interim_details,
                                # 'detected_language': detected_language
                            })
                        except Exception as e:
                            logger.error(f"Error building turn latencies: {e}")

                        # Yield final transcript
                        transcript_packet = {
                            "type": "transcript",
                            "content": segment_text
                        }
                        yield create_ws_data_packet(transcript_packet, self.meta_info)

                        # Reset turn state for next utterance
                        self._reset_turn_state()

                    else:
                        # Partial/interim transcript
                        logger.info(f"Partial segment [{segment_start}s - {segment_end}s]: {segment_text} (rev: {revision})")

                        # Track interim details
                        interim_detail = {
                            'transcript': segment_text,
                            'is_final': False,
                            'latency_ms': latency_ms,
                            'start': segment_start,
                            'end': segment_end,
                            'segment_id': segment_id,
                            'rev': revision,
                            'received_at': now_timestamp
                        }
                        self.current_turn_interim_details.append(interim_detail)

                        # Update final_transcript with latest partial (for potential force-finalize)
                        self.final_transcript = segment_text

                        # Yield interim transcript
                        interim_packet = {
                            "type": "interim_transcript_received",
                            "content": segment_text
                        }
                        yield create_ws_data_packet(interim_packet, self.meta_info)

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse Shunya Labs message: {e}")
            except Exception as e:
                logger.error(f"Error processing Shunya Labs message: {e}")
                traceback.print_exc()

    async def push_to_transcriber_queue(self, data_packet):
        """Push data to the output queue."""
        if self.transcriber_output_queue:
            await self.transcriber_output_queue.put(data_packet)

    def get_meta_info(self):
        """Return current meta_info."""
        return getattr(self, 'meta_info', {})

    async def run(self):
        """Start the transcription task."""
        try:
            self.transcription_task = asyncio.create_task(self.transcribe())
        except Exception as e:
            logger.error(f"Error starting transcription task: {e}")

    async def transcribe(self):
        """Main transcription method."""
        shunya_ws = None
        try:
            start_time = timestamp_ms()

            try:
                shunya_ws = await self.shunya_connect()
            except (ValueError, ConnectionError) as e:
                logger.error(f"Failed to establish Shunya Labs connection: {e}")
                await self.toggle_connection()
                return

            if not self.connection_time:
                self.connection_time = round(timestamp_ms() - start_time)

            if self.stream:
                self.sender_task = asyncio.create_task(self.sender_stream(shunya_ws))
                self.heartbeat_task = asyncio.create_task(self.send_heartbeat(shunya_ws))
                self.utterance_timeout_task = asyncio.create_task(self.monitor_utterance_timeout())

                try:
                    async for message in self.receiver(shunya_ws):
                        if self.connection_on:
                            await self.push_to_transcriber_queue(message)
                        else:
                            logger.info("Closing Shunya Labs connection")
                            end_msg = self._create_end_message(self.frame_seq)
                            await shunya_ws.send(json.dumps(end_msg))
                            break
                except ConnectionClosedError as e:
                    logger.error(f"Shunya Labs WebSocket closed during streaming: {e}")
                except Exception as e:
                    logger.error(f"Error during streaming: {e}")
                    raise
            else:
                # Non-streaming mode not fully supported
                logger.warning("Non-streaming mode not recommended for Shunya Labs")

        except (ValueError, ConnectionError) as e:
            logger.error(f"Connection error in transcribe: {e}")
            await self.toggle_connection()
        except Exception as e:
            logger.error(f"Unexpected error in transcribe: {e}")
            await self.toggle_connection()
        finally:
            if shunya_ws is not None:
                try:
                    await shunya_ws.close()
                    logger.info("Shunya Labs WebSocket closed in finally block")
                except Exception as e:
                    logger.error(f"Error closing WebSocket: {e}")
                finally:
                    self.websocket_connection = None
                    self.connection_authenticated = False
                    self.session_initialized = False

            # Cancel tasks
            if hasattr(self, 'sender_task') and self.sender_task:
                self.sender_task.cancel()
            if hasattr(self, 'heartbeat_task') and self.heartbeat_task:
                self.heartbeat_task.cancel()
            if hasattr(self, 'utterance_timeout_task') and self.utterance_timeout_task:
                self.utterance_timeout_task.cancel()

            # Send connection closed message
            await self.push_to_transcriber_queue(
                create_ws_data_packet("transcriber_connection_closed", getattr(self, 'meta_info', {}))
            )
