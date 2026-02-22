import asyncio
import json
import os
import time
import traceback
from typing import Optional
from urllib.parse import urlencode

import websockets
from websockets.asyncio.client import ClientConnection
from websockets.exceptions import ConnectionClosedError, InvalidHandshake, ConnectionClosed
from dotenv import load_dotenv

from .base_transcriber import BaseTranscriber
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.utils import create_ws_data_packet, timestamp_ms

load_dotenv()
logger = configure_logger(__name__)


class SmallestTranscriber(BaseTranscriber):
    """
    Streaming transcriber using Smallest AI Lightning ASR WebSocket API.

    Smallest AI Lightning ASR is optimized for:
    - Sub-300ms time-to-first-transcript latency
    - High accuracy across 24+ languages
    - Real-time streaming without waiting for complete audio

    API Documentation: https://waves-docs.smallest.ai/v4.0.0/content/api-references/lightning-asr-ws
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
        keywords: str = None,
        word_timestamps: bool = True,
        process_interim_results: str = "true",
        **kwargs
    ):
        super().__init__(input_queue)

        # Provider configuration
        self.provider = telephony_provider
        self.stream = stream
        self.language = language
        self.endpointing = endpointing
        self.model = model
        self.keywords = keywords
        self.word_timestamps = word_timestamps
        self.process_interim_results = process_interim_results

        # API configuration
        self.api_key = kwargs.get("transcriber_key", os.getenv('SMALLEST_API_KEY'))
        self.smallest_host = os.getenv('SMALLEST_HOST', 'waves-api.smallest.ai')

        # Queues
        self.transcriber_output_queue = output_queue

        # Audio configuration (defaults, will be adjusted based on provider)
        self.encoding = encoding
        self.sampling_rate = int(sampling_rate)
        self.audio_frame_duration = 0.2  # Default 200ms chunks

        # Configure audio params based on telephony provider
        self._configure_audio_params()

        # Connection state
        self.websocket_connection: Optional[ClientConnection] = None
        self.connection_authenticated = False
        self.smallest_session_id: Optional[str] = None
        self.connection_error: Optional[str] = None

        # Tasks
        self.transcription_task = None
        self.sender_task = None
        self.heartbeat_task = None
        self.utterance_timeout_task = None

        # Audio tracking
        self.audio_submitted = False
        self.audio_submission_time = None
        self.num_frames = 0
        self.connection_start_time = None
        self.audio_frame_timestamps = []  # List of (frame_start, frame_end, send_timestamp)

        # Transcript state management
        self.final_transcript = ""
        self.is_transcript_sent_for_processing = False
        self.interruption_signalled = False

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

        # Timeout monitoring (like Deepgram)
        self.last_interim_time = None
        self.interim_timeout = kwargs.get("interim_timeout", 5.0)  # Default 5 seconds

        # Dashboard connection flag
        self.connected_via_dashboard = kwargs.get("enforce_streaming", True)

    def _configure_audio_params(self):
        """Configure audio parameters based on telephony provider."""
        if self.provider == "twilio":
            # Twilio sends mulaw at 8kHz
            self.encoding = "mulaw"
            self.sampling_rate = 8000
            self.audio_frame_duration = 0.2
        elif self.provider in ("exotel", "plivo"):
            # Exotel and Plivo send linear16 at 8kHz
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
            # Default configuration - optimal for accuracy
            self.encoding = "linear16"
            self.sampling_rate = 16000
            self.audio_frame_duration = 0.2

    def get_smallest_ws_url(self) -> str:
        """
        Build the WebSocket URL for Smallest AI Lightning ASR.

        Query params:
        - language: Language code (en, hi, multi, etc.)
        - sample_rate: Audio sample rate (8000, 16000, etc.)
        - encoding: Audio encoding (linear16, mulaw, etc.)
        - word_timestamps: Enable word-level timing
        """
        params = {
            'language': self.language,
            'sample_rate': str(self.sampling_rate),
        }

        # Add encoding if not linear16 (default)
        if self.encoding != "linear16":
            params['encoding'] = self.encoding

        # Add word timestamps if enabled
        if self.word_timestamps:
            params['word_timestamps'] = 'true'

        websocket_url = f"wss://{self.smallest_host}/api/v1/lightning/get_text?{urlencode(params)}"
        logger.info(f"Smallest WebSocket URL params - language: {self.language}, sample_rate: {self.sampling_rate}, encoding: {self.encoding}, word_timestamps: {self.word_timestamps}")
        return websocket_url

    async def smallest_connect(self, retries: int = 3, timeout: float = 10.0) -> ClientConnection:
        """
        Establish WebSocket connection to Smallest AI with retry logic.
        """
        attempt = 0
        last_err = None

        while attempt < retries:
            try:
                websocket_url = self.get_smallest_ws_url()
                additional_headers = {
                    'Authorization': f'Bearer {self.api_key}'
                }

                logger.info(f"Attempting to connect to Smallest AI WebSocket: {websocket_url}")

                ws = await asyncio.wait_for(
                    websockets.connect(websocket_url, additional_headers=additional_headers),
                    timeout=timeout
                )

                self.websocket_connection = ws
                self.connection_authenticated = True
                logger.info("Successfully connected to Smallest AI WebSocket")
                return ws

            except asyncio.TimeoutError:
                logger.error("Timeout while connecting to Smallest AI WebSocket")
                raise ConnectionError("Timeout while connecting to Smallest AI WebSocket")
            except InvalidHandshake as e:
                error_msg = str(e)
                if '401' in error_msg or '403' in error_msg:
                    logger.error(f"Smallest AI authentication failed: {e}")
                    raise ConnectionError(f"Smallest AI authentication failed: Invalid API key - {e}")
                else:
                    logger.error(f"Invalid handshake during Smallest AI connection: {e}")
                    last_err = e
                    attempt += 1
                    if attempt < retries:
                        await asyncio.sleep(2 ** attempt)
            except ConnectionClosedError as e:
                logger.error(f"Smallest AI WebSocket connection closed unexpectedly: {e}")
                raise ConnectionError(f"Smallest AI WebSocket connection closed unexpectedly: {e}")
            except Exception as e:
                logger.error(f"Error connecting to Smallest AI (attempt {attempt + 1}/{retries}): {e}")
                last_err = e
                attempt += 1
                if attempt < retries:
                    await asyncio.sleep(2 ** attempt)

        raise ConnectionError(f"Failed to connect to Smallest AI after {retries} attempts: {last_err}")

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
        self.is_transcript_sent_for_processing = True

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
            first_interim_to_final_ms, last_interim_to_final_ms = self.calculate_interim_to_final_latencies(self.current_turn_interim_details)

            self.turn_latencies.append({
                'turn_id': self.current_turn_id,
                'sequence_id': self.current_turn_id,
                'interim_details': self.current_turn_interim_details,
                'first_interim_to_final_ms': first_interim_to_final_ms,
                'last_interim_to_final_ms': last_interim_to_final_ms,
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
        """Monitor for stuck utterances that never receive final transcript."""
        try:
            while True:
                await asyncio.sleep(1.0)

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
                await self._close_smallest(self.websocket_connection)
                logger.info("Smallest AI WebSocket connection closed successfully")
            except Exception as e:
                logger.error(f"Error closing WebSocket: {e}")
            finally:
                self.websocket_connection = None
                self.connection_authenticated = False

    async def cleanup(self):
        """Clean up all resources including websocket."""
        logger.info("Cleaning up Smallest transcriber resources")

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
                    logger.info(f"Smallest {task_name} cancelled")
                except Exception as e:
                    logger.error(f"Error cancelling Smallest {task_name}: {e}")

        # Close websocket
        if self.websocket_connection is not None:
            try:
                await self.websocket_connection.close()
                logger.info("Smallest websocket connection closed")
            except Exception as e:
                logger.error(f"Error closing Smallest websocket: {e}")
            finally:
                self.websocket_connection = None
                self.connection_authenticated = False

    async def _close_smallest(self, ws: ClientConnection):
        """Send end signal and close WebSocket."""
        try:
            # Send Smallest AI end signal
            end_msg = {"type": "end"}
            await ws.send(json.dumps(end_msg))
            await ws.close()
        except Exception as e:
            logger.error(f"Error closing Smallest AI stream: {e}")

    async def _check_and_process_end_of_stream(self, ws_data_packet, ws):
        """Check for end of stream signal."""
        if ws_data_packet.get('meta_info', {}).get('eos') is True:
            await self._close_smallest(ws)
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
        Send audio data to Smallest AI WebSocket.

        Smallest AI expects binary audio chunks (4KB recommended).
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

                    # Signal speech started (Smallest doesn't have VAD events)
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

                # Get audio data and send to Smallest AI
                audio_data = ws_data_packet.get("data")
                if audio_data:
                    try:
                        # Smallest AI expects raw binary audio chunks
                        if isinstance(audio_data, bytes):
                            await ws.send(audio_data)
                        else:
                            # If string (possibly base64), send as-is
                            await ws.send(audio_data)

                    except ConnectionClosedError as e:
                        logger.error(f"Connection closed while sending audio: {e}")
                        break
                    except Exception as e:
                        logger.error(f"Error sending audio to Smallest AI: {e}")
                        break

        except asyncio.CancelledError:
            logger.info("Sender stream task cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in sender_stream: {e}")
            raise

    async def receiver(self, ws: ClientConnection):
        """
        Receive and process messages from Smallest AI WebSocket.

        Response format:
        {
            "session_id": "sess_xxx",
            "transcript": "Hello world",
            "full_transcript": "Hello world",
            "is_final": true,
            "is_last": false,
            "language": "en"
        }
        """
        async for msg in ws:
            try:
                data = json.loads(msg) if isinstance(msg, str) else json.loads(msg.decode())

                if self.connection_start_time is None:
                    self.connection_start_time = time.time() - (self.num_frames * self.audio_frame_duration)

                # Extract session ID if present
                if 'session_id' in data and not self.smallest_session_id:
                    self.smallest_session_id = data['session_id']
                    logger.info(f"Smallest AI session ID: {self.smallest_session_id}")

                # Get transcript text
                transcript = data.get('transcript', '').strip()
                full_transcript = data.get('full_transcript', '').strip()
                is_final = data.get('is_final', False)
                is_last = data.get('is_last', False)
                detected_language = data.get('language')

                if transcript:
                    now_timestamp = time.time()

                    # Calculate latency
                    latency_ms = None
                    # Use audio frame timestamps for latency calculation
                    if self.audio_frame_timestamps and self.num_frames > 0:
                        # Estimate audio position based on current frame count
                        audio_position = self.num_frames * self.audio_frame_duration
                        audio_sent_at = self._find_audio_send_timestamp(audio_position)
                        if audio_sent_at:
                            result_received_at = timestamp_ms()
                            latency_ms = round(result_received_at - audio_sent_at, 5)

                    # Track first result latency
                    if self.first_result_latency_ms is None and self.audio_submission_time:
                        first_latency_seconds = now_timestamp - self.audio_submission_time
                        self.first_result_latency_ms = round(first_latency_seconds * 1000)
                        self.meta_info["transcriber_first_result_latency"] = first_latency_seconds
                        self.meta_info["transcriber_latency"] = first_latency_seconds
                        self.meta_info["first_result_latency_ms"] = self.first_result_latency_ms

                    # Track interim details
                    interim_detail = {
                        'transcript': transcript,
                        'latency_ms': latency_ms,
                        'is_final': is_final,
                        'received_at': now_timestamp,
                        'language': detected_language,
                        'session_id': self.smallest_session_id
                    }
                    self.current_turn_interim_details.append(interim_detail)
                    self.last_interim_time = now_timestamp

                    logger.info(f"Received transcript - is_final: {is_final}, language: {detected_language}, text: {transcript}")

                    if is_final:
                        # Use only the segment transcript (not cumulative full_transcript)
                        # This gives us per-utterance transcripts for turn-based conversation
                        segment_transcript = transcript

                        # Accumulate to our turn transcript
                        if segment_transcript:
                            if self.final_transcript:
                                self.final_transcript += " " + segment_transcript
                            else:
                                self.final_transcript = segment_transcript

                        # Calculate total duration
                        if self.current_turn_start_time:
                            total_stream_duration = time.time() - (self.current_turn_start_time / 1000)
                            self.meta_info['transcriber_total_stream_duration'] = total_stream_duration
                            self.meta_info['transcriber_latency'] = total_stream_duration

                        # Build turn latencies
                        try:
                            first_interim_to_final_ms, last_interim_to_final_ms = self.calculate_interim_to_final_latencies(self.current_turn_interim_details)

                            self.turn_latencies.append({
                                'turn_id': self.current_turn_id,
                                'sequence_id': self.current_turn_id,
                                'interim_details': self.current_turn_interim_details,
                                'first_interim_to_final_ms': first_interim_to_final_ms,
                                'last_interim_to_final_ms': last_interim_to_final_ms
                            })
                        except Exception as e:
                            logger.error(f"Error building turn latencies: {e}")

                        transcript_packet = {
                            "type": "transcript",
                            "content": segment_transcript  # Yield just the segment
                        }
                        logger.info(f"Yielding final transcript segment: {segment_transcript}")
                        yield create_ws_data_packet(transcript_packet, self.meta_info)

                        # Mark transcript as processed but DON'T reset turn state
                        # Let utterance timeout or explicit end signal reset state
                        self.is_transcript_sent_for_processing = True
                        self.current_turn_interim_details = []
                    else:
                        # Interim transcript
                        interim_packet = {
                            "type": "interim_transcript_received",
                            "content": transcript
                        }
                        yield create_ws_data_packet(interim_packet, self.meta_info)

                # Check if this is the last message
                if is_last:
                    logger.info("Received is_last=true, session complete")
                    self.meta_info["transcriber_duration"] = time.time() - (self.connection_start_time or time.time())
                    yield create_ws_data_packet("transcriber_connection_closed", self.meta_info)
                    return

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse Smallest AI message: {e}")
            except Exception as e:
                logger.error(f"Error processing Smallest AI message: {e}")
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
        smallest_ws = None
        try:
            start_time = timestamp_ms()

            try:
                smallest_ws = await self.smallest_connect()
            except (ValueError, ConnectionError) as e:
                logger.error(f"Failed to establish Smallest AI connection: {e}")
                await self.toggle_connection()
                return

            if not self.connection_time:
                self.connection_time = round(timestamp_ms() - start_time)

            if self.stream:
                self.sender_task = asyncio.create_task(self.sender_stream(smallest_ws))
                self.heartbeat_task = asyncio.create_task(self.send_heartbeat(smallest_ws))
                self.utterance_timeout_task = asyncio.create_task(self.monitor_utterance_timeout())

                try:
                    async for message in self.receiver(smallest_ws):
                        if self.connection_on:
                            await self.push_to_transcriber_queue(message)
                        else:
                            logger.info("Closing Smallest AI connection")
                            await self._close_smallest(smallest_ws)
                            break
                except ConnectionClosedError as e:
                    logger.error(f"Smallest AI WebSocket closed during streaming: {e}")
                    self.connection_error = str(e)
                except Exception as e:
                    logger.error(f"Error during streaming: {e}")
                    self.connection_error = str(e)
                    raise
            else:
                # Non-streaming mode not supported for Smallest AI
                logger.warning("Non-streaming mode not supported for Smallest AI")

        except (ValueError, ConnectionError) as e:
            logger.error(f"Connection error in transcribe: {e}")
            await self.toggle_connection()
        except Exception as e:
            logger.error(f"Unexpected error in transcribe: {e}")
            await self.toggle_connection()
        finally:
            if smallest_ws is not None:
                try:
                    await smallest_ws.close()
                    logger.info("Smallest AI WebSocket closed in finally block")
                except Exception as e:
                    logger.error(f"Error closing WebSocket: {e}")
                finally:
                    self.websocket_connection = None
                    self.connection_authenticated = False

            # Cancel tasks
            if hasattr(self, 'sender_task') and self.sender_task:
                self.sender_task.cancel()
            if hasattr(self, 'heartbeat_task') and self.heartbeat_task:
                self.heartbeat_task.cancel()
            if hasattr(self, 'utterance_timeout_task') and self.utterance_timeout_task:
                self.utterance_timeout_task.cancel()

            # Send connection closed message
            meta = dict(getattr(self, 'meta_info', None) or {})
            if self.connection_error:
                meta['connection_error'] = self.connection_error
            await self.push_to_transcriber_queue(
                create_ws_data_packet("transcriber_connection_closed", meta)
            )
