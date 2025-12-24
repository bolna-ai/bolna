import asyncio
import base64
import json
import os
import time
import traceback
from typing import Optional

import aiohttp
import websockets
from websockets.asyncio.client import ClientConnection
from websockets.exceptions import ConnectionClosedError, InvalidHandshake, ConnectionClosed
from dotenv import load_dotenv

from .base_transcriber import BaseTranscriber
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.utils import create_ws_data_packet, timestamp_ms

load_dotenv()
logger = configure_logger(__name__)


class GladiaTranscriber(BaseTranscriber):
    """
    Streaming transcriber using Gladia ASR API.

    Gladia uses a two-step connection:
    1. POST to /v2/live to get session URL
    2. Connect WebSocket to returned URL

    Key features:
    - Native mulaw support (no conversion needed for Twilio)
    - ~300ms latency
    - Code-switching support for multilingual conversations
    """

    def __init__(
        self,
        telephony_provider: str,
        input_queue=None,
        output_queue=None,
        stream: bool = True,
        language: str = "en",
        endpointing: int = 500,  # milliseconds (same format as Deepgram), converted to seconds for Gladia
        maximum_duration_without_endpointing: int = 5,
        speech_threshold: float = 0.6,
        code_switching: bool = False,
        keywords: str = None,  # Custom vocabulary keywords (comma-separated)
        model: str = None,     # Optional model for future Gladia models
        **kwargs
    ):
        super().__init__(input_queue)

        # Provider configuration
        self.provider = telephony_provider
        self.stream = stream
        self.language = language

        # Gladia-specific configuration
        # Convert endpointing from milliseconds (agent config format) to seconds (Gladia API format)
        self.endpointing = endpointing / 1000.0
        self.maximum_duration_without_endpointing = maximum_duration_without_endpointing
        self.speech_threshold = speech_threshold
        self.code_switching = code_switching
        self.keywords = keywords
        self.model = model

        # API configuration
        self.api_key = kwargs.get("transcriber_key", os.getenv('GLADIA_API_KEY'))
        self.gladia_host = os.getenv('GLADIA_HOST', 'api.gladia.io')
        self.session_url = f"https://{self.gladia_host}/v2/live"

        # Queues
        self.transcriber_output_queue = output_queue

        # Audio configuration (set based on provider)
        self.encoding = "wav/pcm"
        self.sample_rate = 16000
        self.bit_depth = 16
        self.audio_frame_duration = 0.2
        self._configure_audio_params()

        # HTTP session for batch mode
        self.http_session: Optional[aiohttp.ClientSession] = None

        # Connection state
        self.websocket_connection: Optional[ClientConnection] = None
        self.connection_authenticated = False
        self.gladia_session_id: Optional[str] = None
        self.gladia_ws_url: Optional[str] = None

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
        self.audio_cursor = 0.0

        # Transcript state management
        self.current_transcript = ""
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
        self.last_vocal_frame_timestamp = None

        # Timeout monitoring (like Deepgram)
        self.last_interim_time = None
        self.interim_timeout = kwargs.get("interim_timeout", 5.0)

        # Dashboard connection flag
        self.connected_via_dashboard = kwargs.get("enforce_streaming", True)

    def _configure_audio_params(self):
        """Configure audio parameters based on telephony provider."""
        if self.provider == "twilio":
            # Twilio sends mulaw at 8kHz - Gladia supports this natively
            self.encoding = "wav/ulaw"
            self.sample_rate = 8000
            self.bit_depth = 8
            self.audio_frame_duration = 0.2
        elif self.provider in ("exotel", "plivo"):
            # Exotel and Plivo send linear16 at 8kHz
            self.encoding = "wav/pcm"
            self.sample_rate = 8000
            self.bit_depth = 16
            self.audio_frame_duration = 0.2
        elif self.provider == "web_based_call":
            # Web calls typically use 16kHz
            self.encoding = "wav/pcm"
            self.sample_rate = 16000
            self.bit_depth = 16
            self.audio_frame_duration = 0.256
        elif self.provider == "playground":
            # Playground/dashboard mode
            self.encoding = "wav/pcm"
            self.sample_rate = 8000
            self.bit_depth = 16
            self.audio_frame_duration = 0.0  # Batch mode
        else:
            # Default configuration
            self.encoding = "wav/pcm"
            self.sample_rate = 16000
            self.bit_depth = 16
            self.audio_frame_duration = 0.2

    async def _create_gladia_session(self) -> tuple[str, str]:
        """
        Create a Gladia streaming session via POST request.
        Returns (session_id, websocket_url).
        """
        headers = {
            "X-Gladia-Key": self.api_key,
            "Content-Type": "application/json"
        }

        # When code_switching is enabled, specify allowed languages to restrict detection
        # This prevents misdetection of other languages (e.g., Chinese instead of Hindi)
        if self.code_switching and self.language:
            # For Hindi with code_switching, allow Hindi + English (Hinglish)
            if self.language in ("hi", "hi-IN"):
                languages_list = ["hi", "en"]
            else:
                # For other languages with code_switching, include English as secondary
                languages_list = [self.language, "en"]
        else:
            languages_list = [self.language] if self.language else []

        payload = {
            "encoding": self.encoding,
            "sample_rate": self.sample_rate,
            "bit_depth": self.bit_depth,
            "channels": 1,
            "language_config": {
                "languages": languages_list,
                "code_switching": self.code_switching
            },
            "endpointing": self.endpointing,
            "maximum_duration_without_endpointing": self.maximum_duration_without_endpointing,
            "pre_processing": {
                "audio_enhancer": self.provider in ("twilio", "exotel", "plivo"),
                "speech_threshold": self.speech_threshold
            },
            "messages_config": {
                "receive_partial_transcripts": True,
                "receive_speech_events": True,
                "receive_lifecycle_events": False
            }
        }

        # Add model if specified (for future Gladia models)
        if self.model:
            payload["model"] = self.model

        # Add custom vocabulary if keywords provided
        if self.keywords:
            vocabulary_list = [kw.strip() for kw in self.keywords.split(",") if kw.strip()]
            if vocabulary_list:
                payload["realtime_processing"] = {
                    "custom_vocabulary": True,
                    "custom_vocabulary_config": {
                        "vocabulary": vocabulary_list,
                        "default_intensity": 0.7
                    }
                }

        logger.info(f"Creating Gladia session: encoding={self.encoding}, "
                   f"sample_rate={self.sample_rate}, languages={languages_list}, "
                   f"endpointing={self.endpointing}s, code_switching={self.code_switching}, "
                   f"audio_enhancer={payload['pre_processing'].get('audio_enhancer', False)}, "
                   f"keywords={bool(self.keywords)}")

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.session_url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status not in (200, 201):
                    error_text = await response.text()
                    logger.error(f"Failed to create Gladia session: {response.status} - {error_text}")
                    raise ConnectionError(f"Failed to create Gladia session: {response.status} - {error_text}")

                data = await response.json()
                session_id = data.get("id")
                ws_url = data.get("url")

                if not ws_url:
                    raise ConnectionError("Gladia session response missing WebSocket URL")

                logger.info(f"Created Gladia session: {session_id}")
                return session_id, ws_url

    async def gladia_connect(self, retries: int = 3, timeout: float = 10.0) -> ClientConnection:
        """
        Establish WebSocket connection to Gladia.
        Two-step process: create session, then connect to WebSocket.
        """
        attempt = 0
        last_err = None

        while attempt < retries:
            try:
                # Step 1: Create session and get WebSocket URL
                self.gladia_session_id, self.gladia_ws_url = await self._create_gladia_session()

                # Step 2: Connect to WebSocket
                logger.info(f"Connecting to Gladia WebSocket: {self.gladia_ws_url}")

                ws = await asyncio.wait_for(
                    websockets.connect(self.gladia_ws_url),
                    timeout=timeout
                )

                self.websocket_connection = ws
                self.connection_authenticated = True
                logger.info("Successfully connected to Gladia WebSocket")
                return ws

            except asyncio.TimeoutError:
                logger.error("Timeout while connecting to Gladia")
                raise ConnectionError("Timeout while connecting to Gladia WebSocket")
            except InvalidHandshake as e:
                error_msg = str(e)
                if '401' in error_msg or '403' in error_msg:
                    logger.error(f"Gladia authentication failed: {e}")
                    raise ConnectionError(f"Gladia authentication failed: Invalid API key - {e}")
                else:
                    logger.error(f"Invalid handshake during Gladia connection: {e}")
                    last_err = e
                    attempt += 1
                    if attempt < retries:
                        await asyncio.sleep(2 ** attempt)
            except ConnectionError:
                raise
            except Exception as e:
                logger.error(f"Error connecting to Gladia (attempt {attempt + 1}/{retries}): {e}")
                last_err = e
                attempt += 1
                if attempt < retries:
                    await asyncio.sleep(2 ** attempt)

        raise ConnectionError(f"Failed to connect to Gladia after {retries} attempts: {last_err}")

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
                await self._close_gladia(self.websocket_connection)
                logger.info("Gladia WebSocket connection closed successfully")
            except Exception as e:
                logger.error(f"Error closing WebSocket: {e}")
            finally:
                self.websocket_connection = None
                self.connection_authenticated = False

    async def cleanup(self):
        """Clean up all resources including HTTP session and websocket."""
        logger.info("Cleaning up Gladia transcriber resources")

        # Close HTTP session
        if self.http_session and not self.http_session.closed:
            try:
                await self.http_session.close()
                logger.info("Gladia HTTP session closed")
            except Exception as e:
                logger.error(f"Error closing Gladia HTTP session: {e}")

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
                    logger.info(f"Gladia {task_name} cancelled")
                except Exception as e:
                    logger.error(f"Error cancelling Gladia {task_name}: {e}")

        # Close websocket
        if self.websocket_connection is not None:
            try:
                await self.websocket_connection.close()
                logger.info("Gladia websocket connection closed")
            except Exception as e:
                logger.error(f"Error closing Gladia websocket: {e}")
            finally:
                self.websocket_connection = None
                self.connection_authenticated = False

    async def _close_gladia(self, ws: ClientConnection):
        """Send stop_recording and close WebSocket."""
        try:
            # Send Gladia stop command
            stop_msg = {"type": "stop_recording"}
            await ws.send(json.dumps(stop_msg))
            await ws.close()
        except Exception as e:
            logger.error(f"Error closing Gladia stream: {e}")

    async def _check_and_process_end_of_stream(self, ws_data_packet, ws):
        """Check for end of stream signal."""
        if ws_data_packet.get('meta_info', {}).get('eos') is True:
            await self._close_gladia(ws)
            return True
        return False

    def _find_audio_send_timestamp(self, audio_position: float):
        """Find when the audio frame containing this position was sent."""
        if not self.audio_frame_timestamps:
            return None

        for frame_start, frame_end, send_timestamp in self.audio_frame_timestamps:
            if frame_start <= audio_position <= frame_end:
                return send_timestamp

        return None

    async def sender_stream(self, ws: ClientConnection):
        """Send audio data to Gladia WebSocket."""
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
                    try:
                        if not self.current_turn_start_time:
                            self.current_turn_start_time = timestamp_ms()
                            self.current_turn_id = self.meta_info.get('turn_id') or self.meta_info.get('request_id')
                    except Exception:
                        pass

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
                self.audio_cursor = self.num_frames * self.audio_frame_duration

                # Get audio data and send to Gladia
                audio_data = ws_data_packet.get("data")
                if audio_data:
                    try:
                        # Gladia expects base64 encoded audio in JSON
                        if isinstance(audio_data, bytes):
                            audio_b64 = base64.b64encode(audio_data).decode("utf-8")
                        else:
                            audio_b64 = audio_data  # Assume already base64 string

                        message = {
                            "type": "audio_chunk",
                            "data": {
                                "chunk": audio_b64
                            }
                        }
                        await ws.send(json.dumps(message))

                    except ConnectionClosedError as e:
                        logger.error(f"Connection closed while sending audio: {e}")
                        break
                    except Exception as e:
                        logger.error(f"Error sending audio to Gladia: {e}")
                        break

        except asyncio.CancelledError:
            logger.info("Sender stream task cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in sender_stream: {e}")
            raise

    async def receiver(self, ws: ClientConnection):
        """Receive and process messages from Gladia WebSocket."""
        async for msg in ws:
            try:
                data = json.loads(msg) if isinstance(msg, str) else msg

                if self.connection_start_time is None:
                    self.connection_start_time = time.time() - (self.num_frames * self.audio_frame_duration)

                msg_type = data.get("type")

                if msg_type == "transcript":
                    # Handle transcript message
                    transcript_data = data.get("data", {})
                    utterance = transcript_data.get("utterance", {})
                    text = utterance.get("text", "").strip()
                    is_final = transcript_data.get("is_final", False)
                    detected_language = utterance.get("language")

                    if text:
                        now_timestamp = time.time()

                        # Calculate latency using end position
                        latency_ms = None
                        end_time = utterance.get("end", 0)
                        audio_sent_at = self._find_audio_send_timestamp(end_time)
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
                            'transcript': text,
                            'latency_ms': latency_ms,
                            'is_final': is_final,
                            'received_at': now_timestamp,
                            'language': detected_language
                        }
                        self.current_turn_interim_details.append(interim_detail)
                        self.last_interim_time = now_timestamp

                        # Store detected language in meta_info
                        if detected_language:
                            self.meta_info['segment_language'] = detected_language

                        if is_final:
                            # Final transcript
                            logger.info(f"Received final transcript: {text}")
                            self.final_transcript = text

                            # Calculate total duration
                            if self.current_turn_start_time:
                                total_stream_duration = time.time() - (self.current_turn_start_time / 1000)
                                self.meta_info['transcriber_total_stream_duration'] = total_stream_duration
                                self.meta_info['transcriber_latency'] = total_stream_duration

                            # Build turn latencies
                            try:
                                self.turn_latencies.append({
                                    'turn_id': self.current_turn_id,
                                    'sequence_id': self.current_turn_id,
                                    'interim_details': self.current_turn_interim_details
                                })
                            except Exception as e:
                                logger.error(f"Error building turn latencies: {e}")

                            transcript_packet = {
                                "type": "transcript",
                                "content": text
                            }
                            yield create_ws_data_packet(transcript_packet, self.meta_info)

                            # Reset turn state
                            self._reset_turn_state()
                        else:
                            # Interim transcript
                            logger.debug(f"Received interim transcript: {text}")
                            interim_packet = {
                                "type": "interim_transcript_received",
                                "content": text
                            }
                            yield create_ws_data_packet(interim_packet, self.meta_info)

                elif msg_type == "speech_begin" or (msg_type == "event" and data.get("data", {}).get("type") == "speech_begin"):
                    # Speech started event (VAD detected voice)
                    logger.info("Received speech_begin event from Gladia")
                    self.turn_counter += 1
                    self.current_turn_id = self.turn_counter
                    self.speech_start_time = timestamp_ms()
                    self.current_turn_start_time = timestamp_ms()
                    self.current_turn_interim_details = []
                    self.is_transcript_sent_for_processing = False

                    yield create_ws_data_packet("speech_started", self.meta_info)

                elif msg_type == "speech_end" or (msg_type == "event" and data.get("data", {}).get("type") == "speech_end"):
                    # Speech ended event
                    logger.info("Received speech_end event from Gladia")
                    self.speech_end_time = timestamp_ms()

                elif msg_type == "error":
                    error_data = data.get("data", {})
                    error_msg = error_data.get("message", str(data))
                    logger.error(f"Gladia error: {error_msg}")

                elif msg_type == "ready" or msg_type == "started":
                    logger.info("Gladia connection ready for streaming")

                elif msg_type == "done" or msg_type == "ended":
                    # Session complete
                    logger.info("Gladia session completed")
                    duration = data.get("data", {}).get("duration", 0)
                    self.meta_info["transcriber_duration"] = duration
                    yield create_ws_data_packet("transcriber_connection_closed", self.meta_info)
                    return

            except Exception as e:
                logger.error(f"Error processing Gladia message: {e}")
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
        gladia_ws = None
        try:
            start_time = timestamp_ms()

            try:
                gladia_ws = await self.gladia_connect()
            except (ValueError, ConnectionError) as e:
                logger.error(f"Failed to establish Gladia connection: {e}")
                await self.toggle_connection()
                return

            if not self.connection_time:
                self.connection_time = round(timestamp_ms() - start_time)

            if self.stream:
                self.sender_task = asyncio.create_task(self.sender_stream(gladia_ws))
                self.heartbeat_task = asyncio.create_task(self.send_heartbeat(gladia_ws))
                self.utterance_timeout_task = asyncio.create_task(self.monitor_utterance_timeout())

                try:
                    async for message in self.receiver(gladia_ws):
                        if self.connection_on:
                            await self.push_to_transcriber_queue(message)
                        else:
                            logger.info("Closing Gladia connection")
                            await self._close_gladia(gladia_ws)
                            break
                except ConnectionClosedError as e:
                    logger.error(f"Gladia WebSocket closed during streaming: {e}")
                except Exception as e:
                    logger.error(f"Error during streaming: {e}")
                    raise
            else:
                # HTTP batch mode - not implemented for Gladia
                # Gladia is primarily a streaming service
                logger.warning("Non-streaming mode not fully supported for Gladia")

        except (ValueError, ConnectionError) as e:
            logger.error(f"Connection error in transcribe: {e}")
            await self.toggle_connection()
        except Exception as e:
            logger.error(f"Unexpected error in transcribe: {e}")
            await self.toggle_connection()
        finally:
            if gladia_ws is not None:
                try:
                    await gladia_ws.close()
                    logger.info("Gladia WebSocket closed in finally block")
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
            await self.push_to_transcriber_queue(
                create_ws_data_packet("transcriber_connection_closed", getattr(self, 'meta_info', {}))
            )
