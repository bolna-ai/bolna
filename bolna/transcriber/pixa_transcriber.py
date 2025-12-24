import asyncio
import json
import os
import time
import traceback
from dotenv import load_dotenv
import websockets
from websockets.asyncio.client import ClientConnection
from websockets.exceptions import ConnectionClosedError, InvalidHandshake, ConnectionClosed

from .base_transcriber import BaseTranscriber
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.utils import create_ws_data_packet, timestamp_ms

load_dotenv()
logger = configure_logger(__name__)


class PixaTranscriber(BaseTranscriber):
    """
    HeyPixa AI STT Transcriber.

    WebSocket URL: wss://transcript.heypixa.ai/v1/listen
    Authentication: X-Pixa-Key: Bearer sk_xxxx

    Key characteristics:
    - No VAD events (SpeechStarted, UtteranceEnd) - relies solely on is_final flag
    - Currently Hindi-only (language parameter accepted but ignored)
    - Supported encodings: linear16, linear32, mulaw, alaw
    - Models: pixa-1 (default), whisper-1
    """

    def __init__(
        self,
        telephony_provider,
        input_queue=None,
        model="pixa-1",
        stream=True,
        language="hi",
        encoding="linear16",
        sampling_rate="16000",
        output_queue=None,
        **kwargs,
    ):
        super().__init__(input_queue)

        # Configuration
        self.telephony_provider = telephony_provider
        self.model = model
        self.language = language  # Hindi only - parameter accepted but ignored by API
        self.stream = stream
        self.encoding = encoding
        self.sampling_rate = int(sampling_rate)

        # API credentials
        self.api_key = kwargs.get("transcriber_key", os.getenv("PIXA_API_KEY"))
        self.ws_host = os.getenv("PIXA_WS_HOST", "transcript.heypixa.ai")

        # Output queue
        self.transcriber_output_queue = output_queue

        # Task handles
        self.transcription_task = None
        self.sender_task = None
        self.heartbeat_task = None
        self.utterance_timeout_task = None

        # State tracking
        self.audio_submitted = False
        self.audio_submission_time = None
        self.num_frames = 0
        self.connection_start_time = None
        self.audio_frame_duration = 0.0

        # Transcript state
        self.final_transcript = ""
        self.websocket_connection = None
        self.connection_authenticated = False
        self.meta_info = {}

        # Turn/latency tracking
        self.current_turn_start_time = None
        self.current_turn_id = None
        self.turn_latencies = []
        self.first_result_latency_ms = None
        self.turn_counter = 0
        self.turn_first_result_latency = None

        # Since Pixa has no VAD, use is_final-based turn detection
        self.is_transcript_sent_for_processing = False
        self.last_interim_time = None
        self.interim_timeout = kwargs.get("interim_timeout", 5.0)  # Default 5 seconds

        # Configure audio params based on telephony provider
        self._configure_audio_params()

    def _configure_audio_params(self):
        """Configure audio parameters based on telephony provider."""
        if self.telephony_provider == "twilio":
            self.encoding = "mulaw"
            self.input_sampling_rate = 8000
            self.sampling_rate = 8000
            self.audio_frame_duration = 0.2
        elif self.telephony_provider in ("plivo", "exotel"):
            self.encoding = "linear16"
            self.input_sampling_rate = 8000
            self.sampling_rate = 8000
            self.audio_frame_duration = 0.2
        elif self.telephony_provider == "web_based_call":
            self.encoding = "linear16"
            self.sampling_rate = 16000
            self.input_sampling_rate = 16000
            self.audio_frame_duration = 0.256
        else:
            # Default configuration
            self.encoding = self.encoding or "linear16"
            self.sampling_rate = int(self.sampling_rate)
            self.input_sampling_rate = self.sampling_rate
            self.audio_frame_duration = 0.2

    def _get_ws_url(self):
        """Construct WebSocket URL with query parameters."""
        params = {
            "language": self.language,
            "encoding": self.encoding,
            "sample_rate": self.sampling_rate,
        }

        # Add model if not default
        if self.model and self.model != "pixa-1":
            params["model"] = self.model

        query_string = "&".join([f"{k}={v}" for k, v in params.items()])
        return f"wss://{self.ws_host}/v1/listen?{query_string}"

    async def pixa_connect(self, retries: int = 3, timeout: float = 10.0) -> ClientConnection:
        """Establish WebSocket connection to Pixa with retry logic."""
        ws_url = self._get_ws_url()
        additional_headers = {
            'X-Pixa-Key': f'Bearer {self.api_key}',
        }

        attempt = 0
        last_err = None

        logger.info(f"Attempting to connect to Pixa WebSocket: {ws_url}")

        while attempt < retries:
            try:
                ws = await asyncio.wait_for(
                    websockets.connect(ws_url, additional_headers=additional_headers),
                    timeout=timeout,
                )
                self.websocket_connection = ws
                self.connection_authenticated = True
                logger.info("Successfully connected to Pixa WebSocket")
                return ws
            except asyncio.TimeoutError:
                logger.error("Timeout while connecting to Pixa websocket")
                raise ConnectionError("Timeout while connecting to Pixa websocket")
            except InvalidHandshake as e:
                error_msg = str(e)
                if '401' in error_msg or '403' in error_msg:
                    logger.error(f"Pixa authentication failed: Invalid or expired API key - {e}")
                    raise ConnectionError(f"Pixa authentication failed: Invalid or expired API key")
                else:
                    logger.error(f"Invalid handshake during Pixa websocket connection: {e}")
                    last_err = e
                    attempt += 1
                    if attempt < retries:
                        await asyncio.sleep(2 ** attempt)
            except Exception as e:
                logger.error(f"Error connecting to Pixa websocket (attempt {attempt + 1}/{retries}): {e}")
                last_err = e
                attempt += 1
                if attempt < retries:
                    await asyncio.sleep(2 ** attempt)

        raise ConnectionError(f"Failed to connect to Pixa after {retries} attempts: {last_err}")

    async def sender_stream(self, ws: ClientConnection):
        """Send audio frames to Pixa WebSocket."""
        try:
            while True:
                ws_data_packet = await self.input_queue.get()
                if ws_data_packet is None:
                    continue

                if not self.audio_submitted:
                    self.meta_info = ws_data_packet.get("meta_info", {})
                    self.audio_submitted = True
                    self.audio_submission_time = time.time()
                    self.current_request_id = self.generate_request_id()
                    self.meta_info["request_id"] = self.current_request_id

                    # Start turn tracking
                    if not self.current_turn_start_time:
                        self.current_turn_start_time = timestamp_ms()
                        self.turn_counter += 1
                        self.current_turn_id = f"turn_{self.turn_counter}"

                # Check for end of stream
                if ws_data_packet.get("meta_info", {}).get("eos") is True:
                    logger.info("Received end of stream signal")
                    # Send Finalize command before closing
                    try:
                        await ws.send(json.dumps({"type": "Finalize"}))
                        await asyncio.sleep(0.5)  # Allow final results to come through
                        await ws.send(json.dumps({"type": "CloseStream"}))
                    except Exception as e:
                        logger.warning(f"Error sending close commands: {e}")
                    break

                self.num_frames += 1

                # Send raw audio bytes directly
                audio_data = ws_data_packet.get("data")
                if audio_data:
                    try:
                        await ws.send(audio_data)
                    except ConnectionClosed as e:
                        logger.error(f"Connection closed while sending audio: {e}")
                        break
                    except Exception as e:
                        logger.error(f"Error sending audio to Pixa: {e}")
                        break

        except asyncio.CancelledError:
            logger.info("Pixa sender stream task cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in Pixa sender_stream: {e}")
            traceback.print_exc()
            raise

    async def receiver(self, ws: ClientConnection):
        """
        Receive and process messages from Pixa WebSocket.

        Important: Pixa does NOT send VAD events (SpeechStarted, UtteranceEnd).
        We rely solely on is_final flag for turn detection.
        """
        try:
            async for message in ws:
                try:
                    data = json.loads(message) if isinstance(message, str) else message

                    if self.connection_start_time is None:
                        self.connection_start_time = time.time() - (self.num_frames * self.audio_frame_duration)

                    if isinstance(data, dict):
                        msg_type = data.get("type")

                        if msg_type == "Metadata":
                            # Connection metadata received on connect
                            logger.info(f"Received Pixa Metadata: {data}")
                            continue

                        elif msg_type == "Results":
                            # Process transcription results
                            # Handle nested structure: results.channels[0].alternatives[0]
                            results = data.get("results", {})
                            channels = results.get("channels", [])

                            transcript = ""
                            is_final = data.get("is_final", False)

                            if channels and len(channels) > 0:
                                alternatives = channels[0].get("alternatives", [])
                                if alternatives and len(alternatives) > 0:
                                    transcript = alternatives[0].get("transcript", "")

                            # Fallback: check direct transcript field
                            if not transcript:
                                transcript = data.get("transcript", "")

                            if transcript and transcript.strip():
                                now_timestamp = time.time()

                                # Track first result latency
                                if self.first_result_latency_ms is None and self.audio_submission_time:
                                    first_result_latency_seconds = now_timestamp - self.audio_submission_time
                                    self.first_result_latency_ms = round(first_result_latency_seconds * 1000)
                                    self.meta_info["transcriber_first_result_latency"] = first_result_latency_seconds
                                    self.meta_info["transcriber_latency"] = first_result_latency_seconds
                                    self.meta_info["first_result_latency_ms"] = self.first_result_latency_ms

                                # Track turn latency
                                if self.current_turn_start_time and self.turn_first_result_latency is None:
                                    turn_latency_ms = timestamp_ms() - self.current_turn_start_time
                                    self.turn_first_result_latency = round(turn_latency_ms)

                                # Update last interim time for timeout monitoring
                                self.last_interim_time = time.time()

                                logger.info(f"Pixa result - is_final: {is_final}, transcript: {transcript}")

                                # Always yield interim transcript
                                yield create_ws_data_packet(
                                    {"type": "interim_transcript_received", "content": transcript.strip()},
                                    self.meta_info,
                                )

                                if is_final:
                                    # Final result - this is our turn boundary since no VAD
                                    self.final_transcript = transcript.strip()
                                    self.meta_info["last_vocal_frame_timestamp"] = now_timestamp

                                    # Build turn latency info
                                    if self.current_turn_start_time:
                                        total_duration_ms = round(timestamp_ms() - self.current_turn_start_time)
                                        turn_info = {
                                            "turn_id": self.current_turn_id,
                                            "sequence_id": self.current_turn_id,
                                            "first_result_latency_ms": self.turn_first_result_latency,
                                            "total_stream_duration_ms": total_duration_ms,
                                        }
                                        self.turn_latencies.append(turn_info)
                                        self.meta_info["turn_latencies"] = self.turn_latencies

                                    # Yield final transcript
                                    yield create_ws_data_packet(
                                        {"type": "transcript", "content": self.final_transcript},
                                        self.meta_info,
                                    )

                                    # Reset turn state for next utterance
                                    self._reset_turn_state()
                                else:
                                    # Accumulate for potential force-finalize
                                    self.final_transcript = transcript.strip()

                        elif msg_type == "error":
                            logger.error(f"Pixa error message: {data}")

                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse Pixa message as JSON: {e}")
                except Exception as e:
                    logger.error(f"Error processing Pixa message: {e}")
                    traceback.print_exc()
        except ConnectionClosed as e:
            logger.info(f"Pixa WebSocket connection closed: {e}")
        except Exception as e:
            logger.error(f"Error in Pixa receiver: {e}")
            traceback.print_exc()

    def _reset_turn_state(self):
        """Reset turn state after finalizing a transcript."""
        self.current_turn_start_time = timestamp_ms()
        self.current_turn_id = f"turn_{self.turn_counter + 1}"
        self.turn_counter += 1
        self.turn_first_result_latency = None
        self.final_transcript = ""
        self.is_transcript_sent_for_processing = True
        self.last_interim_time = None

    async def monitor_utterance_timeout(self):
        """
        Monitor for stuck utterances that never receive is_final=True.
        Since Pixa has no UtteranceEnd event, we need this fallback.
        """
        try:
            while True:
                await asyncio.sleep(1.0)

                if (self.last_interim_time and
                    self.final_transcript.strip() and
                    not self.is_transcript_sent_for_processing):

                    elapsed = time.time() - self.last_interim_time

                    if elapsed > self.interim_timeout:
                        logger.warning(
                            f"Pixa interim timeout: No is_final for {elapsed:.1f}s. "
                            f"Force-finalizing turn {self.current_turn_id}"
                        )
                        await self._force_finalize_utterance()
        except asyncio.CancelledError:
            logger.info("Pixa utterance timeout monitoring task cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in monitor_utterance_timeout: {e}")
            traceback.print_exc()
            raise

    async def _force_finalize_utterance(self):
        """Force-finalize a stuck utterance and send to queue."""
        transcript_to_send = self.final_transcript.strip()

        if not transcript_to_send:
            logger.warning("No transcript available to force-finalize")
            self._reset_turn_state()
            return

        # Build turn latencies
        if self.current_turn_start_time:
            total_duration_ms = round(timestamp_ms() - self.current_turn_start_time)
            turn_info = {
                "turn_id": self.current_turn_id,
                "sequence_id": self.current_turn_id,
                "first_result_latency_ms": self.turn_first_result_latency,
                "total_stream_duration_ms": total_duration_ms,
                "force_finalized": True,
            }
            self.turn_latencies.append(turn_info)
            self.meta_info["turn_latencies"] = self.turn_latencies

        data = {
            "type": "transcript",
            "content": transcript_to_send,
            "force_finalized": True,
        }

        logger.info(f"Force-finalized Pixa transcript after timeout: {transcript_to_send}")

        await self.push_to_transcriber_queue(create_ws_data_packet(data, self.meta_info))

        self._reset_turn_state()

    async def send_heartbeat(self, ws: ClientConnection, interval_sec: float = 10.0):
        """Send keepalive messages to maintain connection."""
        try:
            while True:
                await asyncio.sleep(interval_sec)
                try:
                    await ws.send(json.dumps({"type": "KeepAlive"}))
                    logger.debug("Sent KeepAlive to Pixa")
                except ConnectionClosed:
                    logger.info("Connection closed during heartbeat")
                    break
                except Exception as e:
                    logger.error(f"Error sending heartbeat: {e}")
                    break
        except asyncio.CancelledError:
            logger.info("Pixa heartbeat task cancelled")
            raise

    async def toggle_connection(self):
        """Close connection and cancel all tasks."""
        self.connection_on = False

        for task in [self.sender_task, self.heartbeat_task, self.utterance_timeout_task]:
            if task:
                task.cancel()

        if self.websocket_connection:
            try:
                await self.websocket_connection.close()
                logger.info("Pixa WebSocket connection closed")
            except Exception as e:
                logger.error(f"Error closing Pixa WebSocket: {e}")
            finally:
                self.websocket_connection = None
                self.connection_authenticated = False

    async def cleanup(self):
        """Clean up all resources including websocket."""
        logger.info("Cleaning up Pixa transcriber resources")

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
                    logger.info(f"Pixa {task_name} cancelled")
                except Exception as e:
                    logger.error(f"Error cancelling Pixa {task_name}: {e}")

        # Close websocket
        if self.websocket_connection is not None:
            try:
                await self.websocket_connection.close()
                logger.info("Pixa websocket connection closed")
            except Exception as e:
                logger.error(f"Error closing Pixa websocket: {e}")
            finally:
                self.websocket_connection = None
                self.connection_authenticated = False

    async def push_to_transcriber_queue(self, data_packet):
        """Push data to the output queue."""
        if self.transcriber_output_queue is not None:
            await self.transcriber_output_queue.put(data_packet)

    async def run(self):
        """Start the transcription task."""
        try:
            self.transcription_task = asyncio.create_task(self.transcribe())
        except Exception as e:
            logger.error(f"Error starting Pixa transcription: {e}")
            traceback.print_exc()

    async def transcribe(self):
        """Main transcription loop."""
        pixa_ws = None
        try:
            start_time = time.perf_counter()
            try:
                pixa_ws = await self.pixa_connect()
            except (ValueError, ConnectionError) as e:
                logger.error(f"Failed to connect to Pixa: {e}")
                await self.toggle_connection()
                return

            if not self.connection_time:
                self.connection_time = round((time.perf_counter() - start_time) * 1000)

            if self.stream:
                self.sender_task = asyncio.create_task(self.sender_stream(pixa_ws))
                self.heartbeat_task = asyncio.create_task(self.send_heartbeat(pixa_ws))
                self.utterance_timeout_task = asyncio.create_task(self.monitor_utterance_timeout())

                try:
                    async for message in self.receiver(pixa_ws):
                        if self.connection_on:
                            await self.push_to_transcriber_queue(message)
                        else:
                            logger.info("Closing Pixa connection")
                            try:
                                await pixa_ws.send(json.dumps({"type": "CloseStream"}))
                            except Exception:
                                pass
                            break
                except ConnectionClosedError as e:
                    logger.error(f"Pixa websocket connection closed during streaming: {e}")
                except Exception as e:
                    logger.error(f"Error during Pixa streaming: {e}")
                    traceback.print_exc()

        except (ValueError, ConnectionError) as e:
            logger.error(f"Connection error in Pixa transcribe: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in Pixa transcribe: {e}")
            traceback.print_exc()
        finally:
            # Cleanup tasks
            for task in [self.sender_task, self.heartbeat_task, self.utterance_timeout_task]:
                if task:
                    task.cancel()

            if pixa_ws:
                try:
                    await pixa_ws.close()
                    logger.info("Pixa WebSocket closed in finally block")
                except Exception as e:
                    logger.error(f"Error closing Pixa websocket in finally: {e}")
                finally:
                    self.websocket_connection = None
                    self.connection_authenticated = False

            # Send connection closed notification
            await self.push_to_transcriber_queue(
                create_ws_data_packet("transcriber_connection_closed", getattr(self, 'meta_info', {}))
            )

    def get_meta_info(self):
        """Return current meta info."""
        return getattr(self, "meta_info", {})
