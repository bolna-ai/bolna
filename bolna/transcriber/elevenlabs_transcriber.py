import asyncio
import traceback
import os
import json
import base64
import time
from urllib.parse import urlencode
from dotenv import load_dotenv
import websockets
from websockets.asyncio.client import ClientConnection
from websockets.exceptions import ConnectionClosedError, InvalidHandshake, ConnectionClosed

from .base_transcriber import BaseTranscriber
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.utils import create_ws_data_packet, timestamp_ms


logger = configure_logger(__name__)
load_dotenv()


class ElevenLabsTranscriber(BaseTranscriber):
    def __init__(self, telephony_provider, input_queue=None, model='scribe_v2_realtime', stream=True,
                 language="en", endpointing="400", sampling_rate="16000", encoding="linear16", output_queue=None,
                 commit_strategy="vad", include_timestamps=True,
                 include_language_detection=True, **kwargs):
        super().__init__(input_queue)
        self.endpointing = endpointing
        # Convert endpointing (ms) to vad_silence_threshold_secs (seconds)
        # ElevenLabs requires vad_silence_threshold_secs to be between 0.3 and 3.0
        raw_vad_threshold = int(endpointing) / 1000.0
        self.vad_silence_threshold_secs = max(0.3, min(3.0, raw_vad_threshold))
        self.language = language
        self.stream = stream
        self.provider = telephony_provider
        self.sender_task = None
        self.model = model
        self.sampling_rate = 16000
        self.encoding = encoding
        self.api_key = kwargs.get("transcriber_key", os.getenv('ELEVENLABS_API_KEY'))
        self.elevenlabs_host = os.getenv('ELEVENLABS_API_HOST', 'api.elevenlabs.io')
        self.transcriber_output_queue = output_queue
        self.transcription_task = None
        self.transcription_cursor = 0.0
        self.interruption_signalled = False
        self.audio_submitted = False
        self.audio_submission_time = None
        self.num_frames = 0
        self.connection_start_time = None
        self.audio_frame_duration = 0.0
        self.connected_via_dashboard = kwargs.get("enforce_streaming", True)

        # ElevenLabs specific settings
        self.commit_strategy = commit_strategy
        # Note: self.vad_silence_threshold_secs is set above from endpointing
        self.include_timestamps = include_timestamps
        self.include_language_detection = include_language_detection

        # VAD tuning parameters - balanced for accuracy and latency
        # ElevenLabs valid ranges: vad_threshold (0.1-0.9), min_speech_duration_ms (50-2000), min_silence_duration_ms (50-2000)
        # Defaults tuned for conversational AI with good Hindi/multilingual support
        self.vad_threshold = max(0.1, min(0.9, kwargs.get("vad_threshold", 0.5)))
        self.min_speech_duration_ms = max(50, min(2000, kwargs.get("min_speech_duration_ms", 150)))
        self.min_silence_duration_ms = max(50, min(2000, kwargs.get("min_silence_duration_ms", 300)))

        # Message states
        self.curr_message = ''
        self.finalized_transcript = ""
        self.final_transcript = ""
        self.is_transcript_sent_for_processing = False
        self.current_turn_start_time = None
        self.current_turn_id = None
        self.websocket_connection = None
        self.connection_authenticated = False
        self.speech_start_time = None
        self.speech_end_time = None
        self.current_turn_interim_details = []
        self.audio_frame_timestamps = []
        self.turn_counter = 0

        # Latency tracking
        self.last_audio_send_time = None

        # Timeout tracking for stuck utterances
        self.last_interim_time = None
        self.interim_timeout = kwargs.get("interim_timeout", 5.0)
        self.utterance_timeout_task = None

    def get_elevenlabs_ws_url(self):
        """Build the ElevenLabs WebSocket URL with query parameters"""
        self.audio_frame_duration = 0.5  # Default for 8k samples at 16kHz
        audio_format = 'pcm_16000'  # Default

        if self.provider in ('twilio', 'exotel', 'plivo', 'vobiz'):
            # Twilio uses mulaw at 8kHz, exotel/plivo use linear16 at 8kHz
            self.encoding = 'mulaw' if self.provider == "twilio" else "linear16"
            self.sampling_rate = 8000
            self.audio_frame_duration = 0.2  # 200ms chunks for telephony
            audio_format = 'ulaw_8000' if self.provider == "twilio" else 'pcm_8000'

        elif self.provider == "web_based_call":
            self.encoding = "linear16"
            self.sampling_rate = 16000
            self.audio_frame_duration = 0.256
            audio_format = 'pcm_16000'

        elif not self.connected_via_dashboard:
            self.encoding = "linear16"
            self.sampling_rate = 16000
            audio_format = 'pcm_16000'

        if self.provider == "playground":
            self.sampling_rate = 8000
            self.audio_frame_duration = 0.0  # No streaming from playground
            audio_format = 'pcm_8000'

        params = {
            'model_id': self.model,
            'language_code': self.language,
            'audio_format': audio_format,
            'commit_strategy': self.commit_strategy,
            'vad_silence_threshold_secs': self.vad_silence_threshold_secs,
            # VAD tuning for low latency
            'vad_threshold': self.vad_threshold,
            'min_speech_duration_ms': self.min_speech_duration_ms,
            'min_silence_duration_ms': self.min_silence_duration_ms,
            # Timestamps and language detection
            'include_timestamps': 'true' if self.include_timestamps else 'false',
            'include_language_detection': 'true' if self.include_language_detection else 'false',
        }

        websocket_url = f'wss://{self.elevenlabs_host}/v1/speech-to-text/realtime?{urlencode(params)}'
        logger.info(f"ElevenLabs WebSocket params - language: {self.language}, audio_format: {audio_format}, "
                    f"vad_threshold: {self.vad_threshold}, min_speech_ms: {self.min_speech_duration_ms}, "
                    f"min_silence_ms: {self.min_silence_duration_ms}, vad_silence_secs: {self.vad_silence_threshold_secs}, "
                    f"lang_detection: {self.include_language_detection}")
        return websocket_url

    def _reset_turn_state(self):
        """Reset turn state variables after finalizing a transcript"""
        self.speech_start_time = None
        self.speech_end_time = None
        self.last_interim_time = None
        self.current_turn_interim_details = []
        self.current_turn_start_time = None
        self.current_turn_id = None
        self.final_transcript = ""
        # Set to False to allow next utterance to be processed
        # The flag prevents duplicate processing of same utterance (committed vs committed_with_timestamps)
        # but should not block subsequent utterances
        self.is_transcript_sent_for_processing = False

    def _find_audio_send_timestamp(self, audio_position):
        """
        Find when the audio frame containing this position was sent to ElevenLabs.

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

    async def _force_finalize_utterance(self):
        """Force-finalize a stuck utterance and send to queue"""
        transcript_to_send = self.final_transcript.strip()

        # Fallback: use last interim if no final results received
        if not transcript_to_send and self.current_turn_interim_details:
            transcript_to_send = self.current_turn_interim_details[-1]['transcript']
            logger.info(f"Using last interim as fallback: {transcript_to_send}")

        if not transcript_to_send:
            logger.warning("No transcript available to force-finalize")
            self._reset_turn_state()
            return

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

        logger.info(f"Force-finalized transcript after timeout: {transcript_to_send}")
        await self.push_to_transcriber_queue(create_ws_data_packet(data, self.meta_info))
        self._reset_turn_state()

    async def monitor_utterance_timeout(self):
        """Monitor for stuck utterances that never receive committed transcript"""
        try:
            while True:
                await asyncio.sleep(1.0)

                if (self.last_interim_time and
                    not self.is_transcript_sent_for_processing and
                    (self.final_transcript.strip() or self.current_turn_interim_details)):

                    elapsed = time.time() - self.last_interim_time

                    if elapsed > self.interim_timeout:
                        logger.warning(
                            f"Interim timeout: No finalization for {elapsed:.1f}s. "
                            f"Force-finalizing turn {self.current_turn_id}"
                        )
                        await self._force_finalize_utterance()
        except asyncio.CancelledError:
            logger.info("Utterance timeout monitoring task cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in monitor_utterance_timeout: {e}")
            raise

    async def toggle_connection(self):
        """Close the connection and cancel all tasks"""
        self.connection_on = False
        if self.sender_task is not None:
            self.sender_task.cancel()
        if self.utterance_timeout_task is not None:
            self.utterance_timeout_task.cancel()

        if self.websocket_connection is not None:
            try:
                await self.websocket_connection.close()
                logger.info("Websocket connection closed successfully")
            except Exception as e:
                logger.error(f"Error closing websocket connection: {e}")
            finally:
                self.websocket_connection = None
                self.connection_authenticated = False

    async def cleanup(self):
        """Clean up all resources including websocket."""
        logger.info("Cleaning up ElevenLabs transcriber resources")

        # Cancel tasks properly
        for task_name, task in [
            ("sender_task", getattr(self, 'sender_task', None)),
            ("utterance_timeout_task", getattr(self, 'utterance_timeout_task', None)),
            ("transcription_task", getattr(self, 'transcription_task', None))
        ]:
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    logger.info(f"ElevenLabs {task_name} cancelled")
                except Exception as e:
                    logger.error(f"Error cancelling ElevenLabs {task_name}: {e}")

        # Close websocket
        if self.websocket_connection is not None:
            try:
                await self.websocket_connection.close()
                logger.info("ElevenLabs websocket connection closed")
            except Exception as e:
                logger.error(f"Error closing ElevenLabs websocket: {e}")
            finally:
                self.websocket_connection = None
                self.connection_authenticated = False

    async def _check_and_process_end_of_stream(self, ws_data_packet, ws):
        if 'eos' in ws_data_packet['meta_info'] and ws_data_packet['meta_info']['eos'] is True:
            # ElevenLabs doesn't have a close_stream message, just close the websocket
            try:
                await ws.close()
            except Exception as e:
                logger.debug(f"Error closing websocket on EOS: {e}")
            return True
        return False

    def get_meta_info(self):
        return self.meta_info

    async def sender_stream(self, ws: ClientConnection):
        """Send audio data to ElevenLabs WebSocket"""
        try:
            while True:
                ws_data_packet = await self.input_queue.get()

                # Initialize new request
                if not self.audio_submitted:
                    self.meta_info = ws_data_packet.get('meta_info')
                    self.audio_submitted = True
                    self.audio_submission_time = time.time()
                    self.current_request_id = self.generate_request_id()
                    self.meta_info['request_id'] = self.current_request_id
                    try:
                        if not self.current_turn_start_time:
                            self.current_turn_start_time = timestamp_ms()
                            self.current_turn_id = self.meta_info.get('turn_id') or self.meta_info.get('request_id')
                    except Exception:
                        pass

                end_of_stream = await self._check_and_process_end_of_stream(ws_data_packet, ws)
                if end_of_stream:
                    break

                frame_start = self.num_frames * self.audio_frame_duration
                frame_end = (self.num_frames + 1) * self.audio_frame_duration
                send_timestamp = timestamp_ms()
                self.audio_frame_timestamps.append((frame_start, frame_end, send_timestamp))
                self.num_frames += 1

                try:
                    # Prepare audio message for ElevenLabs
                    audio_data = ws_data_packet.get('data')
                    audio_b64 = base64.b64encode(audio_data).decode('utf-8')

                    message = {
                        "message_type": "input_audio_chunk",
                        "audio_base_64": audio_b64,
                        "sample_rate": self.sampling_rate,
                        "commit": False  # Let VAD handle commits
                    }

                    await ws.send(json.dumps(message))
                    # Track send time for latency calculation
                    self.last_audio_send_time = timestamp_ms()
                except ConnectionClosedError as e:
                    logger.error(f"Connection closed while sending data: {e}")
                    break
                except Exception as e:
                    logger.error(f"Error sending data to websocket: {e}")
                    break

        except asyncio.CancelledError:
            logger.info("Sender stream task cancelled")
            raise
        except Exception as e:
            logger.error(f'Error in sender_stream: {e}')
            raise

    async def receiver(self, ws: ClientConnection):
        """Receive and process messages from ElevenLabs WebSocket"""
        async for msg in ws:
            try:
                msg = json.loads(msg)
                msg_type = msg.get("message_type", "")

                # Initialize connection start time
                if self.connection_start_time is None:
                    self.connection_start_time = (time.time() - (self.num_frames * self.audio_frame_duration))

                if msg_type == "session_started":
                    logger.info(f"ElevenLabs session started: {msg.get('session_id')}")
                    self.connection_authenticated = True
                    continue

                elif msg_type == "partial_transcript":
                    transcript = msg.get("text", "")

                    if transcript.strip():
                        # Start new turn if needed
                        if not self.current_turn_id:
                            self.turn_counter += 1
                            self.current_turn_id = self.turn_counter
                            self.speech_start_time = timestamp_ms()
                            self.current_turn_interim_details = []
                            logger.info(f"Starting new turn with turn_id: {self.current_turn_id}")
                            yield create_ws_data_packet("speech_started", self.meta_info)

                        # Calculate latency from last audio send to transcript receipt
                        latency_ms = None
                        if self.last_audio_send_time:
                            result_received_at = timestamp_ms()
                            latency_ms = round(result_received_at - self.last_audio_send_time, 5)

                        interim_detail = {
                            'transcript': transcript,
                            'is_final': False,
                            'received_at': time.time(),
                            'latency_ms': latency_ms,
                        }

                        logger.info(f"Partial transcript: {transcript} (latency: {latency_ms}ms)")
                        self.current_turn_interim_details.append(interim_detail)
                        self.last_interim_time = time.time()

                        data = {
                            "type": "interim_transcript_received",
                            "content": transcript
                        }
                        yield create_ws_data_packet(data, self.meta_info)

                        # Update final transcript for potential use
                        self.final_transcript = transcript

                        if self.is_transcript_sent_for_processing:
                            self.is_transcript_sent_for_processing = False

                elif msg_type == "committed_transcript":
                    transcript = msg.get("text", "")
                    logger.info(f"Committed transcript: {transcript}")

                    # Skip when include_timestamps is enabled - we'll process committed_transcript_with_timestamps instead
                    # ElevenLabs sends both messages for the same utterance despite docs saying they're mutually exclusive
                    if self.include_timestamps:
                        continue

                    if transcript.strip() and not self.is_transcript_sent_for_processing:
                        data = {
                            "type": "transcript",
                            "content": transcript
                        }

                        try:
                            first_interim_to_final_ms, last_interim_to_final_ms = self.calculate_interim_to_final_latencies(self.current_turn_interim_details)

                            self.turn_latencies.append({
                                'turn_id': self.current_turn_id,
                                'sequence_id': self.current_turn_id,
                                'interim_details': self.current_turn_interim_details,
                                'first_interim_to_final_ms': first_interim_to_final_ms,
                                'last_interim_to_final_ms': last_interim_to_final_ms
                            })

                            # Complete turn reset - set flag to False to allow next utterance
                            self.speech_start_time = None
                            self.speech_end_time = None
                            self.current_turn_interim_details = []
                            self.current_turn_start_time = None
                            self.current_turn_id = None
                            self.final_transcript = ""
                            self.is_transcript_sent_for_processing = False
                        except Exception as e:
                            logger.error(f"Error in committed_transcript handling: {e}")

                        yield create_ws_data_packet(data, self.meta_info)

                elif msg_type == "committed_transcript_with_timestamps":
                    transcript = msg.get("text", "")
                    words = msg.get("words", [])
                    detected_language = msg.get("language_code")
                    logger.info(f"Committed transcript with timestamps: {transcript} ({len(words)} words)")

                    if transcript.strip() and not self.is_transcript_sent_for_processing:
                        data = {
                            "type": "transcript",
                            "content": transcript
                        }

                        try:
                            # Calculate per-word latency using timestamps
                            if words and self.audio_frame_timestamps:
                                for word_obj in words:
                                    if isinstance(word_obj, dict) and 'end' in word_obj:
                                        audio_position = word_obj['end']
                                        audio_sent_at = self._find_audio_send_timestamp(audio_position)
                                        if audio_sent_at:
                                            word_latency = round(timestamp_ms() - audio_sent_at, 5)
                                            word_obj['latency_ms'] = word_latency

                            first_interim_to_final_ms, last_interim_to_final_ms = self.calculate_interim_to_final_latencies(self.current_turn_interim_details)

                            self.turn_latencies.append({
                                'turn_id': self.current_turn_id,
                                'sequence_id': self.current_turn_id,
                                'interim_details': self.current_turn_interim_details,
                                'first_interim_to_final_ms': first_interim_to_final_ms,
                                'last_interim_to_final_ms': last_interim_to_final_ms,
                                'words': words,
                                'detected_language': detected_language
                            })

                            self._reset_turn_state()
                        except Exception as e:
                            logger.error(f"Error in committed_transcript_with_timestamps handling: {e}")

                        yield create_ws_data_packet(data, self.meta_info)

                elif msg_type == "input_error":
                    error_msg = msg.get("error", "Unknown error")
                    logger.warning(f"ElevenLabs input error: {error_msg}")

                elif msg_type == "unaccepted_terms":
                    error_msg = msg.get("error", "Terms not accepted")
                    logger.error(f"ElevenLabs terms not accepted: {error_msg}")
                    logger.error("Please accept terms at https://elevenlabs.io/app/product-terms")
                    break

                elif msg_type == "error":
                    error_msg = msg.get("error", "Unknown error")
                    logger.error(f"ElevenLabs error: {error_msg}")

                else:
                    logger.debug(f"Unknown message type: {msg_type} - {msg}")

            except Exception as e:
                traceback.print_exc()
                logger.error(f"Error processing message: {e}")
                self.interruption_signalled = False

    async def push_to_transcriber_queue(self, data_packet):
        await self.transcriber_output_queue.put(data_packet)

    async def elevenlabs_connect(self):
        """Establish websocket connection to ElevenLabs"""
        try:
            websocket_url = self.get_elevenlabs_ws_url()
            additional_headers = {
                'xi-api-key': self.api_key
            }

            logger.info(f"Attempting to connect to ElevenLabs websocket: {websocket_url}")

            elevenlabs_ws = await asyncio.wait_for(
                websockets.connect(websocket_url, additional_headers=additional_headers),
                timeout=10.0
            )

            self.websocket_connection = elevenlabs_ws
            logger.info("Successfully connected to ElevenLabs websocket")

            return elevenlabs_ws

        except asyncio.TimeoutError:
            logger.error("Timeout while connecting to ElevenLabs websocket")
            raise ConnectionError("Timeout while connecting to ElevenLabs websocket")
        except InvalidHandshake as e:
            logger.error(f"Invalid handshake during ElevenLabs websocket connection: {e}")
            raise ConnectionError(f"Invalid handshake during ElevenLabs websocket connection: {e}")
        except ConnectionClosedError as e:
            logger.error(f"ElevenLabs websocket connection closed unexpectedly: {e}")
            raise ConnectionError(f"ElevenLabs websocket connection closed unexpectedly: {e}")
        except Exception as e:
            logger.error(f"Unexpected error connecting to ElevenLabs websocket: {e}")
            raise ConnectionError(f"Unexpected error connecting to ElevenLabs websocket: {e}")

    async def run(self):
        try:
            self.transcription_task = asyncio.create_task(self.transcribe())
        except Exception as e:
            logger.error(f"Error starting transcription: {e}")

    async def transcribe(self):
        elevenlabs_ws = None
        try:
            start_time = timestamp_ms()
            try:
                elevenlabs_ws = await self.elevenlabs_connect()
            except (ValueError, ConnectionError) as e:
                logger.error(f"Failed to establish ElevenLabs connection: {e}")
                await self.toggle_connection()
                return

            if not self.connection_time:
                self.connection_time = round(timestamp_ms() - start_time)

            if self.stream:
                self.sender_task = asyncio.create_task(self.sender_stream(elevenlabs_ws))
                self.utterance_timeout_task = asyncio.create_task(self.monitor_utterance_timeout())
                try:
                    async for message in self.receiver(elevenlabs_ws):
                        if self.connection_on:
                            await self.push_to_transcriber_queue(message)
                        else:
                            logger.info("Closing the ElevenLabs connection")
                            break
                except ConnectionClosedError as e:
                    logger.error(f"ElevenLabs websocket connection closed during streaming: {e}")
                except Exception as e:
                    logger.error(f"Error during streaming: {e}")
                    raise

        except (ValueError, ConnectionError) as e:
            logger.error(f"Connection error in transcribe: {e}")
            await self.toggle_connection()
        except Exception as e:
            logger.error(f"Unexpected error in transcribe: {e}")
            await self.toggle_connection()
        finally:
            if elevenlabs_ws is not None:
                try:
                    await elevenlabs_ws.close()
                    logger.info("ElevenLabs websocket closed in finally block")
                except Exception as e:
                    logger.error(f"Error closing websocket in finally block: {e}")
                finally:
                    self.websocket_connection = None
                    self.connection_authenticated = False

            if hasattr(self, 'sender_task') and self.sender_task is not None:
                self.sender_task.cancel()
            if hasattr(self, 'utterance_timeout_task') and self.utterance_timeout_task is not None:
                self.utterance_timeout_task.cancel()

            await self.push_to_transcriber_queue(
                create_ws_data_packet("transcriber_connection_closed", getattr(self, 'meta_info', {}))
            )
