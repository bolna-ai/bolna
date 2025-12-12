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
                 language="en", sampling_rate="16000", encoding="linear16", output_queue=None,
                 commit_strategy="vad", vad_silence_threshold_secs=1.0, include_timestamps=False,
                 include_language_detection=False, **kwargs):
        super().__init__(input_queue)
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
        self.vad_silence_threshold_secs = vad_silence_threshold_secs
        self.include_timestamps = include_timestamps
        self.include_language_detection = include_language_detection

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

        # Timeout tracking for stuck utterances
        self.last_interim_time = None
        self.interim_timeout = kwargs.get("interim_timeout", 5.0)
        self.utterance_timeout_task = None

    def get_elevenlabs_ws_url(self):
        """Build the ElevenLabs WebSocket URL with query parameters"""
        params = {
            'model': self.model,
        }

        self.audio_frame_duration = 0.5  # Default for 8k samples at 16kHz

        if self.provider in ('twilio', 'exotel', 'plivo'):
            # Twilio uses mulaw at 8kHz, exotel/plivo use linear16 at 8kHz
            self.encoding = 'mulaw' if self.provider == "twilio" else "linear16"
            self.sampling_rate = 8000
            self.audio_frame_duration = 0.2  # 200ms chunks for telephony

        elif self.provider == "web_based_call":
            self.encoding = "linear16"
            self.sampling_rate = 16000
            self.audio_frame_duration = 0.256

        elif not self.connected_via_dashboard:
            self.encoding = "linear16"
            self.sampling_rate = 16000

        if self.provider == "playground":
            self.sampling_rate = 8000
            self.audio_frame_duration = 0.0  # No streaming from playground

        websocket_url = f'wss://{self.elevenlabs_host}/v1/speech-to-text/realtime?{urlencode(params)}'
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
        self.is_transcript_sent_for_processing = True

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

                        interim_detail = {
                            'transcript': transcript,
                            'is_final': False,
                            'received_at': time.time(),
                        }

                        logger.info(f"Partial transcript: {transcript}")
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

                    if transcript.strip() and not self.is_transcript_sent_for_processing:
                        data = {
                            "type": "transcript",
                            "content": transcript
                        }

                        try:
                            self.turn_latencies.append({
                                'turn_id': self.current_turn_id,
                                'sequence_id': self.current_turn_id,
                                'interim_details': self.current_turn_interim_details
                            })

                            # Complete turn reset
                            self.speech_start_time = None
                            self.speech_end_time = None
                            self.current_turn_interim_details = []
                            self.current_turn_start_time = None
                            self.current_turn_id = None
                            self.final_transcript = ""
                            self.is_transcript_sent_for_processing = True
                        except Exception as e:
                            logger.error(f"Error in committed_transcript handling: {e}")

                        yield create_ws_data_packet(data, self.meta_info)

                elif msg_type == "committed_transcript_with_timestamps":
                    transcript = msg.get("text", "")
                    words = msg.get("words", [])
                    logger.info(f"Committed transcript with timestamps: {transcript} ({len(words)} words)")

                    if transcript.strip() and not self.is_transcript_sent_for_processing:
                        data = {
                            "type": "transcript",
                            "content": transcript
                        }

                        try:
                            self.turn_latencies.append({
                                'turn_id': self.current_turn_id,
                                'sequence_id': self.current_turn_id,
                                'interim_details': self.current_turn_interim_details,
                                'words': words
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
