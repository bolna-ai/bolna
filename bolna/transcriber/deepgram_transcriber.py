import asyncio
import traceback
import os
import json
import aiohttp
import time
from urllib.parse import urlencode
from dotenv import load_dotenv
import websockets
from websockets.asyncio.client import ClientConnection
from websockets.exceptions import ConnectionClosedError, InvalidHandshake, ConnectionClosed

from .base_transcriber import BaseTranscriber
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.utils import create_ws_data_packet, timestamp_ms
from bolna.enums import TelephonyProvider


logger = configure_logger(__name__)
load_dotenv()

class DeepgramTranscriber(BaseTranscriber):
    def __init__(self, telephony_provider, input_queue=None, model='nova-2', stream=True, language="en", endpointing="400",
                 sampling_rate="16000", encoding="linear16", output_queue=None, keywords=None,
                 process_interim_results="true", **kwargs):
        super().__init__(input_queue)
        self.endpointing = endpointing
        self.language = language
        self.stream = stream
        self.provider = telephony_provider
        self.heartbeat_task = None
        self.sender_task = None
        self.model = model
        self.sampling_rate = int(sampling_rate) if isinstance(sampling_rate, (str, int)) else 16000
        self.encoding = encoding
        self.api_key = kwargs.get("transcriber_key", os.getenv('DEEPGRAM_AUTH_TOKEN'))
        self.deepgram_host = os.getenv('DEEPGRAM_HOST', 'api.deepgram.com')
        self.transcriber_output_queue = output_queue
        self.transcription_task = None
        self.keywords = keywords
        self.transcription_cursor = 0.0
        self.interruption_signalled = False
        if not self.stream:
            self.api_url = f"https://{self.deepgram_host}/v1/listen?model={self.model}&filler_words=true&language={self.language}"
            self.session = aiohttp.ClientSession()
            if self.keywords is not None:
                keyword_string = "&keywords=" + "&keywords=".join(self.keywords.split(","))
                self.api_url = f"{self.api_url}{keyword_string}"
        self.audio_submitted = False
        self.audio_submission_time = None
        self.num_frames = 0
        self.connection_start_time = None
        self.process_interim_results = process_interim_results
        self.audio_frame_duration = 0.0
        self.connected_via_dashboard = kwargs.get("enforce_streaming", True)
        #Message states
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
        self.audio_frame_timestamps = []  # List of (frame_start, frame_end, send_timestamp)
        self.turn_counter = 0
        # Timeout tracking for stuck utterances
        self.last_interim_time = None
        self.interim_timeout = kwargs.get("interim_timeout", 5.0)  # Default 5 seconds
        self.utterance_timeout_task = None

    def get_deepgram_ws_url(self):
        dg_params = {
            'model': self.model,
            'filler_words': 'true',
            # 'diarize': 'true',
            'language': self.language,
            'vad_events' : 'true',
            'endpointing': self.endpointing,
            'interim_results': 'true',
            'utterance_end_ms': '1000' if int(self.endpointing) < 1000 else str(self.endpointing)
        }

        self.audio_frame_duration = 0.5  # We're sending 8k samples with a sample rate of 16k

        if self.provider in TelephonyProvider.telephony_values():
            # For sip-trunk (Asterisk), encoding and sampling_rate are already set in task_manager
            # Don't override them - use what was passed from task_config
            if self.provider != TelephonyProvider.SIP_TRUNK.value:
                self.encoding = 'mulaw' if self.provider in ("twilio") else "linear16"
                self.sampling_rate = 8000
            # For sip-trunk, encoding and sampling_rate come from task_config (set in task_manager)
            # They're already set from the __init__ parameters, so we don't override
            self.audio_frame_duration = 0.2  # 200ms chunks for telephony

            dg_params['encoding'] = self.encoding
            dg_params['sample_rate'] = self.sampling_rate
            dg_params['channels'] = "1"

            if self.provider == TelephonyProvider.SIP_TRUNK.value:
                logger.info(f"[SIP-TRUNK] Deepgram transcriber configured with encoding={self.encoding}, sample_rate={self.sampling_rate}")

        elif self.provider == "web_based_call":
            dg_params['encoding'] = "linear16"
            dg_params['sample_rate'] = 16000
            dg_params['channels'] = "1"
            self.sampling_rate = 16000
            # TODO what is the purpose of this?
            self.audio_frame_duration = 0.256

        elif not self.connected_via_dashboard:
            dg_params['encoding'] = "linear16"
            dg_params['sample_rate'] = 16000
            dg_params['channels'] = "1"

        if self.provider == "playground":
            self.sampling_rate = 8000
            self.audio_frame_duration = 0.0  # There's no streaming from the playground

        if "en" not in self.language:
            dg_params['language'] = self.language

        if self.keywords and len(self.keywords.split(",")) > 0:
            if self.model.startswith('nova-3'):
                dg_params['keyterm'] = "&keyterm=".join(self.keywords.split(","))
                if self.language != 'en':
                    dg_params.pop('keyterm', None)
            else:
                dg_params['keywords'] = "&keywords=".join(self.keywords.split(","))

        protocol = os.getenv('DEEPGRAM_HOST_PROTOCOL', 'wss')
        websocket_api = '{}://{}/v1/listen?'.format(protocol, self.deepgram_host)
        
        websocket_url = websocket_api + urlencode(dg_params)
        return websocket_url

    async def send_heartbeat(self, ws: ClientConnection):
        try:
            while True:
                data = {'type': 'KeepAlive'}
                try:
                    await ws.send(json.dumps(data))
                except ConnectionClosed as e:
                    rcvd_code = getattr(e.rcvd, "code", None)
                    sent_code = getattr(e.sent, "code", None)

                    if rcvd_code == 1000 or sent_code == 1000:
                        logger.info("WebSocket closed normally (1000 OK) during heartbeat.")
                    else:
                        logger.warning(
                            f"WebSocket closed: received={rcvd_code}, sent={sent_code}, "
                            f"reason={getattr(e.rcvd, 'reason', '') or getattr(e.sent, 'reason', '')}"
                        )
                    break
                except Exception as e:
                    logger.error(f"Error sending heartbeat: {e}")
                    break
                    
                await asyncio.sleep(5)  # Send a heartbeat message every 5 seconds
        except asyncio.CancelledError:
            logger.info("Heartbeat task cancelled")
            raise
        except Exception as e:
            logger.error('Error in send_heartbeat: ' + str(e))
            raise

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

        # Determine what transcript to use
        transcript_to_send = self.final_transcript.strip()

        # Fallback: use last interim if no is_final results received
        if not transcript_to_send and self.current_turn_interim_details:
            transcript_to_send = self.current_turn_interim_details[-1]['transcript']
            logger.info(f"Using last interim as fallback: {transcript_to_send}")

        if not transcript_to_send:
            logger.warning("No transcript available to force-finalize")
            self._reset_turn_state()
            return

        # Build turn latencies (same as UtteranceEnd logic)
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

        # Create transcript message (same format as UtteranceEnd)
        data = {
            "type": "transcript",
            "content": transcript_to_send,
            "force_finalized": True  # For debugging
        }

        logger.info(f"Force-finalized transcript after timeout: {transcript_to_send}")

        # Send to queue (unblocks _listen_transcriber)
        await self.push_to_transcriber_queue(create_ws_data_packet(data, self.meta_info))

        # Reset state (same as normal UtteranceEnd)
        self._reset_turn_state()

    async def monitor_utterance_timeout(self):
        """Monitor for stuck utterances that never receive UtteranceEnd"""
        try:
            while True:
                await asyncio.sleep(1.0)

                # Check if we have pending interim results without finalization
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
        self.connection_on = False
        if self.heartbeat_task is not None:
            self.heartbeat_task.cancel()
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
        """Clean up all resources including HTTP session and websocket."""
        logger.info("Cleaning up Deepgram transcriber resources")

        # Close HTTP session (for non-streaming mode)
        if hasattr(self, 'session') and self.session and not self.session.closed:
            try:
                await self.session.close()
                logger.info("Deepgram HTTP session closed")
            except Exception as e:
                logger.error(f"Error closing Deepgram HTTP session: {e}")

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
                    logger.info(f"Deepgram {task_name} cancelled")
                except Exception as e:
                    logger.error(f"Error cancelling Deepgram {task_name}: {e}")

        # Close websocket
        if self.websocket_connection is not None:
            try:
                await self.websocket_connection.close()
                logger.info("Deepgram websocket connection closed")
            except Exception as e:
                logger.error(f"Error closing Deepgram websocket: {e}")
            finally:
                self.websocket_connection = None
                self.connection_authenticated = False

    async def _get_http_transcription(self, audio_data):
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()

        headers = {
            'Authorization': 'Token {}'.format(self.api_key),
            'Content-Type': 'audio/webm'  # Currently we are assuming this is via browser
        }

        self.current_request_id = self.generate_request_id()
        self.meta_info['request_id'] = self.current_request_id
        async with self.session as session:
            async with session.post(self.api_url, data=audio_data, headers=headers) as response:
                response_data = await response.json()
                transcript = response_data["results"]["channels"][0]["alternatives"][0]["transcript"]
                self.meta_info['transcriber_duration'] = response_data["metadata"]["duration"]
                return create_ws_data_packet(transcript, self.meta_info)

    async def _check_and_process_end_of_stream(self, ws_data_packet, ws):
        if 'eos' in ws_data_packet['meta_info'] and ws_data_packet['meta_info']['eos'] is True:
            await self._close(ws, data={"type": "CloseStream"})
            return True  # Indicates end of processing

        return False

    def get_meta_info(self):
        return self.meta_info

    async def sender(self, ws=None):
        try:
            while True:
                ws_data_packet = await self.input_queue.get()
                # If audio submitted was false, that means that we're starting the stream now. That's our stream start
                if not self.audio_submitted:
                    self.audio_submitted = True
                    self.audio_submission_time = time.time()
                    # Mark per-turn start (monotonic)
                    try:
                        self.meta_info = ws_data_packet.get('meta_info') if self.meta_info is None else self.meta_info
                        if self.meta_info is not None and not self.current_turn_start_time:
                            self.current_turn_start_time = timestamp_ms()
                            self.current_turn_id = self.meta_info.get('turn_id') or self.meta_info.get('request_id')
                    except Exception:
                        pass
                end_of_stream = await self._check_and_process_end_of_stream(ws_data_packet, ws)
                if end_of_stream:
                    break
                self.meta_info = ws_data_packet.get('meta_info')
                start_time = timestamp_ms()
                transcription = await self._get_http_transcription(ws_data_packet.get('data'))
                transcription['meta_info']["include_latency"] = True
                # HTTP path: first and total are same
                try:
                    elapsed = timestamp_ms() - start_time
                    transcription['meta_info']["transcriber_first_result_latency"] = elapsed
                    transcription['meta_info']["transcriber_total_stream_duration"] = elapsed
                    transcription['meta_info']["transcriber_latency"] = elapsed
                except Exception:
                    pass
                transcription['meta_info']['audio_duration'] = transcription['meta_info']['transcriber_duration']
                transcription['meta_info']['last_vocal_frame_timestamp'] = time.time()
                yield transcription

            if self.transcription_task is not None:
                self.transcription_task.cancel()
        except asyncio.CancelledError:
            logger.info("Cancelled sender task")
            return

    async def sender_stream(self, ws: ClientConnection):
        try:
            while True:
                ws_data_packet = await self.input_queue.get()
                # Initialise new request
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
                    await ws.send(ws_data_packet.get('data'))
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
            logger.error('Error in sender_stream: ' + str(e))
            raise

    async def receiver(self, ws: ClientConnection):
        async for msg in ws:
            try:
                msg = json.loads(msg)

                # If connection_start_time is None, it is the durations of frame submitted till now minus current time
                if self.connection_start_time is None:
                    self.connection_start_time = (time.time() - (self.num_frames * self.audio_frame_duration))

                if msg["type"] == "SpeechStarted":
                    logger.info("Received SpeechStarted event from deepgram")
                    self.turn_counter += 1
                    self.current_turn_id = self.turn_counter
                    self.speech_start_time = timestamp_ms()
                    self.current_turn_interim_details = []

                    logger.info(f"Starting new turn with turn_id: {self.current_turn_id}")
                    yield create_ws_data_packet("speech_started", self.meta_info)
                    pass

                elif msg["type"] == "Results":
                    transcript = msg["channel"]["alternatives"][0]["transcript"]
                    deepgram_request_id = msg.get("metadata", {}).get("request_id")

                    if transcript.strip():
                        # Calculate latency using end position (start + duration) for cumulative transcripts
                        self.__set_transcription_cursor(msg)
                        audio_position_end = self.transcription_cursor
                        latency_ms = None

                        audio_sent_at = self._find_audio_send_timestamp(audio_position_end)
                        if audio_sent_at:
                            result_received_at = timestamp_ms()
                            latency_ms = round(result_received_at - audio_sent_at, 5)

                        interim_detail = {
                            'transcript': transcript,
                            'latency_ms': latency_ms,
                            'is_final': msg.get('is_final', False),
                            'received_at': time.time(),
                            'request_id': deepgram_request_id
                        }

                        logger.info(f"Interim result - request_id: {deepgram_request_id}, is_final: {msg.get('is_final', False)}, transcript: {transcript}")

                        self.current_turn_interim_details.append(interim_detail)
                        # Track time of last interim for timeout monitoring
                        self.last_interim_time = time.time()

                        data = {
                            "type": "interim_transcript_received",
                            "content": transcript
                        }
                        yield create_ws_data_packet(data, self.meta_info)

                    if msg["is_final"] and transcript.strip():
                        logger.info(f"Received interim result with is_final set as True - {transcript}")
                        self.final_transcript += f' {transcript}'

                        if self.is_transcript_sent_for_processing:
                            self.is_transcript_sent_for_processing = False

                    if msg["speech_final"] and self.final_transcript.strip():
                        if not self.is_transcript_sent_for_processing and self.final_transcript.strip():
                            logger.info(f"Received speech final hence yielding the following transcript - {self.final_transcript}")

                            data = {
                                "type": "transcript",
                                "content": self.final_transcript
                            }

                            # Build turn_latencies with new metrics before resetting
                            try:
                                first_interim_to_final_ms, last_interim_to_final_ms = self.calculate_interim_to_final_latencies(self.current_turn_interim_details)

                                self.turn_latencies.append({
                                    'turn_id': self.current_turn_id,
                                    'sequence_id': self.current_turn_id,
                                    'interim_details': self.current_turn_interim_details,
                                    'first_interim_to_final_ms': first_interim_to_final_ms,
                                    'last_interim_to_final_ms': last_interim_to_final_ms
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
                                logger.error(f"Failed to extract transcript from Deepgram response in speech_final: {e}")
                                pass
                            yield create_ws_data_packet(data, self.meta_info)

                elif msg["type"] == "UtteranceEnd":
                    logger.info(f"Value of is_transcript_sent_for_processing in utterance end - {self.is_transcript_sent_for_processing}")
                    if not self.is_transcript_sent_for_processing and self.final_transcript.strip():
                        logger.info(f"Received UtteranceEnd hence yielding the following transcript - {self.final_transcript}")

                        data = {
                            "type": "transcript",
                            "content": self.final_transcript
                        }

                        # Build turn_latencies with new metrics before resetting
                        try:
                            first_interim_to_final_ms, last_interim_to_final_ms = self.calculate_interim_to_final_latencies(self.current_turn_interim_details)

                            self.turn_latencies.append({
                                'turn_id': self.current_turn_id,
                                'sequence_id': self.current_turn_id,
                                'interim_details': self.current_turn_interim_details,
                                'first_interim_to_final_ms': first_interim_to_final_ms,
                                'last_interim_to_final_ms': last_interim_to_final_ms
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
                            logger.error(f"Failed to extract transcript from Deepgram response: {e}")
                            pass
                        yield create_ws_data_packet(data, self.meta_info)
                    else:
                        # Transcript already sent but we still need to notify speech ended
                        # This prevents callee_speaking from staying True indefinitely
                        logger.info(f"UtteranceEnd received but transcript already processed, yielding speech_ended notification")
                        self.speech_start_time = None
                        self.speech_end_time = None
                        self.current_turn_interim_details = []
                        self.current_turn_start_time = None
                        self.current_turn_id = None
                        self.final_transcript = ""
                        yield create_ws_data_packet({"type": "speech_ended"}, self.meta_info)

                elif msg["type"] == "Metadata":
                    # Capture duration from final Metadata message (actual audio processed by Deepgram)
                    deepgram_duration = msg.get("duration")
                    if deepgram_duration is not None:
                        self.meta_info["deepgram_duration"] = deepgram_duration
                        logger.info(f"Received Deepgram Metadata with duration: {deepgram_duration}s")

            except Exception as e:
                traceback.print_exc()
                self.interruption_signalled = False

    async def push_to_transcriber_queue(self, data_packet):
        await self.transcriber_output_queue.put(data_packet)

    async def deepgram_connect(self):
        """Establish websocket connection to Deepgram with proper error handling"""
        try:
            websocket_url = self.get_deepgram_ws_url()
            additional_headers = {
                'Authorization': 'Token {}'.format(self.api_key)
            }
            
            logger.info(f"Attempting to connect to Deepgram websocket: {websocket_url}")
            
            deepgram_ws = await asyncio.wait_for(
                websockets.connect(websocket_url, additional_headers=additional_headers),
                timeout=10.0  # 10 second timeout
            )
            
            self.websocket_connection = deepgram_ws
            self.connection_authenticated = True
            logger.info("Successfully connected to Deepgram websocket")
            
            return deepgram_ws
            
        except asyncio.TimeoutError:
            logger.error("Timeout while connecting to Deepgram websocket")
            raise ConnectionError("Timeout while connecting to Deepgram websocket")
        except InvalidHandshake as e:
            logger.error(f"Invalid handshake during Deepgram websocket connection: {e}")
            raise ConnectionError(f"Invalid handshake during Deepgram websocket connection: {e}")
        except ConnectionClosedError as e:
            logger.error(f"Deepgram websocket connection closed unexpectedly: {e}")
            raise ConnectionError(f"Deepgram websocket connection closed unexpectedly: {e}")
        except Exception as e:
            logger.error(f"Unexpected error connecting to Deepgram websocket: {e}")
            raise ConnectionError(f"Unexpected error connecting to Deepgram websocket: {e}")

    async def run(self):
        try:
            self.transcription_task = asyncio.create_task(self.transcribe())
        except Exception as e:
            logger.error(f"not working {e}")

    def __calculate_utterance_end(self, data):
        utterance_end = None
        if 'channel' in data and 'alternatives' in data['channel']:
            for alternative in data['channel']['alternatives']:
                if 'words' in alternative:
                    final_word = alternative['words'][-1]
                    utterance_end = self.connection_start_time + final_word['end']
                    logger.info(f"Final word ended at {utterance_end}")
        return utterance_end

    def __set_transcription_cursor(self, data):
        if 'start' in data and 'duration' in data:
            self.transcription_cursor = data['start'] + data['duration']
            logger.info(f"Setting transcription cursor at {self.transcription_cursor} (start={data['start']}, duration={data['duration']})")
        else:
            logger.warning(f"Missing start or duration in Deepgram message, cannot update transcription cursor")
        return self.transcription_cursor

    def _find_audio_send_timestamp(self, audio_position):
        """
        Find when the audio frame containing this position was sent to Deepgram.

        This directly matches the audio position to the frame that contains it,
        providing accurate latency measurement from when that specific audio was sent.

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

    async def transcribe(self):
        deepgram_ws = None
        try:
            start_time = timestamp_ms()
            try:
                deepgram_ws = await self.deepgram_connect()
            except (ValueError, ConnectionError) as e:
                logger.error(f"Failed to establish Deepgram connection: {e}")
                await self.toggle_connection()
                return

            if not self.connection_time:
                self.connection_time = round(timestamp_ms() - start_time)

            if self.stream:
                self.sender_task = asyncio.create_task(self.sender_stream(deepgram_ws))
                self.heartbeat_task = asyncio.create_task(self.send_heartbeat(deepgram_ws))
                self.utterance_timeout_task = asyncio.create_task(self.monitor_utterance_timeout())
                try:
                    async for message in self.receiver(deepgram_ws):
                        if self.connection_on:
                            await self.push_to_transcriber_queue(message)
                        else:
                            logger.info("closing the deepgram connection, waiting for Metadata")
                            await self._close(deepgram_ws, data={"type": "CloseStream"})
                            try:
                                async with asyncio.timeout(5):
                                    async for _ in self.receiver(deepgram_ws):
                                        if "deepgram_duration" in self.meta_info:
                                            break
                            except asyncio.TimeoutError:
                                logger.warning("Timeout waiting for Deepgram Metadata after CloseStream")
                            break
                except ConnectionClosedError as e:
                    logger.error(f"Deepgram websocket connection closed during streaming: {e}")
                except Exception as e:
                    logger.error(f"Error during streaming: {e}")
                    raise
            else:
                async for message in self.sender():
                    await self.push_to_transcriber_queue(message)

        except (ValueError, ConnectionError) as e:
            logger.error(f"Connection error in transcribe: {e}")
            await self.toggle_connection()
        except Exception as e:
            logger.error(f"Unexpected error in transcribe: {e}")
            await self.toggle_connection()
        finally:
            if deepgram_ws is not None:
                try:
                    await deepgram_ws.close()
                    logger.info("Deepgram websocket closed in finally block")
                except Exception as e:
                    logger.error(f"Error closing websocket in finally block: {e}")
                finally:
                    self.websocket_connection = None
                    self.connection_authenticated = False
            
            if hasattr(self, 'sender_task') and self.sender_task is not None:
                self.sender_task.cancel()
            if hasattr(self, 'heartbeat_task') and self.heartbeat_task is not None:
                self.heartbeat_task.cancel()
            if hasattr(self, 'utterance_timeout_task') and self.utterance_timeout_task is not None:
                self.utterance_timeout_task.cancel()

            # Use Deepgram's actual audio duration for billing
            if "deepgram_duration" in self.meta_info:
                self.meta_info["transcriber_duration"] = self.meta_info["deepgram_duration"]

            await self.push_to_transcriber_queue(
                create_ws_data_packet("transcriber_connection_closed", getattr(self, 'meta_info', {}))
            )
