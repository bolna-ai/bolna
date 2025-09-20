import asyncio
import traceback
import os
import json
import aiohttp
import time
from audioop import ulaw2lin
from urllib.parse import urlencode
from dotenv import load_dotenv
import websockets
from websockets.asyncio.client import ClientConnection
from websockets.exceptions import ConnectionClosedError, InvalidHandshake

from .base_transcriber import BaseTranscriber
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.utils import create_ws_data_packet

logger = configure_logger(__name__)
load_dotenv()


class AssemblyAITranscriber(BaseTranscriber):
    def __init__(self, telephony_provider, input_queue=None, model='universal-streaming', stream=True, language="en",
                 sampling_rate="16000", encoding="pcm_s16le", output_queue=None, format_turns=True,
                 **kwargs):
        super().__init__(input_queue)
        self.language = language
        self.stream = stream
        self.provider = telephony_provider
        self.heartbeat_task = None
        self.sender_task = None
        self.model = model
        self.sampling_rate = int(sampling_rate)
        self.encoding = encoding
        self.format_turns = format_turns
        
        self.api_key = kwargs.get("transcriber_key", os.getenv('ASSEMBLY_API_KEY'))
        self.assemblyai_host = "streaming.assemblyai.com"
        self.transcriber_output_queue = output_queue
        self.transcription_task = None
        
        # Audio and transcription tracking
        self.audio_cursor = 0.0
        self.transcription_cursor = 0.0
        self.interruption_signalled = False
        
        if not self.stream:
            # For non-streaming HTTP API
            self.api_url = f"https://api.assemblyai.com/v2/transcript"
            self.session = aiohttp.ClientSession()
            
        self.audio_submitted = False
        self.audio_submission_time = None
        self.num_frames = 0
        self.connection_start_time = None
        self.audio_frame_duration = 0.0
        self.connected_via_dashboard = kwargs.get("enforce_streaming", True)
        
        # Message states for turn management
        self.session_id = None
        self.current_transcript = ""
        self.is_transcript_sent_for_processing = False
        self.websocket_connection = None
        self.connection_authenticated = False
        self.current_turn_start_time = None
        self.current_turn_id = None

    def get_assemblyai_ws_url(self):
        """Get the AssemblyAI WebSocket URL with appropriate parameters"""
        # Connection parameters for v3 API
        connection_params = {
            "sample_rate": self.sampling_rate,
            "format_turns": self.format_turns
        }
        
        if self.provider in ('twilio', 'exotel', 'plivo'):
            self.encoding = 'mulaw' if self.provider in ("twilio") else "linear16"
            self.sampling_rate = 8000
            self.audio_frame_duration = 0.2
            connection_params['sample_rate'] = self.sampling_rate
            
        elif self.provider == "web_based_call":
            self.encoding = "linear16"
            self.sampling_rate = 16000
            self.audio_frame_duration = 0.256
            connection_params['sample_rate'] = self.sampling_rate
            
        elif not self.connected_via_dashboard:
            self.encoding = "linear16"
            connection_params['sample_rate'] = 16000
            
        if self.provider == "playground":
            self.sampling_rate = 8000
            self.audio_frame_duration = 0.0
            
        if self.language != "en":
            logger.warning("AssemblyAI Universal Streaming currently only supports English")
            
        # Build WebSocket URL
        websocket_url = f"wss://{self.assemblyai_host}/v3/ws?{urlencode(connection_params)}"
        return websocket_url

    async def send_heartbeat(self, ws: ClientConnection):
        """Send periodic keepalive messages"""
        try:
            while True:
                # AssemblyAI v3 doesn't require explicit heartbeat, but we can send periodic pings
                try:
                    await ws.ping()
                except ConnectionClosedError as e:
                    logger.info(f"Connection closed while sending ping: {e}")
                    break
                except Exception as e:
                    logger.error(f"Error sending ping: {e}")
                    break
                    
                await asyncio.sleep(5)  # Send ping every 5 seconds
                
        except asyncio.CancelledError:
            logger.info("Heartbeat task cancelled")
            raise
        except Exception as e:
            logger.error(f'Error in send_heartbeat: {e}')
            raise

    async def toggle_connection(self):
        """Close the connection and cleanup tasks"""
        self.connection_on = False
        
        if self.heartbeat_task is not None:
            self.heartbeat_task.cancel()
        if self.sender_task is not None:
            self.sender_task.cancel()
        
        if self.websocket_connection is not None:
            try:
                # Send session termination message for v3 API
                termination_msg = {"type": "Terminate"}
                await self.websocket_connection.send(json.dumps(termination_msg))
                await self.websocket_connection.close()
                logger.info("AssemblyAI WebSocket connection closed successfully")
            except Exception as e:
                logger.error(f"Error closing websocket connection: {e}")
            finally:
                self.websocket_connection = None
                self.connection_authenticated = False

    async def _get_http_transcription(self, audio_data):
        """Handle non-streaming HTTP transcription (for non-streaming mode)"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()

        # First upload the audio file
        headers = {
            'Authorization': self.api_key,
        }

        self.current_request_id = self.generate_request_id()
        self.meta_info['request_id'] = self.current_request_id
        start_time = time.time()
        
        # Upload audio
        upload_url = "https://api.assemblyai.com/v2/upload"
        async with self.session.post(upload_url, data=audio_data, headers=headers) as response:
            if response.status != 200:
                raise Exception(f"Failed to upload audio: {response.status}")
            upload_response = await response.json()
            audio_url = upload_response['upload_url']

        # Submit transcription request
        transcript_request = {
            'audio_url': audio_url,
            'language_code': self.language if self.language != 'en' else None
        }
        
        headers['Content-Type'] = 'application/json'
        async with self.session.post(self.api_url, json=transcript_request, headers=headers) as response:
            if response.status != 200:
                raise Exception(f"Failed to submit transcription: {response.status}")
            transcript_response = await response.json()
            transcript_id = transcript_response['id']

        # Poll for completion
        while True:
            async with self.session.get(f"{self.api_url}/{transcript_id}", headers=headers) as response:
                result = await response.json()
                if result['status'] == 'completed':
                    transcript = result['text'] or ""
                    self.meta_info["start_time"] = start_time
                    self.meta_info['transcriber_latency'] = time.time() - start_time
                    self.meta_info['transcriber_duration'] = result.get('audio_duration', 0)
                    return create_ws_data_packet(transcript, self.meta_info)
                elif result['status'] == 'error':
                    raise Exception(f"Transcription failed: {result.get('error')}")
                await asyncio.sleep(1)

    async def _check_and_process_end_of_stream(self, ws_data_packet, ws):
        """Check for end of stream signal"""
        if 'eos' in ws_data_packet['meta_info'] and ws_data_packet['meta_info']['eos'] is True:
            termination_msg = {"type": "Terminate"}
            await ws.send(json.dumps(termination_msg))
            return True
        return False

    def get_meta_info(self):
        return self.meta_info

    async def sender(self, ws=None):
        """Sender for non-streaming mode"""
        try:
            while True:
                ws_data_packet = await self.input_queue.get()
                
                if not self.audio_submitted:
                    self.audio_submitted = True
                    self.audio_submission_time = time.time()
                    
                end_of_stream = await self._check_and_process_end_of_stream(ws_data_packet, ws)
                if end_of_stream:
                    break
                    
                if ws_data_packet is not None:
                    self.meta_info = ws_data_packet.get('meta_info', {})
                    start_time = time.perf_counter()
                    transcription = await self._get_http_transcription(ws_data_packet.get('data'))
                    transcription['meta_info']["include_latency"] = True
                    # HTTP path: first result and total duration are the same
                    elapsed = time.perf_counter() - start_time
                    transcription['meta_info']["transcriber_first_result_latency"] = elapsed
                    transcription['meta_info']["transcriber_total_stream_duration"] = elapsed
                    transcription['meta_info']["transcriber_latency"] = elapsed
                    transcription['meta_info']['audio_duration'] = transcription['meta_info']['transcriber_duration']
                    transcription['meta_info']['last_vocal_frame_timestamp'] = time.time()
                    yield transcription

            if self.transcription_task is not None:
                self.transcription_task.cancel()
                
        except asyncio.CancelledError:
            logger.info("Cancelled sender task")
            return

    async def sender_stream(self, ws: ClientConnection):
        """Send audio data to AssemblyAI WebSocket"""
        try:
            while True:
                ws_data_packet = await self.input_queue.get()
                
                # Initialize new request
                if not self.audio_submitted:
                    if ws_data_packet is not None:
                        self.meta_info = ws_data_packet.get('meta_info', {}) or {}
                        self.audio_submitted = True
                        self.audio_submission_time = time.time()
                        self.current_request_id = self.generate_request_id()
                        self.meta_info['request_id'] = self.current_request_id
                        try:
                            if not self.current_turn_start_time:
                                self.current_turn_start_time = time.perf_counter()
                                meta = self.meta_info or {}
                                self.current_turn_id = meta.get('turn_id') or meta.get('request_id')
                        except Exception:
                            pass

                end_of_stream = await self._check_and_process_end_of_stream(ws_data_packet, ws)
                if end_of_stream:
                    break
                    
                self.num_frames += 1
                self.audio_cursor = self.num_frames * self.audio_frame_duration
                
                audio_data = ws_data_packet.get('data')
                if isinstance(audio_data, bytes):
                    try:
                        if self.provider == "twilio" and self.encoding == "mulaw":
                            audio_data = ulaw2lin(audio_data, 2)
                        
                        await ws.send(audio_data)
                    except ConnectionClosedError as e:
                        logger.error(f"Connection closed while sending data: {e}")
                        break
                    except Exception as e:
                        logger.error(f"Error sending data to websocket: {e}")
                        break
                else:
                    logger.warning(f"Expected bytes for audio data, got: {type(audio_data)}")
                    
        except asyncio.CancelledError:
            logger.info("Sender stream task cancelled")
            raise
        except Exception as e:
            logger.error(f'Error in sender_stream: {e}')
            raise

    async def receiver(self, ws: ClientConnection):
        """Receive and process messages from AssemblyAI WebSocket"""
        async for msg in ws:
            try:
                msg = json.loads(msg)

                if self.connection_start_time is None:
                    self.connection_start_time = time.time() - (self.num_frames * self.audio_frame_duration)

                message_type = msg.get("type")

                if message_type == "Begin":
                    logger.info("AssemblyAI session began")
                    self.session_id = msg.get("id")
                    expires_at = msg.get("expires_at")
                    logger.info(f"Session ID: {self.session_id}, Expires at: {expires_at}")
                    yield create_ws_data_packet("session_started", self.meta_info)

                elif message_type == "Turn":
                    # Handle transcription turn
                    transcript = msg.get("transcript", "").strip()
                    turn_is_formatted = msg.get("turn_is_formatted", False)
                    
                    if transcript:
                        if turn_is_formatted:
                            # This is a final, formatted transcript
                            logger.info(f"Received formatted transcript: {transcript}")
                            data = {
                                "type": "transcript",
                                "content": transcript
                            }
                            # Total stream duration at final
                            try:
                                if self.current_turn_start_time is not None:
                                    total_stream_duration = time.perf_counter() - self.current_turn_start_time
                                    self.meta_info['transcriber_total_stream_duration'] = total_stream_duration
                                    self.meta_info['transcriber_latency'] = total_stream_duration
                                    # Append to turn_latencies for analytics
                                    self.turn_latencies.append({
                                        'turn_id': self.current_turn_id,
                                        'sequence_id': self.current_turn_id,
                                        'first_result_latency_ms': round(((self.meta_info or {}).get('transcriber_first_result_latency', 0)) * 1000),
                                        'total_stream_duration_ms': round(total_stream_duration * 1000)
                                    })
                                    # Reset turn tracking
                                    self.current_turn_start_time = None
                                    self.current_turn_id = None
                            except Exception:
                                pass
                            yield create_ws_data_packet(data, self.meta_info)
                        else:
                            # This is an interim/partial transcript
                            logger.debug(f"Received interim transcript: {transcript}")
                            data = {
                                "type": "interim_transcript_received", 
                                "content": transcript
                            }
                            # First actionable interim → first result latency
                            try:
                                if self.current_turn_start_time is not None and 'transcriber_first_result_latency' not in (self.meta_info or {}):
                                    first_result_latency = time.perf_counter() - self.current_turn_start_time
                                    if self.meta_info is None:
                                        self.meta_info = {}
                                    self.meta_info['transcriber_first_result_latency'] = first_result_latency
                                    self.meta_info['transcriber_latency'] = first_result_latency
                            except Exception:
                                pass
                            yield create_ws_data_packet(data, self.meta_info)

                elif message_type == "Termination":
                    logger.info("AssemblyAI session terminated")
                    audio_duration = msg.get("audio_duration_seconds", 0)
                    session_duration = msg.get("session_duration_seconds", 0)
                    logger.info(f"Audio duration: {audio_duration}s, Session duration: {session_duration}s")
                    yield create_ws_data_packet("transcriber_connection_closed", self.meta_info)
                    return

                elif message_type == "Error":
                    error_msg = msg.get("error", "Unknown error")
                    logger.error(f"AssemblyAI error: {error_msg}")
                    yield create_ws_data_packet("transcriber_error", self.meta_info)

                else:
                    logger.debug(f"Received unknown message type: {message_type}")

            except Exception as e:
                logger.error(f"Error processing message: {e}")
                traceback.print_exc()

    async def push_to_transcriber_queue(self, data_packet):
        """Push data to the output queue"""
        if self.transcriber_output_queue is not None:
            await self.transcriber_output_queue.put(data_packet)

    async def assemblyai_connect(self):
        """Establish WebSocket connection to AssemblyAI"""
        try:
            websocket_url = self.get_assemblyai_ws_url()
            logger.info(f"Attempting to connect to AssemblyAI websocket: {websocket_url}")
            
            # Set up headers with API key authentication
            headers = {
                'Authorization': self.api_key
            }
            
            # Connect to WebSocket with headers
            assemblyai_ws = await asyncio.wait_for(
                websockets.connect(websocket_url, additional_headers=headers),
                timeout=10.0
            )
            
            self.websocket_connection = assemblyai_ws
            self.connection_authenticated = True
            logger.info("Successfully connected to AssemblyAI websocket")
            
            return assemblyai_ws
            
        except asyncio.TimeoutError:
            logger.error("Timeout while connecting to AssemblyAI websocket")
            raise ConnectionError("Timeout while connecting to AssemblyAI websocket")
        except InvalidHandshake as e:
            logger.error(f"Invalid handshake during AssemblyAI websocket connection: {e}")
            raise ConnectionError(f"Invalid handshake during AssemblyAI websocket connection: {e}")
        except ConnectionClosedError as e:
            logger.error(f"AssemblyAI websocket connection closed unexpectedly: {e}")
            raise ConnectionError(f"AssemblyAI websocket connection closed unexpectedly: {e}")
        except Exception as e:
            logger.error(f"Unexpected error connecting to AssemblyAI websocket: {e}")
            raise ConnectionError(f"Unexpected error connecting to AssemblyAI websocket: {e}")

    async def run(self):
        """Start the transcription task"""
        try:
            self.transcription_task = asyncio.create_task(self.transcribe())
        except Exception as e:
            logger.error(f"Error starting transcription task: {e}")

    async def transcribe(self):
        """Main transcription method"""
        assemblyai_ws = None
        try:
            start_time = time.perf_counter()
            
            try:
                assemblyai_ws = await self.assemblyai_connect()
            except (ValueError, ConnectionError) as e:
                logger.error(f"Failed to establish AssemblyAI connection: {e}")
                await self.toggle_connection()
                return
            
            if not self.connection_time:
                self.connection_time = round((time.perf_counter() - start_time) * 1000)

            if self.stream:
                self.sender_task = asyncio.create_task(self.sender_stream(assemblyai_ws))
                self.heartbeat_task = asyncio.create_task(self.send_heartbeat(assemblyai_ws))
                
                try:
                    async for message in self.receiver(assemblyai_ws):
                        if self.connection_on:
                            await self.push_to_transcriber_queue(message)
                        else:
                            logger.info("Closing the AssemblyAI connection")
                            termination_msg = {"type": "Terminate"}
                            await assemblyai_ws.send(json.dumps(termination_msg))
                            break
                except ConnectionClosedError as e:
                    logger.error(f"AssemblyAI websocket connection closed during streaming: {e}")
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
            if assemblyai_ws is not None:
                try:
                    await assemblyai_ws.close()
                    logger.info("AssemblyAI websocket closed in finally block")
                except Exception as e:
                    logger.error(f"Error closing websocket in finally block: {e}")
                finally:
                    self.websocket_connection = None
                    self.connection_authenticated = False
            
            if hasattr(self, 'sender_task') and self.sender_task is not None:
                self.sender_task.cancel()
            if hasattr(self, 'heartbeat_task') and self.heartbeat_task is not None:
                self.heartbeat_task.cancel()
            
            await self.push_to_transcriber_queue(
                create_ws_data_packet("transcriber_connection_closed", getattr(self, 'meta_info', {}))
            )