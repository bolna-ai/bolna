import asyncio
import traceback
import os
import json
import time
from urllib.parse import urlencode
from dotenv import load_dotenv
import websockets
from websockets.asyncio.client import ClientConnection
from websockets.exceptions import ConnectionClosedError, InvalidHandshake
import aiohttp
from audioop import ulaw2lin
import io
import wave

from .base_transcriber import BaseTranscriber
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.utils import create_ws_data_packet

logger = configure_logger(__name__)
load_dotenv()


class AssemblyTranscriber(BaseTranscriber):
    def __init__(self, telephony_provider, input_queue=None, model='slam-1', stream=True, language="en", 
                 sampling_rate="16000", encoding="linear16", output_queue=None, keywords=None,
                 format_turns=True, endpointing=500, **kwargs):
        super().__init__(input_queue)
        self.language = language
        self.stream = stream
        self.provider = telephony_provider
        self.heartbeat_task = None
        self.sender_task = None
        self.model = model
        self.sampling_rate = 16000
        self.encoding = encoding
        self.api_key = kwargs.get("transcriber_key", os.getenv('ASSEMBLY_AI_API_KEY'))
        self.assembly_host = os.getenv('ASSEMBLY_AI_HOST', 'streaming.assemblyai.com')
        self.assembly_api_host = os.getenv('ASSEMBLY_AI_API_HOST', 'api.assemblyai.com')
        self.transcriber_output_queue = output_queue
        self.transcription_task = None
        self.keywords = keywords
        self.format_turns = format_turns
        self.endpointing = endpointing
        self.audio_cursor = 0.0
        self.transcription_cursor = 0.0
        self.interruption_signalled = False
        
        self.audio_submitted = False
        self.audio_submission_time = None
        self.num_frames = 0
        self.connection_start_time = None
        self.audio_frame_duration = 0.0
        self.connected_via_dashboard = kwargs.get("enforce_streaming", True)
        
        # Message states
        self.curr_message = ''
        self.finalized_transcript = ""
        self.final_transcript = ""
        self.is_transcript_sent_for_processing = False
        self.websocket_connection = None
        self.connection_authenticated = False
        self.session_id = None
        
        # HTTP API support
        self.session = None
        self.api_url = None
        
        # Additional connection state tracking
        self.connection_on = True  # Ensure this is explicitly set
        self.connection_established = False  # Track if connection was ever established
        self.cleanup_completed = False  # Track cleanup state
        
        
        # Set model based on streaming mode
        if self.stream:
            self.model = 'universal-streaming'
        else:
            # For HTTP API, use Universal model by default (nova-2 is not supported in HTTP API)
            # Universal model is the default, slam-1 is also available
            self.model = 'slam-1' if model == 'nova-2' else (model if model else None)  # None means use default Universal
            
        # Initialize HTTP API URL for non-streaming mode
        if not self.stream:
            self.api_url = f"https://{self.assembly_api_host}/v2/transcript"
            self.upload_url = f"https://{self.assembly_api_host}/v2/upload"
            self.session = aiohttp.ClientSession()

    def get_assembly_ws_url(self):
        """Build AssemblyAI WebSocket URL with appropriate parameters"""
        assembly_params = {
            'sample_rate': self.sampling_rate,
            'format_turns': str(self.format_turns).lower()
        }
        
        # Add model parameter for streaming
        if self.stream:
            assembly_params['model'] = self.model

        # Set audio frame duration based on provider
        self.audio_frame_duration = 0.5  # Default for 16kHz samples

        if self.provider in ('twilio', 'exotel', 'plivo'):
            # Telephony providers typically use 8kHz
            self.sampling_rate = 8000
            self.audio_frame_duration = 0.2  # 200ms frames for telephony
            assembly_params['sample_rate'] = self.sampling_rate
            # AssemblyAI expects pcm_mulaw format for telephony providers
            assembly_params['encoding'] = 'pcm_mulaw'
            
        elif self.provider == "web_based_call":
            self.sampling_rate = 16000
            self.audio_frame_duration = 0.256
            assembly_params['sample_rate'] = self.sampling_rate
            
        elif not self.connected_via_dashboard:
            self.sampling_rate = 16000
            assembly_params['sample_rate'] = self.sampling_rate

        if self.provider == "playground":
            self.sampling_rate = 8000
            self.audio_frame_duration = 0.0
            assembly_params['sample_rate'] = self.sampling_rate

        # Add language support if not English
        if self.language and self.language != 'en':
            # AssemblyAI uses different language codes, map common ones
            language_mapping = {
                'es': 'es',
                'fr': 'fr', 
                'de': 'de',
                'it': 'it',
                'pt': 'pt',
                'hi': 'hi',
                'ja': 'ja',
                'ko': 'ko',
                'zh': 'zh'
            }
            if self.language in language_mapping:
                assembly_params['language_code'] = language_mapping[self.language]

        websocket_api = f'wss://{self.assembly_host}/v3/ws?'
        websocket_url = websocket_api + urlencode(assembly_params)
        return websocket_url

    def _prepare_audio_for_upload(self, audio_data):
        """Convert raw audio data to WAV format for AssemblyAI upload"""
        if self.provider in ('twilio', 'exotel', 'plivo'):
            # Convert μ-law to linear PCM for telephony providers
            audio_data = ulaw2lin(audio_data, 2)  # 2 bytes per sample
        
        # Create WAV file in memory
        wav_buffer = io.BytesIO()
        wav_file = wave.open(wav_buffer, 'wb')
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 2 bytes per sample (16-bit)
        wav_file.setframerate(self.sampling_rate)
        wav_file.writeframes(audio_data)
        wav_file.close()
        
        wav_buffer.seek(0)
        return wav_buffer.getvalue()

    async def _get_http_transcription(self, audio_data):
        """Process audio using AssemblyAI HTTP API"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()

        # Convert raw audio data to WAV format
        wav_data = self._prepare_audio_for_upload(audio_data)
        
        headers = {
            'Authorization': self.api_key,
            'Content-Type': 'audio/wav'
        }

        self.current_request_id = self.generate_request_id()
        self.meta_info['request_id'] = self.current_request_id
        start_time = time.time()
        
        try:
            # First upload the audio data
            async with self.session.post(self.upload_url, data=wav_data, headers=headers) as upload_response:
                if upload_response.status != 200:
                    error_text = await upload_response.text()
                    logger.error("Error uploading audio to AssemblyAI: %s - %s", upload_response.status, error_text)
                    raise ConnectionError(f"Audio upload failed: {upload_response.status}")
                
                upload_json = await upload_response.json()
                upload_url = upload_json["upload_url"]
            
            # Create transcription request with minimal required parameters
            transcript_data = {
                "audio_url": upload_url
            }
            
            # Add optional parameters only if they have valid values
            if self.language and self.language != 'en':
                transcript_data["language_code"] = self.language
                
            if self.keywords:
                transcript_data["word_boost"] = self.keywords.split(',')
                
            # Note: end_utterance_silence_threshold might not be supported in v2 API
            # if self.endpointing and self.endpointing > 0:
            #     transcript_data["end_utterance_silence_threshold"] = self.endpointing / 1000.0
                
            if self.model:
                transcript_data["speech_model"] = self.model
            
            logger.info(f"AssemblyAI transcription request data: {transcript_data}")
            
            async with self.session.post(self.api_url, json=transcript_data, headers={'Authorization': self.api_key}) as transcript_response:
                if transcript_response.status != 200:
                    error_text = await transcript_response.text()
                    logger.error("Error creating transcription: %s - %s", transcript_response.status, error_text)
                    raise ConnectionError(f"Transcription creation failed: {transcript_response.status}")
                
                transcript_json = await transcript_response.json()
                transcript_id = transcript_json["id"]
                
                # Poll for completion
                polling_endpoint = f"https://{self.assembly_api_host}/v2/transcript/{transcript_id}"
                max_attempts = 20  # 1 minute max (20 * 3 seconds)
                attempt = 0
                
                logger.info(f"Starting to poll transcription status for ID: {transcript_id}")
                
                while attempt < max_attempts:
                    await asyncio.sleep(3)  # Wait 3 seconds between polls
                    attempt += 1
                    
                    logger.info(f"Polling attempt {attempt}/{max_attempts} for transcription {transcript_id}")
                    
                    async with self.session.get(polling_endpoint, headers={'Authorization': self.api_key}) as status_response:
                        if status_response.status != 200:
                            error_text = await status_response.text()
                            logger.error("Error polling transcription status: %s - %s", status_response.status, error_text)
                            break
                        
                        status_json = await status_response.json()
                        logger.info(f"Transcription status: {status_json['status']}")
                        
                        if status_json["status"] == "completed":
                            transcript_text = status_json["text"]
                            self.meta_info["start_time"] = start_time
                            self.meta_info['transcriber_latency'] = time.time() - start_time
                            self.meta_info['transcriber_duration'] = status_json.get("audio_duration", 0)
                            logger.info("AssemblyAI HTTP transcription completed: %s", transcript_text)
                            
                            # Format the message the same way as streaming transcriber
                            data = {
                                "type": "transcript",
                                "content": transcript_text.strip()
                            }
                            return create_ws_data_packet(data, self.meta_info)
                        
                        elif status_json["status"] == "error":
                            error_msg = status_json.get("error", "Unknown error")
                            logger.error("AssemblyAI transcription failed: %s", error_msg)
                            raise ConnectionError(f"Transcription failed: {error_msg}")
                        
                        # Still processing, continue polling
                        logger.debug("AssemblyAI transcription still processing, attempt %d/%d", attempt, max_attempts)
                
                # If we get here, we've exceeded max attempts
                raise ConnectionError("Transcription polling timeout")
                
        except Exception as e:
            logger.error("Error in AssemblyAI HTTP transcription: %s", e)
            raise

    async def send_heartbeat(self, ws: ClientConnection):
        """Send heartbeat messages to keep connection alive"""
        try:
            while True:
                data = {'type': 'KeepAlive'}
                try:
                    await ws.send(json.dumps(data))
                except ConnectionClosedError as e:
                    logger.info("AssemblyAI connection closed while sending heartbeat: %s", e)
                    break
                except Exception as e:
                    logger.error("Error sending AssemblyAI heartbeat: %s", e)
                    break
                    
                await asyncio.sleep(5)  # Send a heartbeat message every 5 seconds
        except asyncio.CancelledError:
            logger.info("AssemblyAI heartbeat task cancelled")
            raise
        except Exception as e:
            logger.error('Error in AssemblyAI send_heartbeat: %s', e)
            raise

    async def toggle_connection(self):
        """Close the WebSocket connection and cleanup tasks"""
        self.connection_on = False
        
        if self.heartbeat_task is not None:
            self.heartbeat_task.cancel()
        if self.sender_task is not None:
            self.sender_task.cancel()
        
        if self.websocket_connection is not None:
            try:
                # Send termination message before closing
                terminate_message = {"type": "Terminate"}
                await self.websocket_connection.send(json.dumps(terminate_message))
                await asyncio.sleep(0.1)  # Brief pause for message to send
                
                await self.websocket_connection.close()
                logger.info("AssemblyAI websocket connection closed successfully")
            except Exception as e:
                logger.error("Error closing AssemblyAI websocket connection: %s", e)
            finally:
                self.websocket_connection = None
                self.connection_authenticated = False
                self.session_id = None
        else:
            pass
            
        self.cleanup_completed = True  # Mark cleanup as completed

    async def _check_and_process_end_of_stream(self, ws_data_packet, ws):
        """Check for end of stream signal and handle cleanup"""
        if 'eos' in ws_data_packet['meta_info'] and ws_data_packet['meta_info']['eos'] is True:
            terminate_message = {"type": "Terminate"}
            try:
                await ws.send(json.dumps(terminate_message))
            except Exception as e:
                pass
            return True  # Indicates end of processing
        return False

    def get_meta_info(self):
        """Return current metadata information"""
        return self.meta_info

    async def sender_stream(self, ws: ClientConnection):
        """Send audio data to AssemblyAI WebSocket"""
        try:
            while True:
                ws_data_packet = await self.input_queue.get()
                
                # Initialize new request on first audio submission
                if not self.audio_submitted:
                    self.meta_info = ws_data_packet.get('meta_info')
                    self.audio_submitted = True
                    self.audio_submission_time = time.time()
                    self.current_request_id = self.generate_request_id()
                    self.meta_info['request_id'] = self.current_request_id

                # Check for end of stream
                end_of_stream = await self._check_and_process_end_of_stream(ws_data_packet, ws)
                if end_of_stream:
                    break
                    
                self.num_frames += 1
                # Update audio cursor for latency tracking
                self.audio_cursor = self.num_frames * self.audio_frame_duration
                
                try:
                    # Send raw audio data as binary message
                    data_to_send = ws_data_packet.get('data')
                    
                    if data_to_send and isinstance(data_to_send, (bytes, bytearray)):
                        # AssemblyAI can handle μ-law audio directly for telephony providers
                        await ws.send(data_to_send)
                except ConnectionClosedError as e:
                    logger.error("AssemblyAI connection closed while sending data: %s", e)
                    break
                except Exception as e:
                    logger.error("Error sending data to AssemblyAI websocket: %s", e)
                    # Don't break the loop for data validation errors, just skip this chunk
                    if "object of type 'NoneType' has no len" in str(e):
                        continue
                    else:
                        break
                    
        except asyncio.CancelledError:
            logger.info("AssemblyAI sender stream task cancelled")
            raise
        except Exception as e:
            logger.error('Error in AssemblyAI sender_stream: %s', e)
            raise

    async def receiver(self, ws: ClientConnection):
        """Receive and process messages from AssemblyAI WebSocket"""
        async for msg in ws:
            try:
                msg_data = json.loads(msg)
                msg_type = msg_data.get('type')

                # Set connection start time on first message
                if self.connection_start_time is None:
                    self.connection_start_time = (time.time() - (self.num_frames * self.audio_frame_duration))

                if msg_type == "Begin":
                    # Session started
                    self.session_id = msg_data.get('id')
                    expires_at = msg_data.get('expires_at')
                    logger.info("AssemblyAI session began: ID=%s, ExpiresAt=%s", self.session_id, expires_at)
                    self.connection_established = True  # Mark connection as established
                    
                    # Signal speech detection started
                    yield create_ws_data_packet("speech_started", self.meta_info)

                elif msg_type == "Turn":
                    transcript = msg_data.get('transcript', '')
                    is_formatted = msg_data.get('turn_is_formatted', False)
                    words = msg_data.get('words', [])

                    if transcript.strip():
                        # Update transcription cursor for latency tracking
                        if words:
                            self.__set_transcription_cursor({'words': words})
                        
                        if is_formatted:
                            # This is a final transcript
                            logger.info("Received formatted transcript from AssemblyAI: %s", transcript)
                            data = {
                                "type": "transcript",
                                "content": transcript.strip()
                            }
                            # Calculate and add latency info
                            latency = self.__calculate_latency()
                            if latency is not None:
                                self.meta_info['transcriber_latency'] = latency
                            yield create_ws_data_packet(data, self.meta_info)
                        else:
                            # This is an interim result
                            data = {
                                "type": "interim_transcript_received", 
                                "content": transcript
                            }
                            # Calculate and add latency info for interim results
                            latency = self.__calculate_latency()
                            if latency is not None:
                                self.meta_info['transcriber_latency'] = latency
                            yield create_ws_data_packet(data, self.meta_info)

                elif msg_type == "Termination":
                    # Session terminated
                    audio_duration = msg_data.get('audio_duration_seconds', 0)
                    session_duration = msg_data.get('session_duration_seconds', 0)
                    logger.info("AssemblyAI session terminated: Audio Duration=%ss, Session Duration=%ss", audio_duration, session_duration)
                    
                    # Update metadata with duration info
                    self.meta_info["transcriber_duration"] = audio_duration
                    yield create_ws_data_packet("transcriber_connection_closed", self.meta_info)
                    return

                elif msg_type == "Error":
                    # Handle errors from AssemblyAI
                    error_code = msg_data.get('error_code')
                    error_message = msg_data.get('error_message', 'Unknown error')
                    logger.error("AssemblyAI error: %s - %s", error_code, error_message)
                    
                else:
                    logger.debug("Received unknown message type from AssemblyAI: %s", msg_type)

            except json.JSONDecodeError as e:
                logger.error("Error decoding AssemblyAI message: %s", e)
            except Exception as e:
                logger.error("Error handling AssemblyAI message: %s", e)
                traceback.print_exc()

    async def sender(self, ws=None):
        """Send audio data for HTTP API processing"""
        try:
            chunk_duration = 3.0  # Process audio in 3-second chunks
            chunk_size = int(self.sampling_rate * chunk_duration * 2)  # 2 bytes per sample
            current_chunk = io.BytesIO()
            chunk_start_time = time.time()
            min_chunk_size = int(self.sampling_rate * 0.5 * 2)  # Minimum 0.5 seconds of audio
            
            while True:
                ws_data_packet = await self.input_queue.get()
                # If audio submitted was false, that means that we're starting the stream now. That's our stream start
                if not self.audio_submitted:
                    self.audio_submitted = True
                    self.audio_submission_time = time.time()
                    self.meta_info = ws_data_packet.get('meta_info')
                
                end_of_stream = await self._check_and_process_end_of_stream(ws_data_packet, ws)
                
                # Accumulate audio data
                audio_data = ws_data_packet.get('data')
                if audio_data:
                    current_chunk.write(audio_data)
                
                # Process chunk if it's large enough or if stream ended
                should_process = (
                    end_of_stream or 
                    (current_chunk.tell() >= chunk_size and current_chunk.tell() >= min_chunk_size) or
                    (time.time() - chunk_start_time) >= chunk_duration
                )
                
                if should_process and current_chunk.tell() >= min_chunk_size:
                    current_chunk.seek(0)
                    chunk_audio = current_chunk.getvalue()
                    current_chunk.close()
                    current_chunk = io.BytesIO()
                    chunk_start_time = time.time()
                    
                    logger.info(f"Processing audio chunk of size: {len(chunk_audio)} bytes")
                    start_time = time.time()
                    
                    try:
                        # Add timeout to prevent hanging
                        transcription = await asyncio.wait_for(
                            self._get_http_transcription(chunk_audio), 
                            timeout=30.0  # 30 second timeout
                        )
                        transcription['meta_info']["include_latency"] = True
                        transcription['meta_info']["transcriber_latency"] = time.time() - start_time
                        transcription['meta_info']['audio_duration'] = transcription['meta_info']['transcriber_duration']
                        transcription['meta_info']['last_vocal_frame_timestamp'] = start_time
                        logger.info(f"Transcription result: {transcription}")
                        logger.info(f"Yielding transcription with content: '{transcription['data']}'")
                        yield transcription
                    except asyncio.TimeoutError:
                        logger.error(f"Transcription timed out after 30 seconds for chunk of size {len(chunk_audio)}")
                        # Yield a fallback transcription to keep the system running
                        fallback_transcription = create_ws_data_packet("", self.meta_info)
                        fallback_transcription['meta_info']["include_latency"] = True
                        fallback_transcription['meta_info']["transcriber_latency"] = time.time() - start_time
                        fallback_transcription['meta_info']['audio_duration'] = 0
                        fallback_transcription['meta_info']['last_vocal_frame_timestamp'] = start_time
                        yield fallback_transcription
                    except Exception as e:
                        logger.error(f"Transcription failed for chunk: {e}")
                        # Yield a fallback transcription to keep the system running
                        fallback_transcription = create_ws_data_packet("", self.meta_info)
                        fallback_transcription['meta_info']["include_latency"] = True
                        fallback_transcription['meta_info']["transcriber_latency"] = time.time() - start_time
                        fallback_transcription['meta_info']['audio_duration'] = 0
                        fallback_transcription['meta_info']['last_vocal_frame_timestamp'] = start_time
                        yield fallback_transcription
                
                if end_of_stream:
                    break

            if self.transcription_task is not None:
                self.transcription_task.cancel()
        except asyncio.CancelledError:
            logger.info("Cancelled AssemblyAI sender task")
            return
        except Exception as e:
            logger.error('Error in AssemblyAI sender: %s', e)
            raise

    async def push_to_transcriber_queue(self, data_packet):
        """Push processed data to the output queue"""
        if self.transcriber_output_queue is None:
            return
        try:
            await self.transcriber_output_queue.put(data_packet)
        except Exception as e:
            pass

    async def assembly_connect(self):
        """Establish WebSocket connection to AssemblyAI with proper error handling"""
        try:
            websocket_url = self.get_assembly_ws_url()
            additional_headers = {
                'Authorization': self.api_key
            }
            
            logger.info("Attempting to connect to AssemblyAI websocket: %s", websocket_url)
            
            assembly_ws = await asyncio.wait_for(
                websockets.connect(websocket_url, additional_headers=additional_headers),
                timeout=10.0  # 10 second timeout
            )
            
            self.websocket_connection = assembly_ws
            self.connection_authenticated = True
            logger.info("Successfully connected to AssemblyAI websocket")
            
            return assembly_ws
            
        except asyncio.TimeoutError:
            logger.error("Timeout while connecting to AssemblyAI websocket")
            raise ConnectionError("Timeout while connecting to AssemblyAI websocket") from None
        except InvalidHandshake as e:
            logger.error("Invalid handshake during AssemblyAI websocket connection: %s", e)
            raise ConnectionError(f"Invalid handshake during AssemblyAI websocket connection: {e}") from e
        except ConnectionClosedError as e:
            logger.error("AssemblyAI websocket connection closed unexpectedly: %s", e)
            raise ConnectionError(f"AssemblyAI websocket connection closed unexpectedly: {e}") from e
        except Exception as e:
            logger.error("Unexpected error connecting to AssemblyAI websocket: %s", e)
            raise ConnectionError(f"Unexpected error connecting to AssemblyAI websocket: {e}") from e


    def __calculate_utterance_end(self, data):
        """Calculate utterance end timing based on response data"""
        utterance_end = None
        if 'channel' in data and 'alternatives' in data['channel']:
            for alternative in data['channel']['alternatives']:
                if 'words' in alternative:
                    final_word = alternative['words'][-1]
                    utterance_end = self.connection_start_time + final_word['end']
                    logger.info("Final word ended at %s", utterance_end)
        return utterance_end

    def __set_transcription_cursor(self, data):
        """Set transcription cursor based on response data"""
        if 'words' in data:
            final_word = data['words'][-1]
            self.transcription_cursor = final_word['end']
        logger.info("Setting transcription cursor at %s", self.transcription_cursor)
        return self.transcription_cursor

    def __calculate_latency(self):
        """Calculate latency between audio and transcription cursors"""
        if self.transcription_cursor is not None:
            logger.info('Audio cursor is at %s & transcription cursor is at %s', self.audio_cursor, self.transcription_cursor)
            return self.audio_cursor - self.transcription_cursor
        return None

    async def run(self):
        """Entry point for the Bolna framework"""
        try:
            self.transcription_task = asyncio.create_task(self.transcribe())
        except Exception as e:
            logger.error("Error starting AssemblyAI transcription task: %s", e)

    async def transcribe(self):
        """Main transcription orchestration method"""
        assembly_ws = None
        try:
            start_time = time.perf_counter()
            
            if self.stream:
                # Only establish WebSocket connection for streaming mode
                try:
                    assembly_ws = await self.assembly_connect()
                except (ValueError, ConnectionError) as e:
                    logger.error("Failed to establish AssemblyAI connection: %s", e)
                    await self.toggle_connection()
                    return
                
                if not self.connection_time:
                    self.connection_time = round((time.perf_counter() - start_time) * 1000)
                
                # Start sender task for streaming audio data
                self.sender_task = asyncio.create_task(self.sender_stream(assembly_ws))
                # Start heartbeat task to keep connection alive
                self.heartbeat_task = asyncio.create_task(self.send_heartbeat(assembly_ws))
                
                try:
                    # Process incoming messages from AssemblyAI
                    async for message in self.receiver(assembly_ws):
                        if self.connection_on:
                            await self.push_to_transcriber_queue(message)
                        else:
                            logger.info("Connection is off, closing the AssemblyAI connection")
                            
                            # Only send termination if we haven't already cleaned up
                            if not self.cleanup_completed:
                                terminate_message = {"type": "Terminate"}
                                try:
                                    await assembly_ws.send(json.dumps(terminate_message))
                                except Exception as e:
                                    pass
                            else:
                                pass
                            break
                except ConnectionClosedError as e:
                    logger.error("AssemblyAI websocket connection closed during streaming: %s", e)
                except Exception as e:
                    logger.error("Error during AssemblyAI streaming: %s", e)
                    raise
            else:
                # Use HTTP API for non-streaming mode
                logger.info("Starting AssemblyAI HTTP API transcription mode")
                async for message in self.sender():
                    await self.push_to_transcriber_queue(message)

        except (ValueError, ConnectionError) as e:
            logger.error("Connection error in AssemblyAI transcribe: %s", e)
            await self.toggle_connection()
        except Exception as e:
            logger.error("Unexpected error in AssemblyAI transcribe: %s", e)
            await self.toggle_connection()
        finally:
            if assembly_ws is not None:
                try:
                    await assembly_ws.close()
                    logger.info("AssemblyAI websocket closed in finally block")
                except Exception as e:
                    logger.error("Error closing AssemblyAI websocket in finally block: %s", e)
                finally:
                    self.websocket_connection = None
                    self.connection_authenticated = False
                    self.session_id = None
            
            if hasattr(self, 'sender_task') and self.sender_task is not None:
                self.sender_task.cancel()
            if hasattr(self, 'heartbeat_task') and self.heartbeat_task is not None:
                self.heartbeat_task.cancel()
            
            await self.push_to_transcriber_queue(
                create_ws_data_packet("transcriber_connection_closed", getattr(self, 'meta_info', {}))
            )
