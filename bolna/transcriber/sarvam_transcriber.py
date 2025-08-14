import asyncio
import json
import time
import aiohttp
import websockets
from websockets.asyncio.client import ClientConnection
import os
import base64
from dotenv import load_dotenv
from .base_transcriber import BaseTranscriber
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.utils import create_ws_data_packet

load_dotenv()
logger = configure_logger(__name__)


class SarvamTranscriber(BaseTranscriber):
    def __init__(self, telephony_provider, input_queue=None, model='saarika:v2.5', stream=True, 
                 language="en-IN", target_language=None, encoding="linear16", sampling_rate="16000",
                 output_queue=None, high_vad_sensitivity=False, vad_signals=False, disable_sdk=False, **kwargs):
        super().__init__(input_queue)

        self.telephony_provider = telephony_provider
        self.provider = telephony_provider
        self.model = model
        self.language = language
        self.target_language = target_language
        self.stream = stream
        self.encoding = encoding
        self.sampling_rate = int(sampling_rate)
        self.high_vad_sensitivity = high_vad_sensitivity
        self.vad_signals = vad_signals
        self.disable_sdk = disable_sdk

        self.api_key = kwargs.get("transcriber_key", os.getenv('SARVAM_API_KEY'))
        self.api_url = "https://api.sarvam.ai/speech-to-text"
        self.ws_url = "wss://api.sarvam.ai/speech-to-text/ws"

        logger.info(f"SarvamTranscriber initialized: model={self.model}, language={self.language}, stream={self.stream}")

        self.transcriber_output_queue = output_queue
        self.transcription_task = None
        self.sender_task = None
        self.heartbeat_task = None

        self.audio_submitted = False
        self.audio_submission_time = None
        self.num_frames = 0
        self.connection_start_time = None
        self.audio_frame_duration = 0.0
        self.audio_cursor = 0.0

        self.current_transcript = ""
        self.is_speech_active = False
        self.final_transcript = ""
        self.is_transcript_sent_for_processing = False

        # Track failed attempts for VAD settings adjustment
        self.failed_attempts = 0
        self.max_failed_attempts = 3
        self.vad_settings_changed = False  # Track if VAD settings have been modified

        self._configure_audio_params()

        if not self.stream:
            self.session = aiohttp.ClientSession()

    def _configure_audio_params(self):
        """Configure audio parameters based on telephony provider"""
        if self.telephony_provider == "plivo":
            self.encoding = "linear16"
            # Plivo sends 8kHz PCM; convert to 16kHz WAV for Sarvam
            self.input_sampling_rate = 8000
            self.sampling_rate = 16000
            self.audio_frame_duration = 0.2
        elif self.telephony_provider == "twilio":
            self.encoding = "mulaw"
            # Twilio media frames are 8000 Hz mu-law; upsample to 16000 Hz WAV for Sarvam
            self.input_sampling_rate = 8000
            self.sampling_rate = 16000
            self.audio_frame_duration = 0.2
        else:
            # Default configuration
            self.encoding = self.encoding or "linear16"
            self.sampling_rate = int(self.sampling_rate) if isinstance(self.sampling_rate, str) else self.sampling_rate
            self.input_sampling_rate = self.sampling_rate
            self.audio_frame_duration = 0.2
        
        logger.info(f"Audio config: {self.encoding}@{self.sampling_rate}Hz, frame_duration={self.audio_frame_duration}s")

    def _get_ws_url(self):
        params = {
            'model': self.model,
            'language-code': self.language,
        }
        if self.high_vad_sensitivity:
            params['high_vad_sensitivity'] = 'true'
        if self.vad_signals:
            params['vad_signals'] = 'true'
        if self.target_language:
            params['target_language'] = self.target_language
        query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
        return f"{self.ws_url}?{query_string}"

    async def _get_http_transcription(self, audio_data):
        logger.info("Starting HTTP transcription request")
        logger.info(f"Audio data type: {type(audio_data)}")
        logger.info(f"Audio data length: {len(audio_data) if audio_data else 0}")
        
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
            logger.info("Created new aiohttp session")

        # Convert audio data to WAV format for HTTP API
        wav_data = self._convert_audio_to_wav(audio_data)
        if wav_data is None:
            logger.error("Failed to convert audio to WAV format")
            return create_ws_data_packet("", self.meta_info)

        try:
            import io
            
            # Create form data with file upload (working approach from bad_sarvam_transcriber.py)
            data = aiohttp.FormData()
            data.add_field('file', 
                          io.BytesIO(wav_data), 
                          filename='audio.wav',
                          content_type='audio/wav')
            data.add_field('model', self.model)
            data.add_field('language_code', self.language)  # Note: language_code not language

            headers = {
                'api-subscription-key': self.api_key
            }

            self.current_request_id = self.generate_request_id()
            self.meta_info['request_id'] = self.current_request_id
            start_time = time.time()

            logger.info(f"HTTP request URL: {self.api_url}")
            logger.info(f"HTTP request headers: {json.dumps({k: v if k != 'api-subscription-key' else f'<{len(v)} chars>' for k, v in headers.items()})}")
            logger.info(f"HTTP request form data fields: model={self.model}, language_code={self.language}")

            async with self.session.post(self.api_url, data=data, headers=headers) as response:
                logger.info(f"HTTP response status: {response.status}")
                response_text = await response.text()
                logger.info(f"HTTP response text: {response_text}")
                
                if response.status == 200:
                    try:
                        response_data = json.loads(response_text)
                        logger.info(f"HTTP response data: {response_data}")
                        
                        self.meta_info["start_time"] = start_time
                        self.meta_info['transcriber_latency'] = time.time() - start_time
                        self.meta_info['transcriber_duration'] = response_data.get("duration", 0)

                        transcript = response_data.get("transcript", "")
                        logger.info(f"Extracted transcript: {transcript}")

                        return create_ws_data_packet(transcript, self.meta_info)
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse JSON response: {e}")
                        logger.error(f"Raw response: {response_text}")
                        return create_ws_data_packet("", self.meta_info)
                elif response.status == 429:
                    # Rate limit exceeded - this should be handled by the caller with backoff
                    logger.warning("Rate limit exceeded (429), caller should handle backoff")
                    raise Exception("Rate limit exceeded")
                else:
                    logger.error(f"HTTP {response.status} error: {response_text}")
                    return create_ws_data_packet("", self.meta_info)
                    
        except Exception as e:
            logger.error(f"Error in HTTP transcription: {e}")
            logger.error(f"Exception details: {type(e).__name__}: {str(e)}")
            raise  # Re-raise to let caller handle it

    def _convert_audio_to_wav(self, audio_data):
        """Convert raw audio data to WAV format for HTTP API."""
        try:
            logger.debug(f"🎧 WAV CONVERT: input type={type(audio_data)}, length={len(audio_data) if audio_data else 0}")
            
            if isinstance(audio_data, str):
                logger.debug("🎧 WAV CONVERT: decoding from base64 string")
                audio_bytes = base64.b64decode(audio_data)
            else:
                logger.debug("🎧 WAV CONVERT: input is already bytes")
                audio_bytes = audio_data

            logger.debug(f"🎧 WAV CONVERT: audio_bytes length: {len(audio_bytes)}")

            if self.encoding == 'mulaw':
                logger.debug("🎧 WAV CONVERT: converting mulaw to linear16")
                import audioop
                original_len = len(audio_bytes)
                audio_bytes = audioop.ulaw2lin(audio_bytes, 2)
                logger.debug(f"🎧 WAV CONVERT: mulaw conversion: {original_len} -> {len(audio_bytes)} bytes")
            
            # Resample if needed to match target WAV sample rate
            try:
                current_rate = getattr(self, 'input_sampling_rate', self.sampling_rate)
                if current_rate != self.sampling_rate:
                    import audioop
                    logger.debug(f"🎧 WAV CONVERT: resampling {current_rate} -> {self.sampling_rate}")
                    audio_bytes, _ = audioop.ratecv(audio_bytes, 2, 1, current_rate, self.sampling_rate, None)
                    logger.debug(f"🎧 WAV CONVERT: resampled length: {len(audio_bytes)} bytes")
            except Exception as rs_err:
                logger.warning(f"🎧 WAV CONVERT: resample failed, continuing with original rate bytes: {rs_err}")

            # Convert to WAV format
            import io
            import wave
            import numpy as np
            
            logger.debug(f"🎧 WAV CONVERT: creating numpy array from {len(audio_bytes)} bytes")
            # Convert to numpy array
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            logger.debug(f"🎧 WAV CONVERT: numpy array shape: {audio_array.shape}, dtype: {audio_array.dtype}")
            
            # Create WAV file in memory
            logger.debug(f"🎧 WAV CONVERT: creating WAV file - channels=1, sampwidth=2, framerate={self.sampling_rate}")
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.sampling_rate)
                wav_file.writeframes(audio_array.tobytes())
            
            wav_buffer.seek(0)
            wav_data = wav_buffer.getvalue()
            logger.debug(f"🎧 WAV CONVERT: SUCCESS - output WAV length: {len(wav_data)} bytes")
            return wav_data
            
        except Exception as e:
            logger.error(f"🎧 WAV CONVERT ERROR: {e}")
            logger.error(f"🎧 WAV CONVERT ERROR details: input_length={len(audio_data) if audio_data else 0}, encoding={self.encoding}")
            return None

    async def _check_and_process_end_of_stream(self, ws_data_packet, ws=None):
        if 'eos' in ws_data_packet['meta_info'] and ws_data_packet['meta_info']['eos'] is True:
            logger.info("End of stream detected")
            if ws is not None:
                try:
                    await ws.close()
                except Exception:
                    pass
            return True
        return False

    async def sender_http(self):
        logger.info("Starting sender_http task")
        try:
            # Batching and rate limiting for HTTP mode
            buffer_flush_interval_sec = 2.5
            last_flush_time = 0.0
            audio_buffer: list[bytes] = []
            consecutive_errors = 0
            max_consecutive_errors = 3
            last_sent_transcript: str | None = None
            
            while True:
                logger.debug("Waiting for audio packet from input queue (HTTP mode)")
                ws_data_packet = await self.input_queue.get()
                logger.debug(f"Received audio packet: keys={list(ws_data_packet.keys()) if ws_data_packet else 'None'}")

                # Check EOS first
                if 'eos' in ws_data_packet.get('meta_info', {}) and ws_data_packet['meta_info']['eos'] is True:
                    logger.info("End of stream detected in HTTP mode, flushing buffer and stopping sender")
                    # Flush remaining buffer once
                    if audio_buffer:
                        try:
                            combined_bytes = b"".join(audio_buffer)
                            packet = await self._get_http_transcription(combined_bytes)
                            # Normalize and deduplicate transcript payload for TaskManager streaming path
                            if isinstance(packet, dict):
                                data = packet.get("data")
                                text = None
                                if isinstance(data, str):
                                    text = data.strip()
                                elif isinstance(data, dict) and data.get("type") == "transcript":
                                    text = (data.get("content") or "").strip()
                                if text:
                                    if text != last_sent_transcript:
                                        packet["data"] = {"type": "transcript", "content": text}
                                        await self.push_to_transcriber_queue(packet)
                                        last_sent_transcript = text
                                    else:
                                        logger.info("Skipping duplicate transcript on EOS in HTTP mode")
                                else:
                                    logger.info("Skipping empty transcript on EOS in HTTP mode")
                            else:
                                await self.push_to_transcriber_queue(packet)
                        except Exception as e:
                            logger.error(f"Error flushing buffer on EOS: {e}")
                        finally:
                            audio_buffer = []
                    break

                # Initialize on first packet
                if not self.audio_submitted:
                    logger.debug("First audio packet received, initializing request")
                    self.meta_info = ws_data_packet.get('meta_info')
                    self.audio_submitted = True
                    self.audio_submission_time = time.time()
                    self.current_request_id = self.generate_request_id()
                    self.meta_info['request_id'] = self.current_request_id
                    logger.debug(f"Initialized with request_id: {self.current_request_id}")

                self.num_frames += 1
                logger.debug(f"Processing frame {self.num_frames}")

                # Append audio to buffer
                frame = ws_data_packet.get("data")
                if frame:
                    audio_buffer.append(frame)
                else:
                    logger.warning("Received audio packet with no data")

                # Decide if we should flush buffer
                now = time.time()
                should_flush = (now - last_flush_time) >= buffer_flush_interval_sec
                if should_flush and audio_buffer:
                    try:
                        combined_bytes = b"".join(audio_buffer)
                        logger.debug(f"Flushing HTTP buffer, bytes={len(combined_bytes)}")
                        packet = await self._get_http_transcription(combined_bytes)
                        # Normalize and deduplicate transcript payload for TaskManager streaming path
                        if isinstance(packet, dict):
                            data = packet.get("data")
                            text = None
                            if isinstance(data, str):
                                text = data.strip()
                            elif isinstance(data, dict) and data.get("type") == "transcript":
                                text = (data.get("content") or "").strip()
                            if text:
                                if text != last_sent_transcript:
                                    packet["data"] = {"type": "transcript", "content": text}
                                    await self.push_to_transcriber_queue(packet)
                                    last_sent_transcript = text
                                else:
                                    logger.info("Skipping empty/duplicate transcript in HTTP mode")
                            else:
                                logger.info("Skipping empty/duplicate transcript in HTTP mode")
                        else:
                            await self.push_to_transcriber_queue(packet)

                        last_flush_time = now
                        audio_buffer = []
                        consecutive_errors = 0
                    except Exception as e:
                        consecutive_errors += 1
                        logger.error(f"HTTP transcription error {consecutive_errors}/{max_consecutive_errors}: {e}")
                        if consecutive_errors >= max_consecutive_errors:
                            logger.error("Too many consecutive HTTP errors, stopping sender")
                            break
                        backoff = min(2 ** consecutive_errors, 10)
                        logger.info(f"Backing off for {backoff}s")
                        await asyncio.sleep(backoff)
        except asyncio.CancelledError:
            logger.info("HTTP sender was cancelled")
        except Exception as e:
            logger.error(f"Error in HTTP sender: {e}")
            logger.error(f"Exception details: {type(e).__name__}: {str(e)}")

    async def sender_stream(self, ws: ClientConnection):
        try:
            while True:
                ws_data_packet = await self.input_queue.get()

                if not self.audio_submitted:
                    self.meta_info = ws_data_packet.get('meta_info')
                    self.audio_submitted = True
                    self.audio_submission_time = time.time()
                    self.current_request_id = self.generate_request_id()
                    self.meta_info['request_id'] = self.current_request_id

                end_of_stream = await self._check_and_process_end_of_stream(ws_data_packet, ws)
                if end_of_stream:
                    break

                self.num_frames += 1
                self.audio_cursor = self.num_frames * self.audio_frame_duration

                audio_data = ws_data_packet.get('data')
                if audio_data:
                    # Convert to WAV for UI-compatible schema
                    wav_bytes = self._convert_audio_to_wav(audio_data)
                    if not wav_bytes:
                        logger.warning("sender_stream: Skipping frame due to WAV conversion failure")
                        continue
                    audio_b64 = base64.b64encode(wav_bytes).decode('utf-8')

                    message = {
                        'audio': {
                            'data': audio_b64,
                            'encoding': 'audio/wav',
                            'sample_rate': self.sampling_rate
                        }
                    }
                    logger.debug(f"sender_stream: sending bytes wav={len(wav_bytes)} b64={len(audio_b64)}")
                    await ws.send(json.dumps(message))

        except Exception as e:
            logger.error(f'Error while sending audio: {e}')

    async def receiver(self, ws: ClientConnection):
        try:
            async for message in ws:
                try:
                    if isinstance(message, str):
                        data = json.loads(message)
                    else:
                        data = message

                    if self.connection_start_time is None:
                        self.connection_start_time = time.time() - (self.num_frames * self.audio_frame_duration)

                    logger.info(f"🔄 RECEIVER: received message: {data}")
                    # UI-compatible schema: type can be 'data' or 'events'
                    if isinstance(data, dict) and data.get('type') == 'data':
                        payload = data.get('data', {})
                        transcript = payload.get('transcript', '')
                        language_code = payload.get('language_code')
                        metrics = payload.get('metrics', {})
                        if transcript and transcript.strip():
                            transcript_data = {
                                "type": "transcript",
                                "content": transcript.strip()
                            }
                            if self.meta_info is not None:
                                self.meta_info["transcriber_duration"] = metrics.get('audio_duration', 0)
                            yield create_ws_data_packet(transcript_data, self.meta_info)
                    elif isinstance(data, dict) and data.get('type') == 'events':
                        vad = data.get('data', {})
                        signal_type = vad.get('signal_type')
                        if signal_type == 'START_SPEECH':
                            yield create_ws_data_packet("speech_started", self.meta_info)
                        elif signal_type == 'END_SPEECH':
                            pass
                    elif isinstance(data, dict) and data.get('type') == 'connection_closed':
                        self.meta_info["transcriber_duration"] = data.get("duration", 0)
                        yield create_ws_data_packet("transcriber_connection_closed", self.meta_info)
                        return

                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing WebSocket message: {e}")
                except Exception as e:
                    logger.error(f"Error processing WebSocket message: {e}")

        except Exception as e:
            logger.error(f"Error in WebSocket receiver: {e}")

    async def sarvam_connect(self):
        ws_url = self._get_ws_url()
        logger.info(f"Connecting to Sarvam WS: {ws_url}")
        try:
            ws = await websockets.connect(ws_url, subprotocols=[f"api-subscription-key.{self.api_key}"])
            logger.info("Sarvam WS connected")
            return ws
        except Exception as e:
            logger.error(f"Failed to connect to Sarvam WebSocket: {e}")
            raise

    async def push_to_transcriber_queue(self, data_packet):
        logger.info("📤 QUEUE OUT: pushing to transcriber_output_queue")
        await self.transcriber_output_queue.put(data_packet)

    async def toggle_connection(self):
        logger.info("🔌 TOGGLE: turning connection off")
        self.connection_on = False

        if self.sender_task:
            logger.info("🔌 TOGGLE: cancelling sender task")
            self.sender_task.cancel()
        if self.heartbeat_task:
            logger.info("🔌 TOGGLE: cancelling heartbeat task")
            self.heartbeat_task.cancel()

    async def run(self):
        logger.info("🚀 RUN: starting Sarvam transcriber run method")
        try:
            logger.info("🚀 RUN: creating transcription task")
            self.transcription_task = asyncio.create_task(self.transcribe())
        except Exception as e:
            logger.error(f"🚀 RUN: ERROR starting Sarvam transcriber: {e}")

    async def send_heartbeat(self, ws: ClientConnection, interval_sec: float = 10.0):
        try:
            while True:
                await asyncio.sleep(interval_sec)
                try:
                    await ws.ping()
                    logger.debug("WS ping sent")
                except Exception as ping_err:
                    logger.warning(f"WS ping failed: {ping_err}")
                    break
        except asyncio.CancelledError:
            logger.debug("Heartbeat task cancelled")
        except Exception as e:
            logger.error(f"Heartbeat error: {e}")

    async def transcribe(self):
        logger.info("Starting transcribe method")
        try:
            if self.stream:
                logger.info("Using streaming mode (raw WebSocket)")
                start_time = time.perf_counter()
                async with await self.sarvam_connect() as sarvam_ws:
                    if not self.connection_time:
                        self.connection_time = round((time.perf_counter() - start_time) * 1000)
                    # Run sender and receiver concurrently
                    self.sender_task = asyncio.create_task(self.sender_stream(sarvam_ws))
                    self.heartbeat_task = asyncio.create_task(self.send_heartbeat(sarvam_ws))
                    try:
                        async for message in self.receiver(sarvam_ws):
                            if self.connection_on:
                                await self.push_to_transcriber_queue(message)
                            else:
                                logger.info("closing the Sarvam connection")
                                try:
                                    await sarvam_ws.close()
                                except Exception:
                                    pass
                        logger.info("Sarvam WS receiver completed")
                    except asyncio.CancelledError:
                        logger.info("Sarvam WS tasks cancelled")
                    except Exception as e:
                        logger.error(f"Sarvam WS error: {e}")
            else:
                logger.info("Using HTTP mode")
                self.sender_task = asyncio.create_task(self.sender_http())
                try:
                    await self.sender_task
                    logger.info("HTTP mode sender task completed")
                except asyncio.CancelledError:
                    logger.info("HTTP mode sender task was cancelled")
                except Exception as e:
                    logger.error(f"HTTP mode sender task failed: {e}")

            logger.info("🏁 TRANSCRIPTION COMPLETED: sending connection closed signal")
            try:
                await self.push_to_transcriber_queue(
                    create_ws_data_packet("transcriber_connection_closed", self.meta_info)
                )
                logger.info("🏁 TRANSCRIPTION COMPLETED: connection closed signal sent successfully")
            except Exception as eos_error:
                logger.error(f"🏁 TRANSCRIPTION COMPLETED: ERROR sending connection closed signal: {eos_error}")

        except Exception as e:
            logger.error(f"Error in Sarvam transcribe: {e}")
            logger.error(f"Exception details: {type(e).__name__}: {str(e)}")
        finally:
            if hasattr(self, 'session') and self.session and not self.session.closed:
                await self.session.close()
                logger.info("Closed aiohttp session")

    def get_meta_info(self):
        return self.meta_info