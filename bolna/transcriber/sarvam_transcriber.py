import asyncio
import base64
import json
import os
import io
import wave
import time
import traceback
import audioop
from dotenv import load_dotenv
import aiohttp
import websockets
from websockets.asyncio.client import ClientConnection
from websockets.exceptions import InvalidHandshake

import numpy as np
from scipy.signal import resample_poly
from typing import Optional

from .base_transcriber import BaseTranscriber
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.utils import create_ws_data_packet

load_dotenv()
logger = configure_logger(__name__)


class SarvamTranscriber(BaseTranscriber):
    def __init__(
        self,
        telephony_provider,
        input_queue=None,
        model="saarika:v2.5",
        stream=True,
        language="en-IN",
        target_language=None,
        encoding="linear16",
        sampling_rate="16000",
        output_queue=None,
        high_vad_sensitivity=False,
        vad_signals=False,
        disable_sdk=False,
        **kwargs,
    ):
        super().__init__(input_queue)

        self.telephony_provider = telephony_provider
        self.model = model
        self.language = language
        self.target_language = target_language
        self.stream = stream
        self.encoding = encoding
        self.sampling_rate = int(sampling_rate)
        self.high_vad_sensitivity = high_vad_sensitivity
        self.vad_signals = vad_signals
        self.disable_sdk = disable_sdk

        self.api_key = kwargs.get("transcriber_key", os.getenv("SARVAM_API_KEY"))
        self.api_host = os.getenv("SARVAM_HOST", "api.sarvam.ai")

        # saaras models use translate endpoint, saarika models use transcription endpoint
        if model.startswith("saaras"):
            self.api_url = f"https://{self.api_host}/speech-to-text-translate"
            self.ws_url = f"wss://{self.api_host}/speech-to-text-translate/ws"
        else:
            self.api_url = f"https://{self.api_host}/speech-to-text"
            self.ws_url = f"wss://{self.api_host}/speech-to-text/ws"

        self.transcriber_output_queue = output_queue
        self.transcription_task = None
        self.sender_task = None
        self.heartbeat_task = None

        self.audio_submitted = False
        self.audio_submission_time = None
        self.num_frames = 0
        self.connection_start_time = None
        self.connection_time = None
        self.audio_frame_duration = 0.0
        self.audio_cursor = 0.0

        self.final_transcript = ""
        self.websocket_connection = None
        self.connection_authenticated = False
        self.meta_info = {}

        self.current_turn_start_time = None
        self.current_turn_id = None
        self.turn_latencies = []
        self.first_result_latency_ms = None
        self.total_stream_duration_ms = None
        self.last_vocal_frame_timestamp = None
        self.turn_counter = 0
        self.turn_first_result_latency = None
        
        self.is_transcript_sent_for_processing = False
        self.curr_message = ''
        self.finalized_transcript = ""
        self.interruption_signalled = False

        self._configure_audio_params()
        self.session: Optional[aiohttp.ClientSession] = None
        if not self.stream:
            self.session = aiohttp.ClientSession()

    def _configure_audio_params(self):
        if self.telephony_provider == "plivo":
            self.encoding = "linear16"
            self.input_sampling_rate = 8000
            self.sampling_rate = 16000
            self.audio_frame_duration = 0.2
        elif self.telephony_provider == "twilio":
            self.encoding = "mulaw"
            self.input_sampling_rate = 8000
            self.sampling_rate = 16000
            self.audio_frame_duration = 0.2
        else:
            self.encoding = self.encoding or "linear16"
            self.sampling_rate = int(self.sampling_rate)
            self.input_sampling_rate = self.sampling_rate
            self.audio_frame_duration = 0.2

    def _get_ws_url(self):
        params = {"model": self.model}

        # saaras auto-detects language, saarika requires language-code
        if not self.model.startswith("saaras"):
            params["language-code"] = self.language

        if self.high_vad_sensitivity:
            params["high_vad_sensitivity"] = "true"
        if self.vad_signals:
            params["vad_signals"] = "true"
        if self.target_language:
            params["target_language"] = self.target_language
        query_string = "&".join([f"{k}={v}" for k, v in params.items()])
        return f"{self.ws_url}?{query_string}"

    async def _get_http_transcription(self, audio_data):
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()

        wav_data = self._convert_audio_to_wav(audio_data)
        if wav_data is None:
            return create_ws_data_packet("", self.meta_info)

        try:
            data = aiohttp.FormData()
            data.add_field("file", io.BytesIO(wav_data), filename="audio.wav", content_type="audio/wav")
            data.add_field("model", self.model)
            data.add_field("language_code", self.language)

            headers = {"api-subscription-key": self.api_key}

            self.current_request_id = self.generate_request_id()
            self.meta_info["request_id"] = self.current_request_id
            start_time = time.time()

            async with self.session.post(self.api_url, data=data, headers=headers) as response:
                response_text = await response.text()
                if response.status == 200:
                    try:
                        response_data = json.loads(response_text)
                        elapsed = time.time() - start_time
                        self.meta_info["start_time"] = start_time
                        self.meta_info["transcriber_first_result_latency"] = elapsed
                        self.meta_info["transcriber_latency"] = elapsed
                        self.meta_info["first_result_latency_ms"] = round(elapsed * 1000)
                        self.meta_info["transcriber_duration"] = response_data.get("duration", 0)
                        transcript = response_data.get("transcript", "")
                        return create_ws_data_packet(transcript, self.meta_info)
                    except json.JSONDecodeError:
                        return create_ws_data_packet("", self.meta_info)
                elif response.status == 429:
                    raise Exception("Rate limit exceeded")
                else:
                    return create_ws_data_packet("", self.meta_info)
        except Exception as e:
            logger.error(f"HTTP transcription error: {e}")
            raise

    def _convert_audio_to_wav(self, audio_data) -> Optional[bytes]:
        try:
            if isinstance(audio_data, str):
                audio_bytes = base64.b64decode(audio_data)
            else:
                audio_bytes = audio_data

            if self.encoding == "mulaw":
                audio_bytes = audioop.ulaw2lin(audio_bytes, 2)

            try:
                current_rate = getattr(self, "input_sampling_rate", self.sampling_rate)
                if current_rate != self.sampling_rate:
                    audio_bytes, _ = audioop.ratecv(audio_bytes, 2, 1, current_rate, self.sampling_rate, None)
            except Exception:
                audio_bytes = self.normalize_to_16k(audio_bytes, current_rate)

            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self.sampling_rate)
                wav_file.writeframes(audio_array.tobytes())
            wav_buffer.seek(0)
            return wav_buffer.getvalue()
        except Exception as e:
            logger.error(f"WAV conversion error: {e}")
            return None

    def normalize_to_16k(self, raw_audio: bytes, in_sr: int) -> bytes:
        if in_sr == self.sampling_rate:
            return raw_audio
        try:
            audio_np = np.frombuffer(raw_audio, dtype=np.int16)
            gcd = np.gcd(in_sr, self.sampling_rate)
            up = self.sampling_rate // gcd
            down = in_sr // gcd
            resampled_np = resample_poly(audio_np, up, down)
            resampled_np = np.clip(resampled_np, -32768, 32767).astype(np.int16)
            return resampled_np.tobytes()
        except Exception:
            return raw_audio

    async def _check_and_process_end_of_stream(self, ws_data_packet, ws=None):
        if ws_data_packet and ws_data_packet.get("meta_info", {}).get("eos") is True:
            if ws is not None:
                try:
                    await ws.close()
                except Exception:
                    pass
            return True
        return False

    async def sender(self):
        # HTTP batching sender
        buffer_flush_interval_sec = 2.5
        last_flush_time = time.time()
        audio_buffer = []
        consecutive_errors = 0

        try:
            while True:
                ws_data_packet = await self.input_queue.get()
                if ws_data_packet is None:
                    continue

                if ws_data_packet.get("meta_info", {}).get("eos") is True:
                    if audio_buffer:
                        combined_bytes = b"".join(audio_buffer)
                        packet = await self._get_http_transcription(combined_bytes)
                        if isinstance(packet, dict):
                            await self.push_to_transcriber_queue(packet)
                        audio_buffer = []
                    break

                if not self.audio_submitted:
                    self.meta_info = ws_data_packet.get("meta_info", {})
                    self.audio_submitted = True
                    self.audio_submission_time = time.time()
                    self.current_request_id = self.generate_request_id()
                    self.meta_info["request_id"] = self.current_request_id

                self.num_frames += 1
                frame = ws_data_packet.get("data")
                if frame:
                    audio_buffer.append(frame)

                now = time.time()
                if (now - last_flush_time) >= buffer_flush_interval_sec and audio_buffer:
                    try:
                        combined_bytes = b"".join(audio_buffer)
                        packet = await self._get_http_transcription(combined_bytes)
                        if isinstance(packet, dict):
                            await self.push_to_transcriber_queue(packet)
                        last_flush_time = now
                        audio_buffer = []
                        consecutive_errors = 0
                    except Exception:
                        consecutive_errors += 1
                        if consecutive_errors >= 3:
                            break
                        await asyncio.sleep(min(2 ** consecutive_errors, 10))
        except asyncio.CancelledError:
            pass

    async def sender_stream(self, ws: ClientConnection):
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

                if await self._check_and_process_end_of_stream(ws_data_packet, ws):
                    break

                self.num_frames += 1
                self.audio_cursor = self.num_frames * self.audio_frame_duration

                audio_data = ws_data_packet.get("data")
                if audio_data:
                    wav_bytes = self._convert_audio_to_wav(audio_data)
                    if not wav_bytes:
                        continue
                    audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")
                    message = {"audio": {"data": audio_b64, "encoding": "audio/wav", "sample_rate": self.sampling_rate}}
                    await ws.send(json.dumps(message))
        except asyncio.CancelledError:
            pass

    async def receiver(self, ws: ClientConnection):
        try:
            async for message in ws:
                try:
                    data = json.loads(message) if isinstance(message, str) else message

                    if self.connection_start_time is None:
                        self.connection_start_time = time.time() - (self.num_frames * self.audio_frame_duration)

                    if isinstance(data, dict) and data.get("type") == "data":
                        payload = data.get("data", {})
                        transcript = payload.get("transcript", "")
                        metrics = payload.get("metrics", {})

                        if transcript and transcript.strip():
                            now_timestamp = time.time()
                            
                            if self.first_result_latency_ms is None and self.audio_submission_time:
                                first_result_latency_seconds = now_timestamp - self.audio_submission_time
                                self.first_result_latency_ms = round(first_result_latency_seconds * 1000)
                                self.meta_info["transcriber_first_result_latency"] = first_result_latency_seconds
                                self.meta_info["transcriber_latency"] = first_result_latency_seconds  
                                self.meta_info["first_result_latency_ms"] = self.first_result_latency_ms

                            if self.current_turn_start_time and self.turn_first_result_latency is None:
                                turn_latency_seconds = time.perf_counter() - self.current_turn_start_time
                                self.turn_first_result_latency = round(turn_latency_seconds * 1000)
                                self.meta_info["transcriber_first_result_latency"] = turn_latency_seconds
                                self.meta_info["transcriber_latency"] = turn_latency_seconds  

                            transcript_data = {"type": "transcript", "content": transcript.strip()}
                            self.meta_info["transcriber_duration"] = metrics.get("audio_duration", 0)

                            self.last_vocal_frame_timestamp = now_timestamp
                            self.meta_info["last_vocal_frame_timestamp"] = self.last_vocal_frame_timestamp

                            yield create_ws_data_packet(
                                {"type": "interim_transcript_received", "content": transcript.strip()},
                                self.meta_info,
                            )
                            yield create_ws_data_packet(transcript_data, self.meta_info)

                    elif isinstance(data, dict) and data.get("type") == "events":
                        vad = data.get("data", {})
                        if vad.get("signal_type") == "START_SPEECH":
                            self.current_turn_start_time = time.perf_counter()
                            self.turn_counter += 1
                            self.current_turn_id = f"turn_{self.turn_counter}"
                            self.turn_first_result_latency = None
                            yield create_ws_data_packet("speech_started", self.meta_info)

                        elif vad.get("signal_type") == "END_SPEECH":
                            now = time.time()
                            self.last_vocal_frame_timestamp = now
                            self.meta_info["last_vocal_frame_timestamp"] = self.last_vocal_frame_timestamp

                            if self.current_turn_start_time:
                                total_stream_duration = time.perf_counter() - self.current_turn_start_time
                                total_stream_duration_ms = round(total_stream_duration * 1000)
                                
                                self.meta_info['transcriber_total_stream_duration'] = total_stream_duration
                                self.meta_info['transcriber_latency'] = total_stream_duration  

                                turn_info = {
                                    "turn_id": self.current_turn_id,
                                    "sequence_id": self.current_turn_id,
                                    "first_result_latency_ms": self.turn_first_result_latency,
                                    "total_stream_duration_ms": total_stream_duration_ms,  
                                }
                                self.turn_latencies.append(turn_info)
                                self.meta_info["turn_latencies"] = self.turn_latencies
                                
                                # Reset turn tracking 
                                self.current_turn_start_time = None
                                self.current_turn_id = None

                            if self.final_transcript:
                                yield create_ws_data_packet(
                                    {"type": "transcript", "content": self.final_transcript}, self.meta_info
                                )
                                self.final_transcript = ""

                            yield create_ws_data_packet("speech_ended", self.meta_info)

                    elif isinstance(data, dict) and data.get("type") == "connection_closed":
                        self.meta_info["transcriber_duration"] = data.get("duration", 0)
                        yield create_ws_data_packet("transcriber_connection_closed", self.meta_info)
                        return
                except Exception:
                    traceback.print_exc()
        except Exception:
            traceback.print_exc()

    async def _close(self, ws: ClientConnection, data=None):
        try:
            if data:
                await ws.send(json.dumps(data))
            await ws.close()
        except Exception:
            pass

    async def sarvam_connect(self, retries: int = 3, timeout: float = 10.0) -> ClientConnection:
        ws_url = self._get_ws_url()
        additional_headers = {
            'api-subscription-key': self.api_key,
        }
        attempt = 0
        last_err = None
        while attempt < retries:
            try:
                logger.info(f"Attempting to connect to Sarvam websocket: {ws_url}")
                ws = await asyncio.wait_for(
                    websockets.connect(ws_url, additional_headers=additional_headers),
                    timeout=timeout,
                )
                self.websocket_connection = ws
                self.connection_authenticated = True
                logger.info("Successfully connected to Sarvam websocket")
                return ws
            except asyncio.TimeoutError:
                logger.error("Timeout while connecting to Sarvam websocket")
                raise ConnectionError("Timeout while connecting to Sarvam websocket")
            except InvalidHandshake as e:
                error_msg = str(e)
                if '401' in error_msg or '403' in error_msg:
                    logger.error(f"Sarvam authentication failed: Invalid or expired API key - {e}")
                    raise ConnectionError(f"Sarvam authentication failed: Invalid or expired API key - {e}")
                elif '404' in error_msg:
                    logger.error(f"Sarvam endpoint not found - check model/configuration: {e}")
                    raise ConnectionError(f"Sarvam endpoint not found: {e}")
                else:
                    logger.error(f"Invalid handshake during Sarvam websocket connection: {e}")
                    last_err = e
                    attempt += 1
                    if attempt < retries:
                        await asyncio.sleep(2 ** attempt)
            except Exception as e:
                logger.error(f"Error connecting to Sarvam websocket (attempt {attempt + 1}/{retries}): {e}")
                last_err = e
                attempt += 1
                if attempt < retries:
                    await asyncio.sleep(2 ** attempt)
        raise ConnectionError(f"Failed to connect to Sarvam after {retries} attempts: {last_err}")

    async def push_to_transcriber_queue(self, data_packet):
        if self.transcriber_output_queue is not None:
            await self.transcriber_output_queue.put(data_packet)

    async def toggle_connection(self):
        self.connection_on = False
        if self.sender_task:
            self.sender_task.cancel()
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
        if self.websocket_connection:
            try:
                await self.websocket_connection.close()
            except Exception:
                pass
            finally:
                self.websocket_connection = None
                self.connection_authenticated = False

    async def cleanup(self):
        """Clean up all resources including HTTP session and websocket."""
        logger.info("Cleaning up Sarvam transcriber resources")

        # Close HTTP session (for non-streaming mode)
        if hasattr(self, 'session') and self.session and not self.session.closed:
            try:
                await self.session.close()
                logger.info("Sarvam HTTP session closed")
            except Exception as e:
                logger.error(f"Error closing Sarvam HTTP session: {e}")

        # Cancel tasks properly
        for task_name, task in [
            ("heartbeat_task", getattr(self, 'heartbeat_task', None)),
            ("sender_task", getattr(self, 'sender_task', None)),
            ("transcription_task", getattr(self, 'transcription_task', None))
        ]:
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    logger.info(f"Sarvam {task_name} cancelled")
                except Exception as e:
                    logger.error(f"Error cancelling Sarvam {task_name}: {e}")

        # Close websocket
        if self.websocket_connection is not None:
            try:
                await self.websocket_connection.close()
                logger.info("Sarvam websocket connection closed")
            except Exception as e:
                logger.error(f"Error closing Sarvam websocket: {e}")
            finally:
                self.websocket_connection = None
                self.connection_authenticated = False

    async def run(self):
        try:
            self.transcription_task = asyncio.create_task(self.transcribe())
        except Exception:
            traceback.print_exc()

    async def send_heartbeat(self, ws: ClientConnection, interval_sec: float = 10.0):
        try:
            while True:
                await asyncio.sleep(interval_sec)
                try:
                    await ws.ping()
                except Exception:
                    break
        except asyncio.CancelledError:
            pass

    async def transcribe(self):
        try:
            start_time = time.perf_counter()
            try:
                sarvam_ws = await self.sarvam_connect()
            except (ValueError, ConnectionError):
                await self.toggle_connection()
                return

            if not self.connection_time:
                self.connection_time = round((time.perf_counter() - start_time) * 1000)

            if self.stream:
                try:
                    async with sarvam_ws:
                        self.sender_task = asyncio.create_task(self.sender_stream(sarvam_ws))
                        self.heartbeat_task = asyncio.create_task(self.send_heartbeat(sarvam_ws))
                        async for message in self.receiver(sarvam_ws):
                            if getattr(self, "connection_on", True):
                                await self.push_to_transcriber_queue(message)
                            else:
                                await self._close(sarvam_ws, {"type": "CloseStream"})
                                break
                except Exception:
                    traceback.print_exc()
            else:
                self.sender_task = asyncio.create_task(self.sender())
                try:
                    await self.sender_task
                except asyncio.CancelledError:
                    pass

            if self.audio_submission_time:
                self.total_stream_duration_ms = round((time.time() - self.audio_submission_time) * 1000)
                self.meta_info["total_stream_duration_ms"] = self.total_stream_duration_ms

            try:
                await self.push_to_transcriber_queue(create_ws_data_packet("transcriber_connection_closed", self.meta_info))
            except Exception:
                traceback.print_exc()
        finally:
            if self.sender_task:
                self.sender_task.cancel()
            if self.heartbeat_task:
                self.heartbeat_task.cancel()
            if self.session and not self.session.closed:
                await self.session.close()
            if self.websocket_connection:
                try:
                    await self.websocket_connection.close()
                except Exception:
                    pass
                finally:
                    self.websocket_connection = None
                    self.connection_authenticated = False

    def get_meta_info(self):
        return getattr(self, "meta_info", {})
