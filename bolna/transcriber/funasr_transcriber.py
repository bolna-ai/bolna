import asyncio
import audioop
import io
import json
import os
import time
import traceback
import wave
from typing import Optional
from urllib.parse import urlparse

import aiohttp
import numpy as np
import websockets
from dotenv import load_dotenv
from scipy.signal import resample_poly
from websockets.asyncio.client import ClientConnection
from websockets.exceptions import ConnectionClosedError, InvalidHandshake

from .base_transcriber import BaseTranscriber
from bolna.constants import FUNASR_DEFAULT_CHUNK_SIZE, FUNASR_DEFAULT_WS_URL, WEB_BASED_CALL_PROVIDER
from bolna.enums import TelephonyProvider
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.ssl_context import get_ssl_context
from bolna.helpers.utils import create_ws_data_packet, timestamp_ms

load_dotenv()
logger = configure_logger(__name__)


class FunASRTranscriber(BaseTranscriber):
    """Self-hosted FunASR / SenseVoice STT.

    Streaming uses the FunASR WebSocket runtime (classic ``2pass`` protocol or the
    newer ``funasr-realtime-server`` START/COMMIT/STOP protocol). Non-stream uses the
    OpenAI-compatible ``POST /v1/audio/transcriptions`` endpoint from ``funasr-server``.

    Model weights stay on the FunASR server — Bolna only speaks the network protocol.
    """

    TARGET_SAMPLE_RATE = 16000

    def __init__(
        self,
        telephony_provider,
        input_queue=None,
        model="sensevoice",
        stream=True,
        language="en",
        encoding="linear16",
        sampling_rate="16000",
        output_queue=None,
        endpointing=500,
        keywords=None,
        **kwargs,
    ):
        super().__init__(input_queue)
        self.telephony_provider = telephony_provider
        self.model = model
        self.stream = stream
        self.language = language or "en"
        self.encoding = encoding
        self.sampling_rate = int(sampling_rate)
        self.endpointing_ms = int(endpointing) if endpointing is not None else 500
        self.keywords = keywords
        self.transcriber_output_queue = output_queue
        self.connected_via_dashboard = kwargs.get("enforce_streaming", True)

        # Protocol: "wss" (classic funasr_wss) or "realtime" (START/COMMIT/STOP).
        self.protocol = (kwargs.get("funasr_protocol") or os.getenv("FUNASR_PROTOCOL") or "wss").lower()
        self.asr_mode = kwargs.get("funasr_mode") or os.getenv("FUNASR_MODE") or "2pass"
        self.chunk_size = kwargs.get("chunk_size") or FUNASR_DEFAULT_CHUNK_SIZE
        self.chunk_interval = int(kwargs.get("chunk_interval") or os.getenv("FUNASR_CHUNK_INTERVAL") or 10)
        self.use_itn = bool(kwargs.get("use_itn", True))

        self.api_key = kwargs.get("transcriber_key", os.getenv("FUNASR_API_KEY", ""))
        self.ws_url = (
            kwargs.get("transcriber_host")
            or os.getenv("FUNASR_WS_URL")
            or FUNASR_DEFAULT_WS_URL
        )
        self.http_base_url = (
            kwargs.get("http_base_url")
            or os.getenv("FUNASR_HTTP_URL")
            or self._derive_http_base(self.ws_url)
        )

        self.audio_frame_duration = 0.2
        self._resolve_audio_params()

        self.websocket_connection: Optional[ClientConnection] = None
        self.connection_authenticated = False
        self.connection_error = None
        self.http_session: Optional[aiohttp.ClientSession] = None

        self.transcription_task = None
        self.sender_task = None
        self.commit_task = None

        self.audio_submitted = False
        self.num_frames = 0
        self.connection_start_time = None
        self.connection_time = None
        self.audio_frame_timestamps = []
        self._batch_pcm = bytearray()

        self.final_transcript = ""
        self.current_turn_id = None
        self.current_turn_start_time = None
        self.current_turn_interim_details = []
        self.turn_counter = 0
        self.last_interim_time = None
        self._turn_first_speech_epoch_ms = None
        self._config_sent = False
        self._speech_active = False

        # Client-endpoint realtime protocol: commit after silence (endpointing_ms).
        self._silence_task: Optional[asyncio.Task] = None

    @staticmethod
    def _derive_http_base(ws_url: str) -> str:
        parsed = urlparse(ws_url)
        scheme = "https" if parsed.scheme == "wss" else "http"
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or (443 if scheme == "https" else 8000)
        return f"{scheme}://{host}:{port}"

    def _resolve_audio_params(self):
        if self.telephony_provider in TelephonyProvider.telephony_values():
            if self.telephony_provider != TelephonyProvider.SIP_TRUNK.value:
                self.encoding = "mulaw" if self.telephony_provider == "twilio" else "linear16"
                self.sampling_rate = 8000
            self.audio_frame_duration = 0.2
        elif self.telephony_provider == WEB_BASED_CALL_PROVIDER:
            self.encoding = "linear16"
            self.sampling_rate = 16000
            self.audio_frame_duration = 0.256
        elif not self.connected_via_dashboard:
            self.encoding = "linear16"
            self.sampling_rate = 16000

    def get_meta_info(self):
        return self.meta_info

    def _to_pcm16k(self, data: bytes) -> bytes:
        """Normalize telephony/web audio to 16-bit PCM at 16 kHz for FunASR."""
        if not data:
            return b""
        if self.encoding in ("mulaw", "ulaw", "audio/x-mulaw"):
            pcm = audioop.ulaw2lin(data, 2)
            src_rate = self.sampling_rate or 8000
        else:
            pcm = data
            src_rate = self.sampling_rate or self.TARGET_SAMPLE_RATE

        if src_rate == self.TARGET_SAMPLE_RATE:
            return pcm

        samples = np.frombuffer(pcm, dtype=np.int16)
        if samples.size == 0:
            return b""
        resampled = resample_poly(samples, self.TARGET_SAMPLE_RATE, src_rate)
        return np.clip(resampled, -32768, 32767).astype(np.int16).tobytes()

    def _build_wss_config(self) -> dict:
        hotword_msg = ""
        if self.keywords:
            hotword_msg = ",".join(kw.strip() for kw in str(self.keywords).split(",") if kw.strip())
        return {
            "mode": self.asr_mode,
            "chunk_size": self.chunk_size,
            "chunk_interval": self.chunk_interval,
            "encoder_chunk_look_back": 4,
            "decoder_chunk_look_back": 0,
            "audio_fs": self.TARGET_SAMPLE_RATE,
            "wav_name": "bolna",
            "wav_format": "pcm",
            "is_speaking": True,
            "hotwords": hotword_msg,
            "itn": self.use_itn,
        }

    async def funasr_connect(self) -> ClientConnection:
        try:
            logger.info(f"Connecting to FunASR websocket: {self.ws_url} (protocol={self.protocol})")
            extra = {}
            if self.protocol == "wss":
                extra["subprotocols"] = ["binary"]
            ws = await asyncio.wait_for(
                websockets.connect(self.ws_url, ssl=get_ssl_context(self.ws_url), **extra),
                timeout=10.0,
            )
            if self.protocol == "realtime":
                await ws.send("START")
            else:
                await ws.send(json.dumps(self._build_wss_config(), ensure_ascii=False))
            self._config_sent = True
            self.websocket_connection = ws
            self.connection_authenticated = True
            logger.info(f"Connected to FunASR (model={self.model}, protocol={self.protocol})")
            return ws
        except asyncio.TimeoutError:
            raise ConnectionError("Timeout while connecting to FunASR websocket")
        except InvalidHandshake as e:
            raise ConnectionError(f"Invalid handshake during FunASR websocket connection: {e}")
        except ConnectionClosedError as e:
            raise ConnectionError(f"FunASR websocket connection closed unexpectedly: {e}")
        except Exception as e:
            raise ConnectionError(f"Unexpected error connecting to FunASR websocket: {e}")

    def _start_turn(self):
        self.turn_counter += 1
        self.current_turn_id = self.turn_counter
        now = timestamp_ms()
        self.current_turn_start_time = now
        self._turn_first_speech_epoch_ms = now
        self.current_turn_interim_details = []
        self.final_transcript = ""
        self.is_transcript_sent_for_processing = False
        self._speech_active = True
        self.turn_latencies.append(
            {
                "turn_id": self.current_turn_id,
                "asr_start_epoch_ms": self.current_turn_start_time,
                "asr_turn_start_epoch_ms": self._turn_first_speech_epoch_ms,
            }
        )
        logger.info(f"FunASR: starting turn {self.current_turn_id}")

    def _reset_turn_state(self):
        self._turn_first_speech_epoch_ms = None
        self.current_turn_interim_details = []
        self.current_turn_start_time = None
        self.current_turn_id = None
        self.final_transcript = ""
        self.last_interim_time = None
        self.is_transcript_sent_for_processing = True
        self._speech_active = False

    def _mark_last_interim_final(self):
        if not self.current_turn_interim_details:
            return
        for entry in self.current_turn_interim_details:
            entry["is_final"] = False
        self.current_turn_interim_details[-1]["is_final"] = True

    def _build_finalized_turn_latency(self, final_transcript):
        self._mark_last_interim_final()
        first_interim_to_final_ms, last_interim_to_final_ms = self.calculate_interim_to_final_latencies(
            self.current_turn_interim_details
        )
        self._upsert_turn_latency(
            {
                "turn_id": self.current_turn_id,
                "asr_start_epoch_ms": self.current_turn_start_time,
                "asr_turn_start_epoch_ms": self._turn_first_speech_epoch_ms,
                "asr_finalized_epoch_ms": timestamp_ms(),
                "final_transcript": final_transcript,
                "interim_details": self.current_turn_interim_details,
                "first_interim_to_final_ms": first_interim_to_final_ms,
                "last_interim_to_final_ms": last_interim_to_final_ms,
            }
        )

    async def _schedule_realtime_commit(self, ws: ClientConnection):
        if self.protocol != "realtime":
            return
        if self._silence_task and not self._silence_task.done():
            self._silence_task.cancel()
            try:
                await self._silence_task
            except asyncio.CancelledError:
                pass

        async def _commit_after_silence():
            try:
                await asyncio.sleep(max(self.endpointing_ms, 100) / 1000.0)
                if self._speech_active and self.connection_on:
                    logger.info("FunASR realtime: COMMIT after client silence")
                    await ws.send("COMMIT")
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"FunASR realtime commit error: {e}")

        self._silence_task = asyncio.create_task(_commit_after_silence())

    async def sender_stream(self, ws: ClientConnection):
        try:
            while True:
                ws_data_packet = await self.input_queue.get()

                if not self.audio_submitted:
                    self.meta_info = ws_data_packet.get("meta_info") or {}
                    self.audio_submitted = True
                    self.current_request_id = self.generate_request_id()
                    self.meta_info["request_id"] = self.current_request_id

                if ws_data_packet.get("meta_info", {}).get("eos") is True:
                    try:
                        if self.protocol == "realtime":
                            if self._speech_active:
                                await ws.send("COMMIT")
                            await ws.send("STOP")
                        else:
                            await ws.send(json.dumps({"is_speaking": False}, ensure_ascii=False))
                    except Exception as e:
                        logger.error(f"Error sending FunASR end-of-stream: {e}")
                    break

                frame_start = self.num_frames * self.audio_frame_duration
                frame_end = (self.num_frames + 1) * self.audio_frame_duration
                self.audio_frame_timestamps.append((frame_start, frame_end, timestamp_ms()))
                self.num_frames += 1

                data = ws_data_packet.get("data")
                if not data:
                    continue
                pcm = self._to_pcm16k(data)
                if not pcm:
                    continue
                try:
                    await ws.send(pcm)
                    await self._schedule_realtime_commit(ws)
                except ConnectionClosedError as e:
                    logger.error(f"FunASR connection closed while sending audio: {e}")
                    break
                except Exception as e:
                    logger.error(f"Error sending audio to FunASR: {e}")
                    break
        except asyncio.CancelledError:
            logger.info("FunASR sender stream task cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in FunASR sender_stream: {e}")
            raise

    def _is_final_message(self, res: dict) -> bool:
        if res.get("is_final") is True:
            return True
        mode = (res.get("mode") or "").lower()
        return mode in ("offline", "2pass-offline")

    def _is_interim_message(self, res: dict) -> bool:
        mode = (res.get("mode") or "").lower()
        if mode in ("online", "2pass-online"):
            return True
        # Realtime protocol often omits mode and uses is_final=false.
        return res.get("is_final") is False and bool(res.get("text"))

    async def receiver(self, ws: ClientConnection):
        async for message in ws:
            try:
                if isinstance(message, bytes):
                    message = message.decode("utf-8")
                res = json.loads(message)

                if self.connection_start_time is None:
                    self.connection_start_time = time.time() - (self.num_frames * self.audio_frame_duration)

                if res.get("error") or res.get("error_code") is not None:
                    self.connection_error = str(res.get("error") or res.get("error_message") or res)
                    logger.error(f"FunASR error: {self.connection_error}")
                    break

                text = (res.get("text") or res.get("transcript") or "").strip()
                if not text and not self._is_final_message(res):
                    continue

                if text and self.current_turn_id is None:
                    self._start_turn()
                    yield create_ws_data_packet("speech_started", self.meta_info)

                if text and self._is_interim_message(res) and not self._is_final_message(res):
                    self.last_interim_time = time.time()
                    running = text
                    # Classic 2pass online is incremental; accumulate for display.
                    if (res.get("mode") or "").lower() in ("online", "2pass-online"):
                        self.final_transcript = text if not self.final_transcript else text
                        running = text
                    self.current_turn_interim_details.append(
                        {
                            "transcript": running,
                            "latency_ms": None,
                            "is_final": False,
                            "received_at": time.time(),
                        }
                    )
                    yield create_ws_data_packet(
                        {"type": "interim_transcript_received", "content": running}, self.meta_info
                    )

                if text and self._is_final_message(res):
                    final = text
                    if not self.is_transcript_sent_for_processing:
                        logger.info(f"FunASR final transcript: {final}")
                        if self.current_turn_id is None:
                            self._start_turn()
                            yield create_ws_data_packet("speech_started", self.meta_info)
                        self.final_transcript = final
                        self._build_finalized_turn_latency(final)
                        lang = res.get("language") or res.get("lang")
                        if lang:
                            self.meta_info["transcriber_detected_language"] = lang
                        yield create_ws_data_packet({"type": "transcript", "content": final}, self.meta_info)
                        self._reset_turn_state()
                    continue

                if res.get("finished"):
                    logger.info("FunASR session finished")
                    break

            except Exception as e:
                logger.error(f"Error processing FunASR message: {e}")
                traceback.print_exc()

    async def push_to_transcriber_queue(self, data_packet):
        await self.transcriber_output_queue.put(data_packet)

    async def toggle_connection(self):
        self.connection_on = False
        if self.sender_task is not None:
            self.sender_task.cancel()
        if self._silence_task is not None:
            self._silence_task.cancel()
        if self.websocket_connection is not None:
            try:
                await self.websocket_connection.close()
            except Exception as e:
                logger.error(f"Error closing FunASR websocket: {e}")
            finally:
                self.websocket_connection = None
                self.connection_authenticated = False

    async def cleanup(self):
        logger.info("Cleaning up FunASR transcriber resources")
        self.connection_on = False
        for task in (self.sender_task, self.transcription_task, self._silence_task):
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.warning(f"Error cancelling FunASR task: {e}")
        if self.websocket_connection is not None:
            try:
                await self.websocket_connection.close()
            except Exception:
                pass
            finally:
                self.websocket_connection = None
                self.connection_authenticated = False
        if self.http_session is not None:
            await self.http_session.close()
            self.http_session = None
        self.audio_frame_timestamps = []
        self.current_turn_interim_details = []
        self._batch_pcm = bytearray()

    def _pcm_to_wav_bytes(self, pcm: bytes) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.TARGET_SAMPLE_RATE)
            wf.writeframes(pcm)
        return buf.getvalue()

    async def _http_transcribe_batch(self, pcm: bytes) -> str:
        if self.http_session is None:
            self.http_session = aiohttp.ClientSession()
        url = f"{self.http_base_url.rstrip('/')}/v1/audio/transcriptions"
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        form = aiohttp.FormData()
        form.add_field("model", self.model)
        form.add_field("language", self.language)
        form.add_field(
            "file",
            self._pcm_to_wav_bytes(pcm),
            filename="audio.wav",
            content_type="audio/wav",
        )
        async with self.http_session.post(url, data=form, headers=headers, timeout=aiohttp.ClientTimeout(total=60)) as resp:
            body = await resp.text()
            if resp.status >= 400:
                raise ConnectionError(f"FunASR HTTP transcription failed ({resp.status}): {body}")
            try:
                payload = json.loads(body)
            except json.JSONDecodeError:
                return body.strip()
            return (payload.get("text") or "").strip()

    async def sender_http(self):
        """Accumulate PCM until EOS, then call OpenAI-compatible FunASR HTTP API."""
        try:
            while True:
                ws_data_packet = await self.input_queue.get()
                if not self.audio_submitted:
                    self.meta_info = ws_data_packet.get("meta_info") or {}
                    self.audio_submitted = True
                    self.current_request_id = self.generate_request_id()
                    self.meta_info["request_id"] = self.current_request_id

                if ws_data_packet.get("meta_info", {}).get("eos") is True:
                    if self._batch_pcm:
                        text = await self._http_transcribe_batch(bytes(self._batch_pcm))
                        if text:
                            self._start_turn()
                            await self.push_to_transcriber_queue(create_ws_data_packet("speech_started", self.meta_info))
                            self._build_finalized_turn_latency(text)
                            await self.push_to_transcriber_queue(
                                create_ws_data_packet({"type": "transcript", "content": text}, self.meta_info)
                            )
                            self._reset_turn_state()
                        self._batch_pcm = bytearray()
                    break

                data = ws_data_packet.get("data")
                if not data:
                    continue
                pcm = self._to_pcm16k(data)
                if pcm:
                    self._batch_pcm.extend(pcm)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Error in FunASR HTTP sender: {e}")
            self.connection_error = str(e)
            raise

    async def run(self):
        try:
            self.transcription_task = asyncio.create_task(self.transcribe())
        except Exception as e:
            logger.error(f"Error starting FunASR transcriber: {e}")

    async def transcribe(self):
        self.num_frames = 0
        self.audio_frame_timestamps = []
        self.connection_start_time = None
        funasr_ws = None
        try:
            if not self.stream:
                self.sender_task = asyncio.create_task(self.sender_http())
                await self.sender_task
                return

            start_time = timestamp_ms()
            try:
                funasr_ws = await self.funasr_connect()
            except (ValueError, ConnectionError) as e:
                logger.error(f"Failed to establish FunASR connection: {e}")
                self.connection_error = str(e)
                await self.toggle_connection()
                return

            if not self.connection_time:
                self.connection_time = round(timestamp_ms() - start_time)

            self.sender_task = asyncio.create_task(self.sender_stream(funasr_ws))
            try:
                async for message in self.receiver(funasr_ws):
                    if self.connection_on:
                        await self.push_to_transcriber_queue(message)
                    else:
                        break
            except ConnectionClosedError as e:
                logger.error(f"FunASR websocket closed during streaming: {e}")
                self.connection_error = str(e)
            except Exception as e:
                logger.error(f"Error during FunASR streaming: {e}")
                self.connection_error = str(e)
                raise
        except (ValueError, ConnectionError) as e:
            logger.error(f"Connection error in FunASR transcribe: {e}")
            self.connection_error = str(e)
            await self.toggle_connection()
        except Exception as e:
            logger.error(f"Unexpected error in FunASR transcribe: {e}")
            self.connection_error = str(e)
            await self.toggle_connection()
        finally:
            if funasr_ws is not None:
                try:
                    await funasr_ws.close()
                except Exception as e:
                    logger.error(f"Error closing FunASR websocket in finally: {e}")
                finally:
                    self.websocket_connection = None
                    self.connection_authenticated = False
            if self.sender_task is not None:
                self.sender_task.cancel()
            meta = dict(getattr(self, "meta_info", None) or {})
            if self.connection_error:
                meta["connection_error"] = self.connection_error
            await self.push_to_transcriber_queue(create_ws_data_packet("transcriber_connection_closed", meta))
