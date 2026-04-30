import asyncio
import base64
import json
import os
import time
import traceback
import audioop
import numpy as np
from scipy.signal import resample_poly
from typing import Optional
from dotenv import load_dotenv
import websockets
from websockets.asyncio.client import ClientConnection
from websockets.exceptions import InvalidHandshake, ConnectionClosed, ConnectionClosedError

from .base_transcriber import BaseTranscriber
from bolna.constants import (
    OPENAI_TRANSCRIBER_HEARTBEAT_INTERVAL_S,
    OPENAI_TRANSCRIBER_UTTERANCE_TIMEOUT_S,
)
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.ssl_context import get_ssl_context
from bolna.helpers.utils import create_ws_data_packet

load_dotenv()
logger = configure_logger(__name__)

_EFFORT_SUPPORTED_MODELS = {"gpt-transcribe-alpha-walrus"}  # only alpha model supports effort


class OpenAITranscriber(BaseTranscriber):
    def __init__(
        self,
        telephony_provider,
        input_queue=None,
        model="gpt-transcribe-alpha-walrus",
        stream=True,
        language="en",
        encoding="linear16",
        sampling_rate="16000",
        output_queue=None,
        endpointing=400,
        effort="medium",
        noise_reduction=False,
        speech_rms_threshold=300,
        **kwargs,
    ):
        super().__init__(input_queue)
        self.telephony_provider = telephony_provider
        self.model = model
        self.language = language
        self.stream = stream
        self.encoding = encoding
        self.sampling_rate = int(sampling_rate)
        self.endpointing_ms = int(endpointing)
        self.effort = effort
        self.noise_reduction = noise_reduction
        self.speech_rms_threshold = speech_rms_threshold

        self.api_host = kwargs.get("transcriber_host", os.getenv("OPENAI_REALTIME_HOST", "api.openai.com"))
        _default_key_env = "OPENAI_API_KEY_EU" if "eu." in self.api_host else "OPENAI_API_KEY"
        self.api_key = kwargs.get("transcriber_key", os.getenv(_default_key_env))

        self.transcriber_output_queue = output_queue
        self.transcription_task = None
        self.sender_task = None
        self.heartbeat_task = None
        self.utterance_timeout_task = None

        self.audio_submitted = False
        self.audio_submission_time = None
        self.num_frames = 0
        self.connection_start_time = None
        self.connection_time = None

        self.websocket_connection = None
        self.connection_authenticated = False
        self.connection_error = None
        self.meta_info = {}

        self.current_turn_id = None
        self.current_turn_start_time = None
        self.turn_counter = 0
        self.current_turn_interim_details = []
        # Saved at commit time so the receiver can still identify the turn after
        # _reset_turn_state() clears current_turn_id (e.g. utterance timeout race).
        self._last_committed_turn_id: Optional[str] = None

        # Wall-clock time when silence began (None while speech is active)
        self._silence_start_time: Optional[float] = None
        self._speech_active = False
        self._first_result_received = False
        # True from the moment audio is appended, False after a commit or before any audio
        self._audio_appended_since_commit = False
        # Signals sender_stream that the final transcript has arrived after EOS
        self._final_transcript_event = asyncio.Event()
        # Set when a turn has been committed; cleared when transcript arrives
        self._turn_committed = False
        # Timestamp of last committed turn (for utterance timeout)
        self._commit_time: Optional[float] = None

        self._configure_audio_params()

    def _configure_audio_params(self):
        if self.telephony_provider == "twilio":
            self.encoding = "mulaw"
            self.input_sampling_rate = 8000
        elif self.telephony_provider in ("plivo", "exotel", "vobiz"):
            self.encoding = "linear16"
            self.input_sampling_rate = 8000
        else:
            self.input_sampling_rate = self.sampling_rate

    def _resample_to_24k(self, audio_bytes: bytes) -> bytes:
        """Convert incoming audio (any rate/encoding) to 24 kHz linear16."""
        if self.encoding == "mulaw":
            audio_bytes = audioop.ulaw2lin(audio_bytes, 2)

        in_rate = self.input_sampling_rate
        if in_rate == 24000:
            return audio_bytes

        try:
            resampled, _ = audioop.ratecv(audio_bytes, 2, 1, in_rate, 24000, None)
            return resampled
        except Exception:
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
            gcd = int(np.gcd(in_rate, 24000))
            resampled_np = resample_poly(audio_np, 24000 // gcd, in_rate // gcd)
            return np.clip(resampled_np, -32768, 32767).astype(np.int16).tobytes()

    @staticmethod
    def _rms(pcm_bytes: bytes) -> float:
        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
        if len(samples) == 0:
            return 0.0
        return float(np.sqrt(np.mean(samples**2)))

    def _reset_turn_state(self):
        """Reset per-turn state after a transcript is delivered.

        If the sender has already started a new turn (_speech_active=True), only
        clear the stale commit-tracking fields from the previous turn — do not
        overwrite current_turn_id or the new turn's interim details.
        """
        if self._speech_active:
            # A new turn is already in progress; only clear leftover commit state
            # from the previous turn so the utterance-timeout monitor is not confused.
            self._turn_committed = False
            self._commit_time = None
            return
        self.current_turn_interim_details = []
        self.current_turn_start_time = None
        self.current_turn_id = None
        self.is_transcript_sent_for_processing = True
        self._turn_committed = False
        self._commit_time = None

    async def openai_connect(self) -> ClientConnection:
        url = f"wss://{self.api_host}/v1/realtime?intent=transcription"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "openai-beta": "realtime=v1",
        }
        try:
            logger.info(f"Attempting to connect to OpenAI Realtime transcription: {url}")
            ws = await asyncio.wait_for(
                websockets.connect(url, additional_headers=headers, ssl=get_ssl_context(url)),
                timeout=10.0,
            )
            self.websocket_connection = ws
            self.connection_authenticated = True

            transcription_cfg = {"model": self.model, "language": self.language}
            if self.effort is not None and self.model in _EFFORT_SUPPORTED_MODELS:
                transcription_cfg["effort"] = self.effort
            session_cfg = {
                "input_audio_format": "pcm16",
                "input_audio_transcription": transcription_cfg,
                "turn_detection": None,
            }
            if self.noise_reduction:
                session_cfg["input_audio_noise_reduction"] = {"type": "near_field"}

            await ws.send(
                json.dumps(
                    {
                        "type": "transcription_session.update",
                        "session": session_cfg,
                    }
                )
            )
            logger.info(
                f"Connected to OpenAI Realtime transcription (model={self.model}, effort={self.effort}, language={self.language})"
            )
            return ws

        except asyncio.TimeoutError:
            raise ConnectionError("Timeout connecting to OpenAI Realtime")
        except InvalidHandshake as e:
            err = str(e)
            if "401" in err or "403" in err:
                raise ConnectionError(f"OpenAI auth failed: {e}")
            raise ConnectionError(f"OpenAI handshake failed: {e}")
        except Exception as e:
            raise ConnectionError(f"OpenAI connection error: {e}")

    async def send_heartbeat(self, ws: ClientConnection):
        """Send WebSocket pings periodically to keep the connection alive."""
        try:
            while True:
                await asyncio.sleep(OPENAI_TRANSCRIBER_HEARTBEAT_INTERVAL_S)
                try:
                    await ws.ping()
                except ConnectionClosed:
                    break
                except Exception as e:
                    logger.error(f"Error sending heartbeat ping: {e}")
                    break
        except asyncio.CancelledError:
            logger.info("OpenAI heartbeat task cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in OpenAI heartbeat: {e}")

    async def monitor_utterance_timeout(self):
        """Force-finalize a committed turn if completed event never arrives."""
        try:
            while True:
                await asyncio.sleep(0.1)
                if (
                    self._turn_committed
                    and self._commit_time is not None
                    and not self.is_transcript_sent_for_processing
                ):
                    elapsed = time.time() - self._commit_time
                    if elapsed > OPENAI_TRANSCRIBER_UTTERANCE_TIMEOUT_S:
                        logger.warning(
                            f"Utterance timeout: completed event missing for {elapsed:.1f}s "
                            f"after commit on turn {self.current_turn_id}. Force-finalizing."
                        )
                        self._final_transcript_event.set()
                        self._reset_turn_state()
        except asyncio.CancelledError:
            logger.info("OpenAI utterance timeout task cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in OpenAI utterance timeout monitor: {e}")

    async def _commit_turn(self, ws: ClientConnection):
        if self.current_turn_id is None:
            # Safety net: a late transcription.completed from the previous turn may have
            # cleared current_turn_id before we got here (shouldn't happen after the
            # _reset_turn_state fix, but guard anyway).
            self.turn_counter += 1
            self.current_turn_id = f"turn_{self.turn_counter}"
            logger.warning(f"_commit_turn: assigned fallback turn_id {self.current_turn_id}")
        self._last_committed_turn_id = self.current_turn_id
        try:
            await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
            self._audio_appended_since_commit = False
            self._turn_committed = True
            self._commit_time = time.time()
            self.meta_info["last_vocal_frame_timestamp"] = self._commit_time
            self.meta_info["user_stop_offset_ms"] = self.endpointing_ms
            self.meta_info["user_stop_ts_wall"] = self._commit_time
            await self.push_to_transcriber_queue(create_ws_data_packet({"type": "speech_ended"}, self.meta_info))
            logger.info(f"Committed turn {self.current_turn_id} after {self.endpointing_ms}ms silence")
        except Exception as e:
            logger.error(f"Error committing turn: {e}")

    async def sender_stream(self, ws: ClientConnection):
        try:
            while True:
                try:
                    ws_data_packet = await asyncio.wait_for(self.input_queue.get(), timeout=0.05)
                except asyncio.TimeoutError:
                    if self._speech_active:
                        if self._silence_start_time is None:
                            self._silence_start_time = time.time()
                        elif (time.time() - self._silence_start_time) * 1000 >= self.endpointing_ms:
                            await self._commit_turn(ws)
                            self._speech_active = False
                            self._silence_start_time = None
                    # Only inject silence while speech is active — after a commit the server
                    # starts a fresh empty buffer, and sending silence into it would cause a
                    # spurious second transcription item when EOS fires its own commit.
                    if self._speech_active:
                        silent_pcm = b"\x00" * int(24000 * 0.05 * 2)  # 50 ms @ 24kHz PCM16
                        await ws.send(
                            json.dumps(
                                {
                                    "type": "input_audio_buffer.append",
                                    "audio": base64.b64encode(silent_pcm).decode(),
                                }
                            )
                        )
                    continue

                if not self.audio_submitted:
                    self.meta_info = ws_data_packet.get("meta_info", {})
                    self.audio_submitted = True
                    self.audio_submission_time = time.time()
                    self.current_request_id = self.generate_request_id()
                    self.meta_info["request_id"] = self.current_request_id

                if await self._check_and_process_end_of_stream(ws_data_packet, ws):
                    # Wait for OpenAI to deliver the final transcript before closing,
                    # mirroring Deepgram's 5s post-CloseStream drain window.
                    try:
                        await asyncio.wait_for(self._final_transcript_event.wait(), timeout=5.0)
                    except asyncio.TimeoutError:
                        logger.warning("Timeout waiting for final transcript after EOS commit")
                    try:
                        await ws.close()
                    except Exception:
                        pass
                    break

                raw_audio = ws_data_packet.get("data")
                if not raw_audio:
                    continue

                pcm_24k = self._resample_to_24k(raw_audio)
                rms = self._rms(pcm_24k)

                if rms > self.speech_rms_threshold:
                    self._silence_start_time = None
                    if not self._speech_active:
                        self._speech_active = True
                        self._first_result_received = False
                        self.audio_submission_time = time.time()
                        self._final_transcript_event.clear()
                        self._audio_appended_since_commit = False
                        self.turn_counter += 1
                        self.current_turn_id = f"turn_{self.turn_counter}"
                        self.current_turn_start_time = time.perf_counter()
                        self.current_turn_interim_details = []
                        self.is_transcript_sent_for_processing = False
                        logger.info(f"Speech detected, starting turn {self.current_turn_id}")
                        await self.push_to_transcriber_queue(create_ws_data_packet("speech_started", self.meta_info))
                else:
                    if self._speech_active:
                        if self._silence_start_time is None:
                            self._silence_start_time = time.time()
                        elif (time.time() - self._silence_start_time) * 1000 >= self.endpointing_ms:
                            await self._commit_turn(ws)
                            self._speech_active = False
                            self._silence_start_time = None

                self.num_frames += 1
                if self._speech_active:
                    self._audio_appended_since_commit = True
                    await ws.send(
                        json.dumps(
                            {
                                "type": "input_audio_buffer.append",
                                "audio": base64.b64encode(pcm_24k).decode(),
                            }
                        )
                    )

        except asyncio.CancelledError:
            logger.info("OpenAI sender_stream task cancelled")
            raise
        except ConnectionClosed:
            logger.info("OpenAI WebSocket closed during sender_stream")
        except Exception as e:
            logger.error(f"Error in OpenAI sender_stream: {e}")
            raise

    async def receiver(self, ws: ClientConnection):
        try:
            async for message in ws:
                try:
                    data = json.loads(message) if isinstance(message, str) else {}
                    event_type = data.get("type", "")

                    if event_type == "conversation.item.input_audio_transcription.delta":
                        delta = data.get("delta", "")
                        if delta:
                            received_at = time.time()
                            if not self._first_result_received and self.audio_submission_time:
                                latency = received_at - self.audio_submission_time
                                self.meta_info["transcriber_first_result_latency"] = latency
                                self.meta_info["transcriber_latency"] = latency
                                self._first_result_received = True
                            self.current_turn_interim_details.append(
                                {
                                    "transcript": delta,
                                    "received_at": received_at,
                                    "is_final": False,
                                }
                            )
                            yield create_ws_data_packet(
                                {"type": "interim_transcript_received", "content": delta},
                                self.meta_info,
                            )

                    elif event_type == "conversation.item.input_audio_transcription.completed":
                        transcript = data.get("transcript", "").strip()
                        item_id = data.get("item_id")
                        if item_id:
                            self.previous_request_id = self.current_request_id
                            self.current_request_id = item_id
                            self.meta_info["request_id"] = item_id
                            self.meta_info["previous_request_id"] = self.previous_request_id

                        # Always unblock EOS drain, even for empty transcripts
                        self._final_transcript_event.set()

                        if transcript:
                            # _last_committed_turn_id is saved at commit time and is the
                            # most reliable identifier — current_turn_id may already point
                            # to the next turn if the sender started speaking immediately.
                            turn_id = self._last_committed_turn_id or self.current_turn_id or item_id
                            logger.info(f"Transcript completed for turn {turn_id}: {transcript[:80]}")
                            if self.current_turn_start_time:
                                total_ms = round((time.perf_counter() - self.current_turn_start_time) * 1000)
                                self.meta_info["transcriber_total_stream_duration"] = total_ms / 1000

                            first_interim_to_final_ms, last_interim_to_final_ms = (
                                self.calculate_interim_to_final_latencies(self.current_turn_interim_details)
                            )
                            self.turn_latencies.append(
                                {
                                    "turn_id": turn_id,
                                    "sequence_id": turn_id,
                                    "interim_details": self.current_turn_interim_details,
                                    "first_interim_to_final_ms": first_interim_to_final_ms,
                                    "last_interim_to_final_ms": last_interim_to_final_ms,
                                    "total_stream_duration_ms": round(
                                        (self.meta_info.get("transcriber_total_stream_duration") or 0) * 1000
                                    ),
                                }
                            )
                            self._reset_turn_state()

                            # Attach turn_latencies only on the final transcript packet,
                            # then remove it so interim packets for the next turn stay small.
                            self.meta_info["turn_latencies"] = self.turn_latencies
                            yield create_ws_data_packet(
                                {"type": "transcript", "content": transcript},
                                self.meta_info,
                            )
                            del self.meta_info["turn_latencies"]

                    elif event_type == "conversation.item.input_audio_transcription.failed":
                        err = data.get("error", {})
                        logger.error(f"Transcription failed for item {data.get('item_id')}: {err.get('message', err)}")
                        self._final_transcript_event.set()
                        self._reset_turn_state()
                        yield create_ws_data_packet(
                            {"type": "transcript_failed", "error": err.get("message", "")},
                            self.meta_info,
                        )

                    elif event_type == "error":
                        logger.error(f"OpenAI Realtime error event: {data.get('error', {})}")

                    elif event_type in (
                        "transcription_session.updated",
                        "session.updated",
                        "transcription_session.created",
                        "session.created",
                    ):
                        logger.info("OpenAI session config accepted")

                except Exception:
                    traceback.print_exc()

        except ConnectionClosedError as e:
            logger.error(f"OpenAI WebSocket closed during receiver: {e}")
        except Exception:
            traceback.print_exc()

    async def _check_and_process_end_of_stream(self, ws_data_packet, ws=None):
        if ws_data_packet and ws_data_packet.get("meta_info", {}).get("eos") is True:
            # Only commit if there is un-committed audio in the buffer; a prior endpointing
            # commit already drained the buffer and reset _audio_appended_since_commit.
            if ws is not None and self._audio_appended_since_commit:
                try:
                    await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
                    self._audio_appended_since_commit = False
                    self._turn_committed = True
                    self._commit_time = time.time()
                except Exception:
                    pass
            return True
        return False

    async def push_to_transcriber_queue(self, data_packet):
        if self.transcriber_output_queue is not None:
            await self.transcriber_output_queue.put(data_packet)

    def get_meta_info(self):
        return getattr(self, "meta_info", {})

    async def toggle_connection(self):
        self.connection_on = False
        for task in [self.heartbeat_task, self.sender_task, self.utterance_timeout_task]:
            if task is not None:
                task.cancel()
        if self.websocket_connection:
            try:
                await self.websocket_connection.close()
            except Exception:
                pass
            finally:
                self.websocket_connection = None
                self.connection_authenticated = False

    async def cleanup(self):
        logger.info("Cleaning up OpenAI transcriber resources")
        for task_name, task in [
            ("sender_task", getattr(self, "sender_task", None)),
            ("heartbeat_task", getattr(self, "heartbeat_task", None)),
            ("utterance_timeout_task", getattr(self, "utterance_timeout_task", None)),
            ("transcription_task", getattr(self, "transcription_task", None)),
        ]:
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    logger.info(f"OpenAI {task_name} cancelled")
                except Exception as e:
                    logger.warning(f"Error cancelling OpenAI {task_name}: {e}")

        if self.websocket_connection is not None:
            try:
                await self.websocket_connection.close()
            except Exception as e:
                logger.error(f"Error closing OpenAI websocket: {e}")
            finally:
                self.websocket_connection = None
                self.connection_authenticated = False

        self.current_turn_interim_details = []

    async def run(self):
        try:
            self.transcription_task = asyncio.create_task(self.transcribe())
        except Exception:
            traceback.print_exc()

    async def transcribe(self):
        ws = None
        try:
            start_time = time.perf_counter()
            try:
                ws = await self.openai_connect()
            except (ValueError, ConnectionError) as e:
                self.connection_error = str(e)
                await self.toggle_connection()
                meta = dict(self.meta_info or {})
                meta["connection_error"] = self.connection_error
                await self.push_to_transcriber_queue(create_ws_data_packet("transcriber_connection_closed", meta))
                return

            if not self.connection_time:
                self.connection_time = round((time.perf_counter() - start_time) * 1000)

            self.sender_task = asyncio.create_task(self.sender_stream(ws))
            self.heartbeat_task = asyncio.create_task(self.send_heartbeat(ws))
            self.utterance_timeout_task = asyncio.create_task(self.monitor_utterance_timeout())

            try:
                async for message in self.receiver(ws):
                    if self.connection_on:
                        await self.push_to_transcriber_queue(message)
                    else:
                        break
            except ConnectionClosedError as e:
                logger.error(f"OpenAI WebSocket closed during streaming: {e}")
                self.connection_error = str(e)
            except Exception as e:
                logger.error(f"Error during OpenAI streaming: {e}")
                self.connection_error = str(e)
                raise

        except (ValueError, ConnectionError) as e:
            self.connection_error = str(e)
            await self.toggle_connection()
        except Exception as e:
            logger.error(f"Unexpected error in OpenAI transcribe: {e}")
            self.connection_error = str(e)
            await self.toggle_connection()
        finally:
            if ws is not None:
                try:
                    await ws.close()
                except Exception:
                    pass
                finally:
                    self.websocket_connection = None
                    self.connection_authenticated = False

            for task in [self.sender_task, self.heartbeat_task, self.utterance_timeout_task]:
                if task is not None:
                    task.cancel()

            meta = dict(self.meta_info or {})
            if self.connection_error:
                meta["connection_error"] = self.connection_error
            await self.push_to_transcriber_queue(create_ws_data_packet("transcriber_connection_closed", meta))
