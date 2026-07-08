import asyncio
import json
import os
import time
import traceback
from dotenv import load_dotenv
import websockets
from websockets.asyncio.client import ClientConnection
from websockets.exceptions import ConnectionClosedError, InvalidHandshake

from .base_transcriber import BaseTranscriber
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.ssl_context import get_ssl_context
from bolna.helpers.utils import build_soniox_config, create_ws_data_packet, soniox_ws_url, timestamp_ms
from bolna.enums import TelephonyProvider
from bolna.constants import (
    SONIOX_AUTO_LANGUAGE_VALUES,
    SONIOX_DEFAULT_MULTILINGUAL_HINTS,
    SONIOX_ENDPOINT_TOKEN,
    SONIOX_WEBSOCKET_HOST,
)

logger = configure_logger(__name__)
load_dotenv()


class SonioxTranscriber(BaseTranscriber):
    """Soniox real-time STT over a single WebSocket (config-first-frame, binary audio, token stream)."""

    def __init__(
        self,
        telephony_provider,
        input_queue=None,
        model="stt-rt-v5",
        stream=True,
        language="en",
        endpointing="400",
        sampling_rate="16000",
        encoding="linear16",
        output_queue=None,
        keywords=None,
        process_interim_results="true",
        language_hints=None,
        **kwargs,
    ):
        super().__init__(input_queue)
        self.provider = telephony_provider
        self.model = model
        self.stream = stream
        self.language = language
        self.encoding = encoding
        self.sampling_rate = int(sampling_rate) if isinstance(sampling_rate, (str, int)) else 16000
        self.keywords = keywords
        self.language_hints = language_hints
        self.transcriber_output_queue = output_queue
        self.connected_via_dashboard = kwargs.get("enforce_streaming", True)

        self.api_key = kwargs.get("transcriber_key", os.getenv("SONIOX_API_KEY"))
        self.soniox_host = os.getenv("SONIOX_HOST", SONIOX_WEBSOCKET_HOST)

        # Soniox uses semantic endpoint detection (not a pure silence timer). Leave its own
        # default delay (~2s) unless explicitly tuned, so a brief pause mid-thought isn't cut off.
        _delay = kwargs.get("endpoint_delay_ms")
        self.max_endpoint_delay_ms = max(500, min(3000, int(_delay))) if _delay is not None else None
        _sensitivity = kwargs.get("endpoint_sensitivity")
        self.endpoint_sensitivity = float(_sensitivity) if _sensitivity is not None else None
        self.user_stop_offset_ms = self.max_endpoint_delay_ms if self.max_endpoint_delay_ms is not None else 2000

        # Resolve audio_format / sample_rate / frame duration from the call's I/O provider.
        self.soniox_audio_format = "pcm_s16le"
        self.audio_frame_duration = 0.5
        self._resolve_audio_params()

        # Connection + task state
        self.websocket_connection = None
        self.connection_authenticated = False
        self.sender_task = None
        self.transcription_task = None
        self.connection_error = None

        # Per-stream audio bookkeeping (reset on each (re)connect in transcribe())
        self.audio_submitted = False
        self.num_frames = 0
        self.connection_start_time = None
        # (frame_start_s, frame_end_s, send_ts_ms) per audio frame — maps a token's audio
        # position back to when that audio was sent, for per-result transcriber latency.
        self.audio_frame_timestamps = []

        # Per-turn transcript state
        self.final_transcript = ""
        self.current_turn_id = None
        self.current_turn_start_time = None
        self._turn_first_speech_epoch_ms = None
        self.current_turn_interim_details = []
        self.turn_counter = 0
        self.last_interim_time = None
        self._last_detected_language = None

    def _resolve_audio_params(self):
        """Set encoding, sample rate and frame duration from the telephony/web I/O provider."""
        if self.provider in TelephonyProvider.telephony_values():
            # sip-trunk passes its own encoding/sample_rate from task_config; don't override.
            if self.provider != TelephonyProvider.SIP_TRUNK.value:
                self.encoding = "mulaw" if self.provider == "twilio" else "linear16"
                self.sampling_rate = 8000
            self.audio_frame_duration = 0.2
        elif self.provider == "web_based_call":
            self.encoding = "linear16"
            self.sampling_rate = 16000
            self.audio_frame_duration = 0.256
        elif not self.connected_via_dashboard:
            self.encoding = "linear16"
            self.sampling_rate = 16000

        if self.provider == "playground":
            self.sampling_rate = 8000
            self.audio_frame_duration = 0.0

        self.soniox_audio_format = "mulaw" if self.encoding == "mulaw" else "pcm_s16le"

    def _resolve_language_hints(self):
        """Concrete language -> single hint; multi/auto -> the Indian+English set (or explicit hints)."""
        if self.language_hints:
            return [h for h in self.language_hints if h]
        lang = (self.language or "").lower()
        if lang in SONIOX_AUTO_LANGUAGE_VALUES:
            return SONIOX_DEFAULT_MULTILINGUAL_HINTS
        return [self.language]

    def _build_config(self):
        """First WebSocket frame: auth + stream config (Soniox carries the api_key here, not a header)."""
        terms = [kw.strip() for kw in self.keywords.split(",") if kw.strip()] if self.keywords else []
        return build_soniox_config(
            self.api_key,
            self.model,
            self.soniox_audio_format,
            self.sampling_rate,
            max_endpoint_delay_ms=self.max_endpoint_delay_ms,
            endpoint_sensitivity=self.endpoint_sensitivity,
            language_hints=self._resolve_language_hints() or None,
            context={"terms": terms} if terms else None,
        )

    def get_soniox_ws_url(self):
        return soniox_ws_url(self.soniox_host)

    def get_meta_info(self):
        return self.meta_info

    async def soniox_connect(self):
        """Open the WebSocket and send the config frame; raises ConnectionError on failure."""
        try:
            ws_url = self.get_soniox_ws_url()
            logger.info(f"Connecting to Soniox websocket: {ws_url}")
            soniox_ws = await asyncio.wait_for(
                websockets.connect(ws_url, ssl=get_ssl_context(ws_url)),
                timeout=10.0,
            )
            await soniox_ws.send(json.dumps(self._build_config()))
            self.websocket_connection = soniox_ws
            self.connection_authenticated = True
            logger.info(
                f"Connected to Soniox (model={self.model}, audio_format={self.soniox_audio_format}, "
                f"sample_rate={self.sampling_rate})"
            )
            return soniox_ws
        except asyncio.TimeoutError:
            raise ConnectionError("Timeout while connecting to Soniox websocket")
        except InvalidHandshake as e:
            raise ConnectionError(f"Invalid handshake during Soniox websocket connection: {e}")
        except ConnectionClosedError as e:
            raise ConnectionError(f"Soniox websocket connection closed unexpectedly: {e}")
        except Exception as e:
            raise ConnectionError(f"Unexpected error connecting to Soniox websocket: {e}")

    async def sender_stream(self, ws: ClientConnection):
        """Forward raw audio frames to Soniox; an empty-string frame signals end-of-stream."""
        try:
            while True:
                ws_data_packet = await self.input_queue.get()

                if not self.audio_submitted:
                    self.meta_info = ws_data_packet.get("meta_info")
                    self.audio_submitted = True
                    self.current_request_id = self.generate_request_id()
                    self.meta_info["request_id"] = self.current_request_id

                if ws_data_packet.get("meta_info", {}).get("eos") is True:
                    try:
                        await ws.send("")
                    except Exception as e:
                        logger.error(f"Error sending Soniox end-of-stream: {e}")
                    break

                frame_start = self.num_frames * self.audio_frame_duration
                frame_end = (self.num_frames + 1) * self.audio_frame_duration
                self.audio_frame_timestamps.append((frame_start, frame_end, timestamp_ms()))
                self.num_frames += 1

                data = ws_data_packet.get("data")
                if not data:
                    continue
                try:
                    await ws.send(data)
                except ConnectionClosedError as e:
                    logger.error(f"Soniox connection closed while sending audio: {e}")
                    break
                except Exception as e:
                    logger.error(f"Error sending audio to Soniox: {e}")
                    break
        except asyncio.CancelledError:
            logger.info("Soniox sender stream task cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in Soniox sender_stream: {e}")
            raise

    def _start_turn(self):
        """Begin a new user turn on the first content token and record an eager latency stub."""
        self.turn_counter += 1
        self.current_turn_id = self.turn_counter
        now = timestamp_ms()
        self.current_turn_start_time = now
        self._turn_first_speech_epoch_ms = now
        self.current_turn_interim_details = []
        self.final_transcript = ""
        self.is_transcript_sent_for_processing = False
        self.turn_latencies.append(
            {
                "turn_id": self.current_turn_id,
                "asr_start_epoch_ms": self.current_turn_start_time,
                "asr_turn_start_epoch_ms": self._turn_first_speech_epoch_ms,
            }
        )
        logger.info(f"Soniox: starting new turn with turn_id {self.current_turn_id}")

    def _reset_turn_state(self):
        self._turn_first_speech_epoch_ms = None
        self.current_turn_interim_details = []
        self.current_turn_start_time = None
        self.current_turn_id = None
        self.final_transcript = ""
        self.last_interim_time = None
        self.is_transcript_sent_for_processing = True

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

    def _find_audio_send_timestamp(self, audio_position_s):
        """Send-time (epoch ms) of the audio frame containing this position (seconds), or None."""
        for frame_start, frame_end, send_timestamp in self.audio_frame_timestamps:
            if frame_start <= audio_position_s <= frame_end:
                return send_timestamp
        return None

    async def receiver(self, ws: ClientConnection):
        """Parse the Soniox token stream into speech_started / interim / transcript events."""
        async for message in ws:
            try:
                res = json.loads(message)

                if self.connection_start_time is None:
                    self.connection_start_time = time.time() - (self.num_frames * self.audio_frame_duration)

                if res.get("error_code") is not None:
                    self.connection_error = f"{res.get('error_code')}: {res.get('error_message')}"
                    logger.error(f"Soniox error: {self.connection_error}")
                    break

                endpoint_hit = False
                new_final_text = ""
                non_final_text = ""
                latest_end_ms = None
                for token in res.get("tokens", []):
                    text = token.get("text", "")
                    if not text:
                        continue
                    if text == SONIOX_ENDPOINT_TOKEN:
                        endpoint_hit = True
                        continue
                    if token.get("language"):
                        self._last_detected_language = token.get("language")
                    end_ms = token.get("end_ms")
                    if end_ms is not None:
                        latest_end_ms = end_ms if latest_end_ms is None else max(latest_end_ms, end_ms)
                    if token.get("is_final"):
                        new_final_text += text
                    else:
                        non_final_text += text

                has_content = bool(new_final_text or non_final_text)

                if has_content and self.current_turn_id is None:
                    self._start_turn()
                    yield create_ws_data_packet("speech_started", self.meta_info)

                if new_final_text:
                    self.final_transcript += new_final_text

                if has_content:
                    running = (self.final_transcript + non_final_text).strip()
                    if running:
                        latency_ms = None
                        if latest_end_ms is not None:
                            audio_sent_at = self._find_audio_send_timestamp(latest_end_ms / 1000.0)
                            if audio_sent_at:
                                latency_ms = round(timestamp_ms() - audio_sent_at, 5)
                        self.last_interim_time = time.time()
                        self.current_turn_interim_details.append(
                            {
                                "transcript": running,
                                "latency_ms": latency_ms,
                                "is_final": False,
                                "received_at": time.time(),
                            }
                        )
                        yield create_ws_data_packet(
                            {"type": "interim_transcript_received", "content": running}, self.meta_info
                        )

                if endpoint_hit:
                    final = self.final_transcript.strip()
                    if final and not self.is_transcript_sent_for_processing:
                        logger.info(f"Soniox endpoint reached, yielding transcript: {final}")
                        self._build_finalized_turn_latency(final)
                        self.meta_info["user_stop_offset_ms"] = self.user_stop_offset_ms
                        if self._last_detected_language:
                            self.meta_info["transcriber_detected_language"] = self._last_detected_language
                        yield create_ws_data_packet({"type": "transcript", "content": final}, self.meta_info)
                        self._reset_turn_state()
                    else:
                        # Endpoint with nothing to send — still close the turn so callee_speaking clears.
                        yield create_ws_data_packet({"type": "speech_ended"}, self.meta_info)
                        self._reset_turn_state()

                if res.get("finished"):
                    logger.info("Soniox session finished")
                    break

            except Exception as e:
                logger.error(f"Error processing Soniox message: {e}")
                traceback.print_exc()

    async def push_to_transcriber_queue(self, data_packet):
        await self.transcriber_output_queue.put(data_packet)

    async def toggle_connection(self):
        self.connection_on = False
        if self.sender_task is not None:
            self.sender_task.cancel()
        if self.websocket_connection is not None:
            try:
                await self.websocket_connection.close()
                logger.info("Soniox websocket connection closed")
            except Exception as e:
                logger.error(f"Error closing Soniox websocket: {e}")
            finally:
                self.websocket_connection = None
                self.connection_authenticated = False

    async def cleanup(self):
        """Cancel tasks and close the websocket; clear accumulated per-call data."""
        logger.info("Cleaning up Soniox transcriber resources")
        self.connection_on = False
        for task_name, task in [
            ("sender_task", getattr(self, "sender_task", None)),
            ("transcription_task", getattr(self, "transcription_task", None)),
        ]:
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    logger.info(f"Soniox {task_name} cancelled")
                except Exception as e:
                    logger.warning(f"Error cancelling Soniox {task_name}: {e}")

        if self.websocket_connection is not None:
            try:
                await self.websocket_connection.close()
            except Exception as e:
                logger.error(f"Error closing Soniox websocket: {e}")
            finally:
                self.websocket_connection = None
                self.connection_authenticated = False

        self.audio_frame_timestamps = []
        self.current_turn_interim_details = []

    async def run(self):
        try:
            self.transcription_task = asyncio.create_task(self.transcribe())
        except Exception as e:
            logger.error(f"Error starting Soniox transcriber: {e}")

    async def transcribe(self):
        soniox_ws = None
        # Audio positions restart at 0 on each connection; reset local frame bookkeeping with them.
        self.num_frames = 0
        self.audio_frame_timestamps = []
        self.connection_start_time = None
        try:
            start_time = timestamp_ms()
            try:
                soniox_ws = await self.soniox_connect()
            except (ValueError, ConnectionError) as e:
                logger.error(f"Failed to establish Soniox connection: {e}")
                self.connection_error = str(e)
                await self.toggle_connection()
                return

            if not self.connection_time:
                self.connection_time = round(timestamp_ms() - start_time)

            self.sender_task = asyncio.create_task(self.sender_stream(soniox_ws))

            try:
                async for message in self.receiver(soniox_ws):
                    if self.connection_on:
                        await self.push_to_transcriber_queue(message)
                    else:
                        break
            except ConnectionClosedError as e:
                logger.error(f"Soniox websocket closed during streaming: {e}")
                self.connection_error = str(e)
            except Exception as e:
                logger.error(f"Error during Soniox streaming: {e}")
                self.connection_error = str(e)
                raise

        except (ValueError, ConnectionError) as e:
            logger.error(f"Connection error in Soniox transcribe: {e}")
            self.connection_error = str(e)
            await self.toggle_connection()
        except Exception as e:
            logger.error(f"Unexpected error in Soniox transcribe: {e}")
            self.connection_error = str(e)
            await self.toggle_connection()
        finally:
            if soniox_ws is not None:
                try:
                    await soniox_ws.close()
                except Exception as e:
                    logger.error(f"Error closing Soniox websocket in finally: {e}")
                finally:
                    self.websocket_connection = None
                    self.connection_authenticated = False

            if self.sender_task is not None:
                self.sender_task.cancel()

            meta = dict(getattr(self, "meta_info", None) or {})
            if self.connection_error:
                meta["connection_error"] = self.connection_error
            await self.push_to_transcriber_queue(create_ws_data_packet("transcriber_connection_closed", meta))
