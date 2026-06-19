import asyncio
import traceback
import os
import json
import aiohttp
import time
from urllib.parse import urlencode, quote
from dotenv import load_dotenv
import websockets
from websockets.asyncio.client import ClientConnection
from websockets.exceptions import ConnectionClosedError, InvalidHandshake, ConnectionClosed

from .base_transcriber import BaseTranscriber
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.ssl_context import get_ssl_context
from bolna.helpers.utils import create_ws_data_packet, timestamp_ms
from bolna.enums import TelephonyProvider
from bolna.constants import (
    DEEPGRAM_FLUX_EOT_THRESHOLD,
    DEEPGRAM_FLUX_EAGER_EOT_THRESHOLD,
    DEEPGRAM_FLUX_EOT_TIMEOUT_MS,
    DEEPGRAM_FLUX_TURN_STALL_FLOOR_S,
)


logger = configure_logger(__name__)
load_dotenv()


class DeepgramTranscriber(BaseTranscriber):
    @property
    def is_english(self):
        return bool(self.language and self.language.startswith("en"))

    def __init__(
        self,
        telephony_provider,
        input_queue=None,
        model="nova-2",
        stream=True,
        language="en",
        endpointing="400",
        sampling_rate="16000",
        encoding="linear16",
        output_queue=None,
        keywords=None,
        process_interim_results="true",
        **kwargs,
    ):
        super().__init__(input_queue)
        self.endpointing = endpointing
        self.endpointing_ms = int(endpointing)
        self.utterance_end_ms = 1000 if self.endpointing_ms < 1000 else self.endpointing_ms
        self.language = language
        self.stream = stream
        self.provider = telephony_provider
        self.heartbeat_task = None
        self.sender_task = None
        self.model = model
        self.sampling_rate = int(sampling_rate) if isinstance(sampling_rate, (str, int)) else 16000
        self.encoding = encoding
        self.api_key = kwargs.get("transcriber_key", os.getenv("DEEPGRAM_AUTH_TOKEN"))
        self.deepgram_host = os.getenv("DEEPGRAM_HOST", "api.deepgram.com")
        self.deepgram_flux_host = os.getenv("DEEPGRAM_FLUX_HOST", "api.deepgram.com")
        self.transcriber_output_queue = output_queue
        self.transcription_task = None
        self.keywords = keywords
        self.transcription_cursor = 0.0
        self.interruption_signalled = False
        self.run_id = kwargs.get("run_id")
        if not self.stream:
            self.api_url = f"https://{self.deepgram_host}/v1/listen?model={self.model}&language={self.language}"
            if self.is_english:
                self.api_url += "&filler_words=true"
            if self.run_id:
                self.api_url += f"&tag={quote(self.run_id)}&extra={quote(f'run_id:{self.run_id}')}"
            self.session = aiohttp.ClientSession()
            if self.keywords is not None:
                keyword_list = [quote(kw.strip()) for kw in self.keywords.split(",") if kw.strip()]
                if keyword_list:
                    if self.model.startswith("nova-3"):
                        keyword_string = "&keyterm=" + "&keyterm=".join(keyword_list)
                    else:
                        keyword_string = "&keywords=" + "&keywords=".join(keyword_list)
                    self.api_url = f"{self.api_url}{keyword_string}"
        self.audio_submitted = False
        self.audio_submission_time = None
        self.num_frames = 0
        self.connection_start_time = None
        self.process_interim_results = process_interim_results
        self.audio_frame_duration = 0.0
        self.connected_via_dashboard = kwargs.get("enforce_streaming", True)
        # Message states
        self.curr_message = ""
        self.finalized_transcript = ""
        self.final_transcript = ""
        self.current_turn_start_time = None
        self.current_turn_id = None
        self.websocket_connection = None
        self.connection_authenticated = False
        self.speech_start_time = None
        self.speech_end_time = None
        self._turn_first_speech_epoch_ms = None  # epoch ms of first SpeechStarted per turn
        self._turn_pending = False  # True after SpeechStarted until first real interim confirms speech
        self.current_turn_interim_details = []
        self.audio_frame_timestamps = []  # List of (frame_start, frame_end, send_timestamp)
        # Wall-clock epoch-ms send time of the audio behind the latest transcript content
        # in the current flux turn — per-turn proxy for "user stopped speaking" (flux has
        # no per-word timestamps). Read at EndOfTurn for user_speech_end_epoch_ms.
        self.last_transcript_audio_sent_at = None
        self.turn_counter = 0
        # Timeout tracking for stuck utterances
        self.last_interim_time = None
        self.interim_timeout = kwargs.get("interim_timeout", 1.2)
        self.utterance_timeout_task = None
        self.connection_error = None

        # Flux model support
        self.is_flux_model = model.startswith("flux-")
        self.is_flux_multi = model == "flux-general-multi"
        _eot_threshold = kwargs.get("eot_threshold")
        self.eot_threshold = _eot_threshold if _eot_threshold is not None else DEEPGRAM_FLUX_EOT_THRESHOLD
        self.eager_eot_threshold = kwargs.get("eager_eot_threshold")
        _eot_timeout_ms = kwargs.get("eot_timeout_ms")
        self.eot_timeout_ms = _eot_timeout_ms if _eot_timeout_ms is not None else DEEPGRAM_FLUX_EOT_TIMEOUT_MS
        # Kept above the normal end-of-turn wait so an ordinary pause isn't treated as a stall.
        self.flux_turn_stall_timeout_s = max(DEEPGRAM_FLUX_TURN_STALL_FLOOR_S, (self.eot_timeout_ms / 1000.0) * 4)
        self.flux_watchdog_task = None
        self._last_flux_msg_time = None
        self.eager_transcript_pending = None
        self.language_hints = kwargs.get("language_hints")
        # ASR-native LID events (flux-general-multi only) — collected per turn and
        # merged into lid_shadow_events.asr_lid_events at call end.
        self.flux_lid_events: list[dict] = []

    def get_deepgram_ws_url(self):
        if self.is_flux_model:
            return self._get_flux_ws_url()
        else:
            return self._get_nova_ws_url()

    def _get_nova_ws_url(self):
        dg_params = {
            "model": self.model,
            # 'diarize': 'true',
            "language": self.language,
            "vad_events": "true",
            "endpointing": self.endpointing,
            "interim_results": "true",
            "utterance_end_ms": str(self.utterance_end_ms),
        }

        self.audio_frame_duration = 0.5  # We're sending 8k samples with a sample rate of 16k

        if self.provider in TelephonyProvider.telephony_values():
            # For sip-trunk (Asterisk), encoding and sampling_rate are already set in task_manager
            # Don't override them - use what was passed from task_config
            if self.provider != TelephonyProvider.SIP_TRUNK.value:
                self.encoding = "mulaw" if self.provider in ("twilio") else "linear16"
                self.sampling_rate = 8000
            # For sip-trunk, encoding and sampling_rate come from task_config (set in task_manager)
            # They're already set from the __init__ parameters, so we don't override
            self.audio_frame_duration = 0.2  # 200ms chunks for telephony

            dg_params["encoding"] = self.encoding
            dg_params["sample_rate"] = self.sampling_rate
            dg_params["channels"] = "1"

            if self.provider == TelephonyProvider.SIP_TRUNK.value:
                logger.info(
                    f"[SIP-TRUNK] Deepgram transcriber configured with encoding={self.encoding}, sample_rate={self.sampling_rate}"
                )

        elif self.provider == "web_based_call":
            dg_params["encoding"] = "linear16"
            dg_params["sample_rate"] = 16000
            dg_params["channels"] = "1"
            self.sampling_rate = 16000
            self.audio_frame_duration = 0.256

        elif not self.connected_via_dashboard:
            dg_params["encoding"] = "linear16"
            dg_params["sample_rate"] = 16000
            dg_params["channels"] = "1"

        if self.provider == "playground":
            self.sampling_rate = 8000
            self.audio_frame_duration = 0.0  # There's no streaming from the playground

        if self.is_english:
            dg_params["filler_words"] = "true"

        if "en" not in self.language:
            dg_params["language"] = self.language

        if self.run_id:
            dg_params["tag"] = self.run_id
            dg_params["extra"] = f"run_id:{self.run_id}"

        websocket_api = "{}://{}/v1/listen?".format(os.getenv("DEEPGRAM_HOST_PROTOCOL", "wss"), self.deepgram_host)
        websocket_url = websocket_api + urlencode(dg_params)

        if self.keywords:
            keyword_list = [quote(kw.strip()) for kw in self.keywords.split(",") if kw.strip()]
            if keyword_list:
                if self.model.startswith("nova-3"):
                    websocket_url += "&keyterm=" + "&keyterm=".join(keyword_list)
                else:
                    websocket_url += "&keywords=" + "&keywords=".join(keyword_list)

        return websocket_url

    def _get_flux_ws_url(self):
        dg_params = {
            "model": self.model,
            "eot_threshold": self.eot_threshold,
            "eot_timeout_ms": self.eot_timeout_ms,
        }

        if self.eager_eot_threshold:
            dg_params["eager_eot_threshold"] = self.eager_eot_threshold

        self.audio_frame_duration = 0.5

        if self.provider in TelephonyProvider.telephony_values():
            if self.provider != TelephonyProvider.SIP_TRUNK.value:
                self.encoding = "mulaw" if self.provider == "twilio" else "linear16"
                self.sampling_rate = 8000
            self.audio_frame_duration = 0.2
            dg_params["encoding"] = self.encoding
            dg_params["sample_rate"] = self.sampling_rate

        elif self.provider == "web_based_call":
            dg_params["encoding"] = "linear16"
            dg_params["sample_rate"] = 16000
            self.sampling_rate = 16000
            self.audio_frame_duration = 0.256

        elif not self.connected_via_dashboard:
            dg_params["encoding"] = "linear16"
            dg_params["sample_rate"] = 16000

        if self.provider == "playground":
            self.sampling_rate = 8000
            self.audio_frame_duration = 0.0

        if self.keywords:
            keyword_list = [kw.strip() for kw in self.keywords.split(",") if kw.strip()]
            if keyword_list:
                dg_params["keyterm"] = keyword_list

        if self.is_flux_multi:
            hints = self._resolve_language_hints()
            if hints:
                dg_params["language_hint"] = hints

        if self.run_id:
            dg_params["tag"] = self.run_id

        websocket_api = "{}://{}/v2/listen?".format(os.getenv("DEEPGRAM_HOST_PROTOCOL", "wss"), self.deepgram_flux_host)
        websocket_url = websocket_api + urlencode(dg_params, doseq=True)
        return websocket_url

    def _resolve_language_hints(self):
        """Resolve language_hint values for flux-general-multi.

        Precedence: explicit language_hints kwarg > derived from self.language.
        Returns None for auto-detect (no hint param sent).
        """
        if self.language_hints:
            return [h for h in self.language_hints if h]
        if not self.language or self.language == "multi":
            return None
        if self.language.startswith("multi-"):
            return [self.language.split("-", 1)[1]]
        return [self.language]

    async def send_heartbeat(self, ws: ClientConnection):
        try:
            while True:
                data = {"type": "KeepAlive"}
                try:
                    await ws.send(json.dumps(data))
                except ConnectionClosed as e:
                    rcvd_code = getattr(e.rcvd, "code", None)
                    sent_code = getattr(e.sent, "code", None)

                    if rcvd_code == 1000 or sent_code == 1000:
                        logger.info("WebSocket closed normally (1000 OK) during heartbeat.")
                    else:
                        logger.error(
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
        except Exception as e:
            logger.error("Error in send_heartbeat: " + str(e))
            raise

    async def send_heartbeat_flux(self, ws: ClientConnection):
        """Flux uses WebSocket ping frames instead of KeepAlive JSON"""
        try:
            while True:
                try:
                    pong_waiter = await ws.ping()
                    await asyncio.wait_for(pong_waiter, timeout=10)
                    logger.debug("Flux heartbeat ping/pong successful")
                except asyncio.TimeoutError:
                    logger.warning("Flux heartbeat ping timeout - connection may be stale")
                    break
                except ConnectionClosed as e:
                    rcvd_code = getattr(e.rcvd, "code", None)
                    sent_code = getattr(e.sent, "code", None)

                    if rcvd_code == 1000 or sent_code == 1000:
                        logger.info("WebSocket closed normally (1000 OK) during Flux heartbeat.")
                    else:
                        logger.error(f"WebSocket closed during Flux heartbeat: received={rcvd_code}, sent={sent_code}")
                    break
                except Exception as e:
                    logger.error(f"Error sending Flux heartbeat ping: {e}")
                    break

                await asyncio.sleep(5)  # Send ping every 5 seconds
        except asyncio.CancelledError:
            logger.error("Flux heartbeat task cancelled")
        except Exception as e:
            logger.error(f"Error in send_heartbeat_flux: {e}")
            raise

    def _reset_turn_state(self):
        """Reset turn state variables after finalizing a transcript"""
        self.speech_start_time = None
        self.speech_end_time = None
        self._turn_first_speech_epoch_ms = None
        self._turn_pending = False
        self.last_interim_time = None
        self.current_turn_interim_details = []
        self.last_transcript_audio_sent_at = None
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
            transcript_to_send = self.current_turn_interim_details[-1]["transcript"]
            logger.info(f"Using last interim as fallback: {transcript_to_send}")

        if not transcript_to_send:
            logger.warning("No transcript available to force-finalize")
            self._reset_turn_state()
            return

        # Build turn latencies (same as UtteranceEnd logic)
        try:
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
                    "final_transcript": transcript_to_send,
                    "interim_details": self.current_turn_interim_details,
                    "first_interim_to_final_ms": first_interim_to_final_ms,
                    "last_interim_to_final_ms": last_interim_to_final_ms,
                    "force_finalized": True,
                }
            )
        except Exception as e:
            logger.error(f"Error building turn latencies: {e}")

        # Create transcript message (same format as UtteranceEnd)
        data = {
            "type": "transcript",
            "content": transcript_to_send,
            "force_finalized": True,  # For debugging
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
                if (
                    self.last_interim_time
                    and not self.is_transcript_sent_for_processing
                    and (self.final_transcript.strip() or self.current_turn_interim_details)
                ):
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

    def _flux_turn_is_stalled(self, now):
        """True if a turn is open (interim seen) but no Flux event arrived within the stall window."""
        if self.last_interim_time is None or self._last_flux_msg_time is None:
            return False
        return (now - self._last_flux_msg_time) > self.flux_turn_stall_timeout_s

    async def _release_stuck_flux_turn(self):
        # speech_ended ends the user turn downstream (resets callee_speaking) without injecting a
        # transcript or LLM call; _reset_turn_state then drops any later finalization for this turn.
        await self.push_to_transcriber_queue(create_ws_data_packet({"type": "speech_ended"}, self.meta_info))
        self._reset_turn_state()

    async def monitor_flux_turn_timeout(self):
        """Force-close a Flux turn that stays open without any closing event, so a missing
        EndOfTurn cannot leave the user turn open and the agent's audio held indefinitely."""
        try:
            while True:
                await asyncio.sleep(1.0)
                if self._flux_turn_is_stalled(time.time()):
                    logger.warning(
                        f"Flux turn stall: no event for >{self.flux_turn_stall_timeout_s:.1f}s "
                        f"(turn {self.current_turn_id}), releasing via speech_ended."
                    )
                    await self._release_stuck_flux_turn()
        except asyncio.CancelledError:
            logger.info("Flux turn timeout monitoring task cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in monitor_flux_turn_timeout: {e}")
            raise

    async def toggle_connection(self):
        self.connection_on = False
        if self.heartbeat_task is not None:
            self.heartbeat_task.cancel()
        if self.sender_task is not None:
            self.sender_task.cancel()
        if self.utterance_timeout_task is not None:
            self.utterance_timeout_task.cancel()
        if self.flux_watchdog_task is not None:
            self.flux_watchdog_task.cancel()

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
        if hasattr(self, "session") and self.session and not self.session.closed:
            try:
                await self.session.close()
                logger.info("Deepgram HTTP session closed")
            except Exception as e:
                logger.error(f"Error closing Deepgram HTTP session: {e}")

        # Cancel tasks properly
        for task_name, task in [
            ("heartbeat_task", getattr(self, "heartbeat_task", None)),
            ("sender_task", getattr(self, "sender_task", None)),
            ("utterance_timeout_task", getattr(self, "utterance_timeout_task", None)),
            ("flux_watchdog_task", getattr(self, "flux_watchdog_task", None)),
            ("transcription_task", getattr(self, "transcription_task", None)),
        ]:
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    logger.info(f"Deepgram {task_name} cancelled")
                except Exception as e:
                    logger.warning(f"Error cancelling Deepgram {task_name}: {e}")

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

        # Always clear accumulated per-call data to prevent memory leaks
        self.audio_frame_timestamps = []
        self.current_turn_interim_details = []

    async def _get_http_transcription(self, audio_data):
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()

        headers = {
            "Authorization": "Token {}".format(self.api_key),
            "Content-Type": "audio/webm",  # Currently we are assuming this is via browser
        }

        self.current_request_id = self.generate_request_id()
        self.meta_info["request_id"] = self.current_request_id
        async with self.session as session:
            async with session.post(self.api_url, data=audio_data, headers=headers) as response:
                response_data = await response.json()
                transcript = response_data["results"]["channels"][0]["alternatives"][0]["transcript"]
                self.meta_info["transcriber_duration"] = response_data["metadata"]["duration"]
                return create_ws_data_packet(transcript, self.meta_info)

    async def _check_and_process_end_of_stream(self, ws_data_packet, ws):
        if "eos" in ws_data_packet["meta_info"] and ws_data_packet["meta_info"]["eos"] is True:
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
                        self.meta_info = ws_data_packet.get("meta_info") if self.meta_info is None else self.meta_info
                        if self.meta_info is not None and not self.current_turn_start_time:
                            self.current_turn_start_time = timestamp_ms()
                            self.current_turn_id = self.meta_info.get("turn_id") or self.meta_info.get("request_id")
                    except Exception:
                        pass
                end_of_stream = await self._check_and_process_end_of_stream(ws_data_packet, ws)
                if end_of_stream:
                    break
                self.meta_info = ws_data_packet.get("meta_info")
                start_time = timestamp_ms()
                transcription = await self._get_http_transcription(ws_data_packet.get("data"))
                transcription["meta_info"]["include_latency"] = True
                # HTTP path: first and total are same
                try:
                    elapsed = timestamp_ms() - start_time
                    transcription["meta_info"]["transcriber_first_result_latency"] = elapsed
                    transcription["meta_info"]["transcriber_total_stream_duration"] = elapsed
                    transcription["meta_info"]["transcriber_latency"] = elapsed
                except Exception:
                    pass
                transcription["meta_info"]["audio_duration"] = transcription["meta_info"]["transcriber_duration"]
                transcription["meta_info"]["last_vocal_frame_timestamp"] = time.time()
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
                    self.meta_info = ws_data_packet.get("meta_info")
                    self.audio_submitted = True
                    self.audio_submission_time = time.time()
                    self.current_request_id = self.generate_request_id()
                    self.meta_info["request_id"] = self.current_request_id
                    try:
                        if not self.current_turn_start_time:
                            self.current_turn_start_time = timestamp_ms()
                            self.current_turn_id = self.meta_info.get("turn_id") or self.meta_info.get("request_id")
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
                    await ws.send(ws_data_packet.get("data"))
                except ConnectionClosedError as e:
                    logger.error(f"Connection closed while sending data: {e}")
                    break
                except Exception as e:
                    logger.error(f"Error sending data to websocket: {e}")
                    break

        except asyncio.CancelledError:
            logger.info("Sender stream task cancelled")
            # Stamp cancellation on the open ASR stub if a turn was in progress
            if self.current_turn_id is not None:
                self._upsert_turn_latency(
                    {
                        "turn_id": self.current_turn_id,
                        "asr_start_epoch_ms": self.current_turn_start_time,
                        "asr_turn_start_epoch_ms": self._turn_first_speech_epoch_ms,
                        "cancelled_at_ms": timestamp_ms(),
                    }
                )
            raise
        except Exception as e:
            logger.error("Error in sender_stream: " + str(e))
            raise

    async def receiver(self, ws: ClientConnection):
        async for msg in ws:
            try:
                msg = json.loads(msg)

                # If connection_start_time is None, it is the durations of frame submitted till now minus current time
                if self.connection_start_time is None:
                    self.connection_start_time = time.time() - (self.num_frames * self.audio_frame_duration)

                if msg["type"] == "SpeechStarted":
                    logger.info("Received SpeechStarted event from deepgram")
                    if not isinstance(self.current_turn_id, int):
                        self._turn_first_speech_epoch_ms = timestamp_ms()
                        self._turn_pending = True  # counter incremented on first real interim
                    self.speech_start_time = timestamp_ms()
                    self.current_turn_interim_details = []
                    self.is_transcript_sent_for_processing = False

                    logger.info(f"Starting new turn with turn_id: {self.current_turn_id}")
                    logger.info(
                        "BOLNA_TRACE_DG speech_started dg_turn=%s request_id=%s",
                        self.current_turn_id,
                        self.meta_info.get("request_id"),
                    )
                    yield create_ws_data_packet("speech_started", self.meta_info)

                elif msg["type"] == "Results":
                    transcript = msg["channel"]["alternatives"][0]["transcript"]
                    deepgram_request_id = msg.get("metadata", {}).get("request_id")

                    if transcript.strip():
                        if self._turn_pending:
                            self.turn_counter += 1
                            self.current_turn_id = self.turn_counter
                            self._turn_pending = False
                            logger.info(f"Starting new turn with turn_id: {self.current_turn_id}")
                            # Eager stub — captured even if turn is never finalized (e.g. transcriber drop)
                            self.turn_latencies.append(
                                {
                                    "turn_id": self.current_turn_id,
                                    "asr_start_epoch_ms": self.current_turn_start_time,
                                    "asr_turn_start_epoch_ms": self._turn_first_speech_epoch_ms,
                                }
                            )
                        elif self.current_turn_id is None:
                            # SpeechStarted was suppressed (e.g. Deepgram VAD inhibited during
                            # agent audio playback) but real speech arrived — still assign a turn_id.
                            self.turn_counter += 1
                            self.current_turn_id = self.turn_counter
                            logger.info(f"Turn id assigned without SpeechStarted: {self.current_turn_id}")
                            # Eager stub for turns where SpeechStarted was suppressed
                            self.turn_latencies.append(
                                {
                                    "turn_id": self.current_turn_id,
                                    "asr_start_epoch_ms": self.current_turn_start_time,
                                    "asr_turn_start_epoch_ms": self._turn_first_speech_epoch_ms,
                                }
                            )
                        # Calculate latency using end position (start + duration) for cumulative transcripts
                        self.__set_transcription_cursor(msg)
                        audio_position_end = self.transcription_cursor
                        latency_ms = None

                        audio_sent_at = self._find_audio_send_timestamp(audio_position_end)
                        if audio_sent_at:
                            result_received_at = timestamp_ms()
                            latency_ms = round(result_received_at - audio_sent_at, 5)

                        interim_detail = {
                            "transcript": transcript,
                            "latency_ms": latency_ms,
                            "is_final": msg.get("is_final", False),
                            "received_at": time.time(),
                            "request_id": deepgram_request_id,
                        }

                        logger.info(
                            f"Interim result - request_id: {deepgram_request_id}, is_final: {msg.get('is_final', False)}, transcript: {transcript}"
                        )

                        self.current_turn_interim_details.append(interim_detail)
                        # Track time of last interim for timeout monitoring
                        self.last_interim_time = time.time()

                        data = {"type": "interim_transcript_received", "content": transcript}
                        yield create_ws_data_packet(data, self.meta_info)

                    if msg["is_final"] and transcript.strip():
                        logger.info(f"Received interim result with is_final set as True - {transcript}")
                        self.final_transcript += f" {transcript}"
                        logger.info(
                            "BOLNA_TRACE_DG result_final dg_turn=%s request_id=%s speech_final=%s final_len=%s text=%r",
                            self.current_turn_id,
                            deepgram_request_id,
                            msg.get("speech_final", False),
                            len(self.final_transcript.strip()),
                            transcript[:80],
                        )

                        if self.is_transcript_sent_for_processing:
                            self.is_transcript_sent_for_processing = False

                    if msg["speech_final"] and self.final_transcript.strip():
                        if not self.is_transcript_sent_for_processing and self.final_transcript.strip():
                            logger.info(
                                f"Received speech final hence yielding the following transcript - {self.final_transcript}"
                            )
                            logger.info(
                                "BOLNA_TRACE_DG emit_speech_final dg_turn=%s request_id=%s text_len=%s text=%r",
                                self.current_turn_id,
                                deepgram_request_id,
                                len(self.final_transcript.strip()),
                                self.final_transcript.strip()[:120],
                            )

                            data = {"type": "transcript", "content": self.final_transcript}

                            # Build turn_latencies with new metrics before resetting
                            try:
                                first_interim_to_final_ms, last_interim_to_final_ms = (
                                    self.calculate_interim_to_final_latencies(self.current_turn_interim_details)
                                )

                                self._upsert_turn_latency(
                                    {
                                        "turn_id": self.current_turn_id,
                                        "asr_start_epoch_ms": self.current_turn_start_time,
                                        "asr_turn_start_epoch_ms": self._turn_first_speech_epoch_ms,
                                        "asr_finalized_epoch_ms": timestamp_ms(),
                                        "final_transcript": self.final_transcript,
                                        "interim_details": self.current_turn_interim_details,
                                        "first_interim_to_final_ms": first_interim_to_final_ms,
                                        "last_interim_to_final_ms": last_interim_to_final_ms,
                                    }
                                )

                                # Complete turn reset
                                self.speech_start_time = None
                                self.speech_end_time = None
                                self._turn_first_speech_epoch_ms = None
                                self.current_turn_interim_details = []
                                self.current_turn_start_time = None
                                self.current_turn_id = None
                                self.final_transcript = ""
                                self.is_transcript_sent_for_processing = True
                            except Exception as e:
                                logger.error(
                                    f"Failed to extract transcript from Deepgram response in speech_final: {e}"
                                )
                                pass
                            self.meta_info["user_stop_offset_ms"] = self.endpointing_ms
                            # Always assign (even None) to clear any stale value from a previous turn.
                            # None is safe: interruption_manager guards on it before use.
                            self.meta_info["user_stop_ts_wall"] = self._compute_last_word_end_wall(msg)
                            yield create_ws_data_packet(data, self.meta_info)

                elif msg["type"] == "UtteranceEnd":
                    logger.info(
                        f"Value of is_transcript_sent_for_processing in utterance end - {self.is_transcript_sent_for_processing}"
                    )
                    if not self.is_transcript_sent_for_processing and self.final_transcript.strip():
                        logger.info(
                            f"Received UtteranceEnd hence yielding the following transcript - {self.final_transcript}"
                        )
                        logger.info(
                            "BOLNA_TRACE_DG emit_utterance_end dg_turn=%s request_id=%s text_len=%s text=%r",
                            self.current_turn_id,
                            self.meta_info.get("request_id"),
                            len(self.final_transcript.strip()),
                            self.final_transcript.strip()[:120],
                        )

                        data = {"type": "transcript", "content": self.final_transcript}

                        # Build turn_latencies with new metrics before resetting
                        try:
                            first_interim_to_final_ms, last_interim_to_final_ms = (
                                self.calculate_interim_to_final_latencies(self.current_turn_interim_details)
                            )

                            self._upsert_turn_latency(
                                {
                                    "turn_id": self.current_turn_id,
                                    "asr_start_epoch_ms": self.current_turn_start_time,
                                    "asr_turn_start_epoch_ms": self._turn_first_speech_epoch_ms,
                                    "asr_finalized_epoch_ms": timestamp_ms(),
                                    "final_transcript": self.final_transcript,
                                    "interim_details": self.current_turn_interim_details,
                                    "first_interim_to_final_ms": first_interim_to_final_ms,
                                    "last_interim_to_final_ms": last_interim_to_final_ms,
                                }
                            )

                            # Complete turn reset
                            self.speech_start_time = None
                            self.speech_end_time = None
                            self._turn_first_speech_epoch_ms = None
                            self.current_turn_interim_details = []
                            self.current_turn_start_time = None
                            self.current_turn_id = None
                            self.final_transcript = ""
                            self.is_transcript_sent_for_processing = True
                        except Exception as e:
                            logger.error(f"Failed to extract transcript from Deepgram response: {e}")
                            pass
                        self.meta_info["user_stop_offset_ms"] = self.utterance_end_ms
                        last_word_end_audio = msg.get("last_word_end")
                        if last_word_end_audio is not None:
                            self.meta_info["user_stop_ts_wall"] = self.connection_start_time + last_word_end_audio
                        yield create_ws_data_packet(data, self.meta_info)
                    else:
                        # Transcript already sent but we still need to notify speech ended
                        # This prevents callee_speaking from staying True indefinitely
                        logger.info(
                            f"UtteranceEnd received but transcript already processed, yielding speech_ended notification"
                        )
                        logger.info(
                            "BOLNA_TRACE_DG emit_speech_ended request_id=%s",
                            self.meta_info.get("request_id"),
                        )
                        self.speech_start_time = None
                        self.speech_end_time = None
                        self._turn_first_speech_epoch_ms = None
                        self._turn_pending = False
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

    async def receiver_flux(self, ws: ClientConnection):
        async for msg in ws:
            try:
                msg = json.loads(msg)

                # Track liveness off every message (any type), so the stall watchdog also covers
                # turns that only produced an Update and never a StartOfTurn.
                self._last_flux_msg_time = time.time()

                if self.connection_start_time is None:
                    self.connection_start_time = time.time() - (self.num_frames * self.audio_frame_duration)

                if msg["type"] == "Connected":
                    logger.info(f"Connected to Deepgram Flux: request_id={msg.get('request_id')}")
                    continue

                elif msg["type"] == "TurnInfo":
                    event = msg.get("event")
                    transcript = msg.get("transcript", "").strip().rstrip(",.'?|'!।")
                    turn_index = msg.get("turn_index")
                    eot_confidence = msg.get("end_of_turn_confidence")
                    words = msg.get("words", [])
                    audio_window_end = msg.get("audio_window_end")
                    # flux-general-multi: languages detected this turn, sorted by word count
                    languages = msg.get("languages")
                    languages_hinted = msg.get("languages_hinted")

                    if event == "StartOfTurn":
                        logger.info(f"Flux: StartOfTurn (turn_index={turn_index}, transcript={transcript!r})")
                        self.turn_counter += 1
                        self.current_turn_id = self.turn_counter
                        self.speech_start_time = timestamp_ms()
                        self.current_turn_interim_details = []
                        self.last_transcript_audio_sent_at = None
                        self.is_transcript_sent_for_processing = False
                        self.final_transcript = ""
                        # Eager stub — captured even if turn is never finalized
                        self.turn_latencies.append(
                            {
                                "turn_id": self.current_turn_id,
                                "asr_start_epoch_ms": self.speech_start_time,
                                "asr_turn_start_epoch_ms": self.speech_start_time,
                            }
                        )
                        yield create_ws_data_packet("speech_started", self.meta_info)
                        # StartOfTurn is guaranteed non-empty — use it immediately for barge-in
                        # instead of waiting for the first Update (~0.25s later)
                        if transcript:
                            # Seed current_turn_interim_details so short turns (StartOfTurn → EndOfTurn
                            # with no Update events) still produce first/last_interim_to_final_ms in
                            # turn_latencies, which feeds voiceai.transcriber.* DD distributions.
                            entry = self._build_interim_entry(transcript, words, msg)
                            self.last_interim_time = entry["received_at"]
                            self.current_turn_interim_details.append(entry)
                            data = {"type": "interim_transcript_received", "content": transcript}
                            yield create_ws_data_packet(data, self.meta_info)

                    elif event == "Update":
                        if transcript:
                            entry = self._build_interim_entry(transcript, words, msg)
                            self.last_interim_time = entry["received_at"]
                            self.current_turn_interim_details.append(entry)

                            if languages:
                                logger.info(f"Flux LID Update: languages={languages} hinted={languages_hinted}")
                                self.flux_lid_events.append(
                                    {
                                        "detected_lang": languages[0],
                                        "all_languages": languages,
                                        "event_type": "Update",
                                        "turn_index": turn_index,
                                        "transcript": transcript,
                                        "lid_provider": "deepgram_flux",
                                        "detected_at": time.time(),
                                    }
                                )

                            data = {"type": "interim_transcript_received", "content": transcript}
                            yield create_ws_data_packet(data, self.meta_info)

                    elif event == "EagerEndOfTurn":
                        if languages:
                            logger.info(f"Flux LID EagerEndOfTurn: languages={languages} hinted={languages_hinted}")
                            self.flux_lid_events.append(
                                {
                                    "detected_lang": languages[0],
                                    "all_languages": languages,
                                    "event_type": "EagerEndOfTurn",
                                    "turn_index": turn_index,
                                    "transcript": transcript,
                                    "lid_provider": "deepgram_flux",
                                    "detected_at": time.time(),
                                }
                            )
                        logger.info(f"Flux: EagerEndOfTurn (confidence={eot_confidence}, transcript={transcript!r})")
                        if transcript:
                            self.eager_transcript_pending = transcript
                            self.last_interim_time = time.time()

                            eager_latency_ms = None
                            if words and audio_window_end:
                                audio_sent_at = self._find_audio_send_timestamp(audio_window_end)
                                if audio_sent_at:
                                    eager_latency_ms = round(timestamp_ms() - audio_sent_at, 5)
                            self._mark_last_interim_final(latency_ms=eager_latency_ms)

                            data = {"type": "eager_end_of_turn", "content": transcript, "confidence": eot_confidence}
                            yield create_ws_data_packet(data, self.meta_info)
                        else:
                            logger.warning("Flux: EagerEndOfTurn received with empty transcript, ignoring")

                    elif event == "TurnResumed":
                        logger.info(f"Flux: TurnResumed - user continued speaking after EagerEndOfTurn")
                        self.eager_transcript_pending = None

                        data = {"type": "turn_resumed"}
                        yield create_ws_data_packet(data, self.meta_info)

                    elif event == "EndOfTurn":
                        if languages:
                            logger.info(f"Flux LID EndOfTurn: languages={languages} hinted={languages_hinted}")
                            self.flux_lid_events.append(
                                {
                                    "detected_lang": languages[0],
                                    "all_languages": languages,
                                    "event_type": "EndOfTurn",
                                    "turn_index": turn_index,
                                    "transcript": transcript,
                                    "lid_provider": "deepgram_flux",
                                    "detected_at": time.time(),
                                }
                            )
                        logger.info(f"Flux: EndOfTurn (confidence={eot_confidence}) transcript={transcript!r}")

                        if transcript and not self.is_transcript_sent_for_processing:
                            try:
                                if not self.current_turn_interim_details:
                                    logger.warning(
                                        "Flux: EndOfTurn has transcript but no interim_details to mark final "
                                        "(turn_id=%s, transcript=%r)",
                                        self.current_turn_id,
                                        transcript,
                                    )
                                self._mark_last_interim_final()
                                first_interim_to_final_ms, last_interim_to_final_ms = (
                                    self.calculate_interim_to_final_latencies(self.current_turn_interim_details)
                                )
                                asr_finalized_epoch_ms = timestamp_ms()
                                turn_latency = {
                                    "turn_id": self.current_turn_id,
                                    "sequence_id": self.current_turn_id,
                                    "interim_details": self.current_turn_interim_details,
                                    "first_interim_to_final_ms": first_interim_to_final_ms,
                                    "last_interim_to_final_ms": last_interim_to_final_ms,
                                    "asr_start_epoch_ms": self.speech_start_time,
                                    "asr_turn_start_epoch_ms": self.speech_start_time,
                                    "asr_finalized_epoch_ms": asr_finalized_epoch_ms,
                                    "final_transcript": transcript,
                                }
                                # Observability only: asr_finalized minus user_speech_end measures
                                # flux's end-of-turn detection delay. Omit when missing or out of
                                # order (frame-mapping anomaly, e.g. post-reconnect) — absent beats
                                # wrong for a latency metric.
                                user_speech_end_epoch_ms = self.last_transcript_audio_sent_at
                                if (
                                    user_speech_end_epoch_ms is not None
                                    and user_speech_end_epoch_ms <= asr_finalized_epoch_ms
                                ):
                                    turn_latency["user_speech_end_epoch_ms"] = user_speech_end_epoch_ms
                                self._upsert_turn_latency(turn_latency)
                            except Exception as e:
                                logger.error(f"Error building turn latencies: {e}")

                            data = {
                                "type": "transcript",
                                "content": transcript,
                                "was_eager": self.eager_transcript_pending is not None,
                            }

                            self._reset_turn_state()
                            self.eager_transcript_pending = None

                            yield create_ws_data_packet(data, self.meta_info)
                        else:
                            if transcript:
                                logger.warning(
                                    f"Flux: EndOfTurn suppressed — transcript already sent for processing "
                                    f"(turn_id={self.current_turn_id}, transcript={transcript!r})"
                                )
                                # Eager path already fired — still mark is_final so the DD FINAL metric
                                # counts this turn.
                                self._mark_last_interim_final()
                            else:
                                logger.warning("Flux: EndOfTurn received with empty transcript")
                            if self.eager_transcript_pending is not None:
                                # EagerEndOfTurn fired but the turn produced nothing — cancel speculative LLM
                                logger.info("Flux: cancelling speculative LLM task via turn_resumed (empty EndOfTurn)")
                                yield create_ws_data_packet({"type": "turn_resumed"}, self.meta_info)
                            self._reset_turn_state()
                            self.eager_transcript_pending = None

                elif msg["type"] == "Error":
                    error_code = msg.get("code", "unknown")
                    error_desc = msg.get("description", "No description")
                    logger.error(f"Flux error: {error_code} - {error_desc}")

            except Exception as e:
                traceback.print_exc()
                logger.error(f"Error processing Flux message: {e}")

    async def push_to_transcriber_queue(self, data_packet):
        await self.transcriber_output_queue.put(data_packet)

    async def deepgram_connect(self):
        """Establish websocket connection to Deepgram with proper error handling"""
        try:
            websocket_url = self.get_deepgram_ws_url()
            additional_headers = {"Authorization": "Token {}".format(self.api_key)}

            logger.info(f"Attempting to connect to Deepgram websocket: {websocket_url}")

            deepgram_ws = await asyncio.wait_for(
                websockets.connect(
                    websocket_url, additional_headers=additional_headers, ssl=get_ssl_context(websocket_url)
                ),
                timeout=10.0,  # 10 second timeout
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

    def _compute_last_word_end_wall(self, data):
        """Wall-clock seconds when the last transcribed word ended, or None."""
        try:
            alternatives = (data.get("channel") or {}).get("alternatives") or []
            for alternative in alternatives:
                words = alternative.get("words") or []
                if words:
                    return self.connection_start_time + words[-1]["end"]
        except Exception:
            pass
        return None

    def __set_transcription_cursor(self, data):
        if "start" in data and "duration" in data:
            self.transcription_cursor = data["start"] + data["duration"]
            logger.info(
                f"Setting transcription cursor at {self.transcription_cursor} (start={data['start']}, duration={data['duration']})"
            )
        else:
            logger.warning(f"Missing start or duration in Deepgram message, cannot update transcription cursor")
        return self.transcription_cursor

    def _mark_last_interim_final(self, latency_ms=None):
        """Mark the last interim entry as final and optionally update its latency.
        Clears prior is_final flags first — prevents double-counting on the
        EagerEndOfTurn → TurnResumed → Updates → EndOfTurn path where an earlier
        entry was already marked final by the eager handler.
        Called at every turn-finalization path so the DD FINAL metric fires consistently."""
        if not self.current_turn_interim_details:
            return
        for entry in self.current_turn_interim_details:
            entry["is_final"] = False
        last = self.current_turn_interim_details[-1]
        last["is_final"] = True
        if latency_ms is not None:
            last["latency_ms"] = latency_ms

    def _build_interim_entry(self, transcript, words, msg):
        now = time.time()
        latency_ms = None
        if words:
            audio_sent_at = self._find_audio_send_timestamp(msg.get("audio_window_end", 0))
            if audio_sent_at:
                latency_ms = round(now * 1000 - audio_sent_at, 5)
                self.last_transcript_audio_sent_at = audio_sent_at
        return {"transcript": transcript, "latency_ms": latency_ms, "is_final": False, "received_at": now}

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
        # Per-connection stream state: Deepgram audio positions (audio_window_end,
        # word offsets) restart at 0 on every new websocket, so the local frame
        # bookkeeping must restart with them. A pool reconnect re-runs transcribe()
        # on the same instance — stale values from the previous connection would
        # map new positions onto old wall-clock times.
        self.num_frames = 0
        self.audio_frame_timestamps = []
        self.connection_start_time = None
        try:
            start_time = timestamp_ms()
            try:
                deepgram_ws = await self.deepgram_connect()
            except (ValueError, ConnectionError) as e:
                logger.error(f"Failed to establish Deepgram connection: {e}")
                self.connection_error = str(e)
                await self.toggle_connection()
                return

            if not self.connection_time:
                self.connection_time = round(timestamp_ms() - start_time)

            if self.stream:
                self.sender_task = asyncio.create_task(self.sender_stream(deepgram_ws))

                if self.is_flux_model:
                    self.heartbeat_task = asyncio.create_task(self.send_heartbeat_flux(deepgram_ws))
                    self.flux_watchdog_task = asyncio.create_task(self.monitor_flux_turn_timeout())
                else:
                    self.heartbeat_task = asyncio.create_task(self.send_heartbeat(deepgram_ws))
                    # Nova relies on UtteranceEnd, which can be unreliable — force-finalize on timeout.
                    self.utterance_timeout_task = asyncio.create_task(self.monitor_utterance_timeout())

                receiver_method = self.receiver_flux if self.is_flux_model else self.receiver
                logger.info(f"Using {'Flux' if self.is_flux_model else 'Nova'} receiver for model: {self.model}")

                try:
                    async for message in receiver_method(deepgram_ws):
                        if self.connection_on:
                            await self.push_to_transcriber_queue(message)
                        else:
                            await self._close(deepgram_ws, data={"type": "CloseStream"})
                            if not self.is_flux_model:
                                # Nova sends a Metadata message after CloseStream — drain it for billing duration
                                logger.info("closing the deepgram connection, waiting for Metadata")
                                async def drain_metadata():
                                    async for _ in self.receiver(deepgram_ws):
                                        if "deepgram_duration" in self.meta_info:
                                            return

                                try:
                                    # wait_for, not asyncio.timeout (3.11+, crashes on the 3.10 runtime).
                                    await asyncio.wait_for(drain_metadata(), timeout=5)
                                except asyncio.TimeoutError:
                                    logger.warning("Timeout waiting for Deepgram Metadata after CloseStream")
                            break
                except ConnectionClosedError as e:
                    logger.error(f"Deepgram websocket connection closed during streaming: {e}")
                    self.connection_error = str(e)
                except Exception as e:
                    logger.error(f"Error during streaming: {e}")
                    self.connection_error = str(e)
                    raise
            else:
                async for message in self.sender():
                    await self.push_to_transcriber_queue(message)

        except (ValueError, ConnectionError) as e:
            logger.error(f"Connection error in transcribe: {e}")
            self.connection_error = str(e)
            await self.toggle_connection()
        except Exception as e:
            logger.error(f"Unexpected error in transcribe: {e}")
            self.connection_error = str(e)
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

            if hasattr(self, "sender_task") and self.sender_task is not None:
                self.sender_task.cancel()
            if hasattr(self, "heartbeat_task") and self.heartbeat_task is not None:
                self.heartbeat_task.cancel()
            if hasattr(self, "utterance_timeout_task") and self.utterance_timeout_task is not None:
                self.utterance_timeout_task.cancel()
            if hasattr(self, "flux_watchdog_task") and self.flux_watchdog_task is not None:
                self.flux_watchdog_task.cancel()

            # Use Deepgram's actual audio duration for billing
            if self.meta_info is not None and "deepgram_duration" in self.meta_info:
                self.meta_info["transcriber_duration"] = self.meta_info["deepgram_duration"]

            meta = dict(getattr(self, "meta_info", None) or {})
            if self.connection_error:
                meta["connection_error"] = self.connection_error
            await self.push_to_transcriber_queue(create_ws_data_packet("transcriber_connection_closed", meta))
