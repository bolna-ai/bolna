import asyncio
import base64
import json
import os
import time
import traceback

import websockets
from dotenv import load_dotenv
from websockets.exceptions import ConnectionClosed, InvalidHandshake

from .base_transcriber import BaseTranscriber
from bolna.enums import TelephonyProvider
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.utils import create_ws_data_packet, resample, timestamp_ms, ulaw_to_pcm

logger = configure_logger(__name__)
load_dotenv()

GEMINI_LIVE_URL = (
    "wss://generativelanguage.googleapis.com/ws/"
    "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
)
# The Live API only accepts 16 kHz mono PCM-16, regardless of the call's telephony rate.
GEMINI_INPUT_SAMPLE_RATE = 16000
# Values of `language` that mean "let the model detect", which maps to an empty languageCodes.
AUTO_LANGUAGE_VALUES = {"", "auto", "multi", "multilingual"}


class GeminiTranscriber(BaseTranscriber):
    """Gemini 3.5 Transcribe streaming STT over the Gemini Live API (BidiGenerateContent).

    Distinct from GoogleTranscriber (Google Cloud Speech V2, service-account auth). This talks to
    the Gemini Live socket with an API key, resamples the call to 16 kHz PCM-16 that the API
    requires, and rotates the session on the Live API's ~10 minute cap since a transcriber, unlike
    the s2s provider, has no conversation context to resume.
    """

    def __init__(
        self,
        telephony_provider,
        input_queue=None,
        model="gemini-3.5-transcribe-live",
        stream=True,
        language=None,
        endpointing="500",
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
        self.language_hints = language_hints
        self.encoding = encoding
        self.sampling_rate = int(sampling_rate) if isinstance(sampling_rate, (str, int)) else 16000
        self.keywords = keywords
        self.transcriber_output_queue = output_queue
        self.connected_via_dashboard = kwargs.get("enforce_streaming", True)

        # GOOGLE_API_KEY is the same key GeminiLLM reads; accept either name.
        self.api_key = kwargs.get("transcriber_key") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

        # SMART strips disfluencies and self-corrections; VERBATIM keeps every word. Left unset the
        # server default (VERBATIM) applies, which is what a downstream LLM should reason over.
        self.transcription_mode = kwargs.get("transcription_mode")
        self.silence_duration_ms = int(endpointing) if endpointing is not None else None
        _prefix = kwargs.get("vad_prefix_padding_ms")
        self.vad_prefix_padding_ms = int(_prefix) if _prefix is not None else None
        # Gemini's VAD owns the endpoint, so the caller stopped roughly one silence window earlier.
        self.user_stop_offset_ms = self.silence_duration_ms if self.silence_duration_ms is not None else 500

        # Force-finalize a turn that got a final segment but never a turnComplete.
        _interim_timeout = kwargs.get("interim_timeout")
        self.interim_timeout = float(_interim_timeout) if _interim_timeout is not None else 8.0

        self._resolve_audio_params()

        self.websocket_connection = None
        self.connection_authenticated = False
        self.sender_task = None
        self.transcription_task = None
        self.utterance_timeout_task = None
        self.connection_error = None
        self.audio_submitted = False
        self._eos_received = False
        # Gemini Live reports no billed duration, so bill on the audio actually streamed.
        self.audio_duration_s = 0.0

        # Per-turn transcript state
        self.final_transcript = ""
        self.running_interim = ""
        self.current_turn_id = None
        self.current_turn_start_time = None
        self._turn_first_speech_epoch_ms = None
        self.current_turn_interim_details = []
        self.turn_counter = 0
        self.last_interim_time = None

    def _resolve_audio_params(self):
        """Set encoding and sample rate from the telephony/web I/O provider (task_manager also
        pre-sets these for sip-trunk and web/freeswitch, so this stays in agreement with it)."""
        if self.provider in TelephonyProvider.telephony_values():
            self.encoding = "mulaw" if self.provider in TelephonyProvider.mulaw_values() else "linear16"
            self.sampling_rate = 8000
        elif self.provider == "web_based_call":
            self.encoding = "linear16"
            self.sampling_rate = 16000
        elif not self.connected_via_dashboard:
            self.encoding = "linear16"
            self.sampling_rate = 16000
        if self.provider == "playground":
            self.sampling_rate = 8000

    def _resolve_language_codes(self):
        """Explicit hints win; a concrete language becomes a single code; anything else auto-detects."""
        if self.language_hints:
            return [h for h in self.language_hints if h]
        lang = (self.language or "").lower()
        if lang in AUTO_LANGUAGE_VALUES:
            return []
        return [self.language]

    def _build_setup(self):
        """The one client frame Gemini needs before audio. Keep it to keys the API is known to
        accept: an unrecognized key makes Gemini reject the whole setup and the call never connects.
        That is why keywords/custom-vocabulary biasing is not sent here yet."""
        transcription_config = {"languageCodes": self._resolve_language_codes()}
        if self.transcription_mode:
            transcription_config["mode"] = self.transcription_mode

        setup = {
            "model": self.model if self.model.startswith("models/") else f"models/{self.model}",
            "generationConfig": {"responseModalities": ["TEXT"]},
            "inputAudioTranscription": transcription_config,
        }

        automatic_vad = {
            key: value
            for key, value in (
                ("silenceDurationMs", self.silence_duration_ms),
                ("prefixPaddingMs", self.vad_prefix_padding_ms),
            )
            if value is not None
        }
        if automatic_vad:
            setup["realtimeInputConfig"] = {"automaticActivityDetection": automatic_vad}
        return setup

    def get_meta_info(self):
        return self.meta_info

    async def gemini_connect(self):
        """Open the socket, send setup, and block on setupComplete before any audio is allowed."""
        if not self.api_key:
            raise ConnectionError("No Gemini API key: set GEMINI_API_KEY or pass transcriber_key")
        url = f"{GEMINI_LIVE_URL}?key={self.api_key}"
        try:
            gemini_ws = await asyncio.wait_for(websockets.connect(url, max_size=None), timeout=10.0)
            await gemini_ws.send(json.dumps({"setup": self._build_setup()}))
            raw = await asyncio.wait_for(gemini_ws.recv(), timeout=10.0)
            message = json.loads(raw)
            if "setupComplete" not in message:
                error = message.get("error", {})
                raise ConnectionError(f"Gemini setup failed: {error.get('message', json.dumps(message)[:200])}")
            self.websocket_connection = gemini_ws
            self.connection_authenticated = True
            logger.info(f"Connected to Gemini Live (model={self.model}, source={self.encoding}@{self.sampling_rate})")
            return gemini_ws
        except asyncio.TimeoutError:
            raise ConnectionError("Timeout while connecting to Gemini Live websocket")
        except InvalidHandshake as e:
            raise ConnectionError(f"Invalid handshake during Gemini Live websocket connection: {e}")
        except ConnectionClosed as e:
            raise ConnectionError(f"Gemini Live websocket closed during setup: {e}")

    def _to_gemini_pcm(self, data):
        """Telephony audio to the 16 kHz PCM-16 the Live API requires: decode mulaw, then upsample."""
        pcm = ulaw_to_pcm(data) if self.encoding == "mulaw" else data
        if self.sampling_rate != GEMINI_INPUT_SAMPLE_RATE:
            pcm = resample(pcm, GEMINI_INPUT_SAMPLE_RATE, format="pcm", original_sample_rate=self.sampling_rate)
        return pcm

    async def sender_stream(self, ws):
        """Resample each frame and forward it as base64 PCM; signal audioStreamEnd on eos."""
        try:
            while True:
                ws_data_packet = await self.input_queue.get()

                if not self.audio_submitted:
                    self.meta_info = ws_data_packet.get("meta_info")
                    self.audio_submitted = True
                    self.current_request_id = self.generate_request_id()
                    self.meta_info["request_id"] = self.current_request_id

                if ws_data_packet.get("meta_info", {}).get("eos") is True:
                    self._eos_received = True
                    try:
                        await ws.send(json.dumps({"realtimeInput": {"audioStreamEnd": True}}))
                        # Gemini keeps the session open when the caller stops, so close it after a
                        # brief grace for the last final. Otherwise the receiver blocks until the
                        # idle deadline minutes later, stalling teardown and the duration never bills.
                        await asyncio.sleep(0.5)
                        await ws.close()
                    except Exception as e:
                        logger.error(f"Error finalizing Gemini stream on eos: {e}")
                    break

                data = ws_data_packet.get("data")
                if not data:
                    continue
                self.audio_duration_s += len(data) / ((1 if self.encoding == "mulaw" else 2) * self.sampling_rate)
                try:
                    pcm = self._to_gemini_pcm(data)
                    await ws.send(
                        json.dumps(
                            {
                                "realtimeInput": {
                                    "audio": {
                                        "data": base64.b64encode(pcm).decode("ascii"),
                                        "mimeType": f"audio/pcm;rate={GEMINI_INPUT_SAMPLE_RATE}",
                                    }
                                }
                            }
                        )
                    )
                except ConnectionClosed:
                    break
                except Exception as e:
                    logger.error(f"Error sending audio to Gemini Live: {e}")
                    break
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Error in Gemini sender_stream: {e}")
            raise

    def _start_turn(self):
        self.turn_counter += 1
        self.current_turn_id = self.turn_counter
        now = timestamp_ms()
        self.current_turn_start_time = now
        self._turn_first_speech_epoch_ms = now
        self.current_turn_interim_details = []
        self.final_transcript = ""
        self.running_interim = ""
        self.is_transcript_sent_for_processing = False
        self.turn_latencies.append(
            {
                "turn_id": self.current_turn_id,
                "asr_start_epoch_ms": self.current_turn_start_time,
                "asr_turn_start_epoch_ms": self._turn_first_speech_epoch_ms,
            }
        )

    def _reset_turn_state(self):
        self._turn_first_speech_epoch_ms = None
        self.current_turn_interim_details = []
        self.current_turn_start_time = None
        self.current_turn_id = None
        self.final_transcript = ""
        self.running_interim = ""
        self.last_interim_time = None
        self.is_transcript_sent_for_processing = True

    def _finalize_turn_latency(self, final_transcript):
        if self.current_turn_interim_details:
            for entry in self.current_turn_interim_details:
                entry["is_final"] = False
            self.current_turn_interim_details[-1]["is_final"] = True
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

    def _finalized_transcript_packet(self, final):
        """Build the end-of-turn transcript packet and reset for the next turn."""
        self._finalize_turn_latency(final)
        self.meta_info["user_stop_offset_ms"] = self.user_stop_offset_ms
        packet = create_ws_data_packet({"type": "transcript", "content": final}, self.meta_info)
        self._reset_turn_state()
        return packet

    async def receiver(self, ws):
        """Map Gemini serverContent onto speech_started / interim / transcript / speech_ended.

        The transcribe-live model streams interimInputTranscription (a cumulative partial) and then
        inputTranscription (the whole finalized turn once the speaker pauses), followed by
        generationComplete. The final is emitted on inputTranscription; generationComplete is the
        turn-end backstop for an utterance that produced only partials.
        """
        async for raw in ws:
            try:
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                message = json.loads(raw)

                if "goAway" in message:
                    logger.info(f"Gemini goAway; session will rotate: {message['goAway']}")
                    continue

                if "error" in message:
                    self.connection_error = json.dumps(message["error"])[:300]
                    logger.error(f"Gemini Live error: {self.connection_error}")
                    break

                server_content = message.get("serverContent")
                if not server_content:
                    continue

                interim = (server_content.get("interimInputTranscription") or {}).get("text")
                if interim:
                    if self.current_turn_id is None:
                        self._start_turn()
                        yield create_ws_data_packet("speech_started", self.meta_info)
                    self.running_interim = interim
                    self.last_interim_time = time.time()
                    self.current_turn_interim_details.append(
                        {"transcript": interim, "is_final": False, "received_at": time.time()}
                    )
                    yield create_ws_data_packet(
                        {"type": "interim_transcript_received", "content": interim}, self.meta_info
                    )

                final = (server_content.get("inputTranscription") or {}).get("text")
                if final:
                    if self.current_turn_id is None:
                        self._start_turn()
                        yield create_ws_data_packet("speech_started", self.meta_info)
                    self.final_transcript += final
                    text = self.final_transcript.strip()
                    if text and not self.is_transcript_sent_for_processing:
                        logger.info(f"Gemini final transcript: {text}")
                        yield self._finalized_transcript_packet(text)

                if server_content.get("turnComplete") or server_content.get("generationComplete"):
                    text = self.final_transcript.strip() or self.running_interim.strip()
                    if text and not self.is_transcript_sent_for_processing:
                        yield self._finalized_transcript_packet(text)
                    elif self.current_turn_id is not None:
                        yield create_ws_data_packet({"type": "speech_ended"}, self.meta_info)
                        self._reset_turn_state()

            except Exception as e:
                logger.error(f"Error processing Gemini message: {e}")
                traceback.print_exc()

    def _flush_pending_final(self):
        """On a session rotation mid-turn, don't lose a partial: publish it as the turn's transcript."""
        text = self.final_transcript.strip() or self.running_interim.strip()
        if text and self.current_turn_id is not None and not self.is_transcript_sent_for_processing:
            return self._finalized_transcript_packet(text)
        return None

    async def monitor_utterance_timeout(self):
        """Force-finalize a turn that got content but no turnComplete within interim_timeout."""
        try:
            while True:
                await asyncio.sleep(1.0)
                if (
                    self.last_interim_time
                    and not self.is_transcript_sent_for_processing
                    and (self.final_transcript.strip() or self.running_interim.strip())
                    and time.time() - self.last_interim_time > self.interim_timeout
                ):
                    text = self.final_transcript.strip() or self.running_interim.strip()
                    logger.warning(f"Gemini utterance timeout, force-finalizing turn {self.current_turn_id}")
                    self._finalize_turn_latency(text)
                    self.meta_info["user_stop_offset_ms"] = self.user_stop_offset_ms
                    data = {"type": "transcript", "content": text, "force_finalized": True}
                    await self.push_to_transcriber_queue(create_ws_data_packet(data, self.meta_info))
                    self._reset_turn_state()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Gemini utterance timeout monitor error: {e}")
            raise

    async def push_to_transcriber_queue(self, data_packet):
        await self.transcriber_output_queue.put(data_packet)

    async def toggle_connection(self):
        self.connection_on = False
        for task in (self.sender_task, self.utterance_timeout_task):
            if task is not None:
                task.cancel()
        self.utterance_timeout_task = None
        if self.websocket_connection is not None:
            try:
                await self.websocket_connection.close()
            except Exception as e:
                logger.error(f"Error closing Gemini websocket: {e}")
            finally:
                self.websocket_connection = None
                self.connection_authenticated = False

    async def cleanup(self):
        logger.info("Cleaning up Gemini transcriber resources")
        self.connection_on = False
        for name, task in (
            ("sender_task", self.sender_task),
            ("transcription_task", self.transcription_task),
            ("utterance_timeout_task", self.utterance_timeout_task),
        ):
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    logger.info(f"Gemini {name} cancelled")
                except Exception as e:
                    logger.warning(f"Error cancelling Gemini {name}: {e}")
        if self.websocket_connection is not None:
            try:
                await self.websocket_connection.close()
            except Exception as e:
                logger.error(f"Error closing Gemini websocket: {e}")
            finally:
                self.websocket_connection = None
                self.connection_authenticated = False

    async def run(self):
        try:
            self.transcription_task = asyncio.create_task(self.transcribe())
        except Exception as e:
            logger.error(f"Error starting Gemini transcriber: {e}")

    async def transcribe(self):
        """Stream until eos or shutdown, reopening a fresh session across the Live API's ~10 min cap."""
        start_time = timestamp_ms()
        self.utterance_timeout_task = asyncio.create_task(self.monitor_utterance_timeout())
        try:
            while self.connection_on and not self._eos_received:
                try:
                    gemini_ws = await self.gemini_connect()
                except (ValueError, ConnectionError) as e:
                    logger.error(f"Failed to establish Gemini connection: {e}")
                    self.connection_error = str(e)
                    break

                if self.connection_time is None:
                    self.connection_time = round(timestamp_ms() - start_time)

                self.sender_task = asyncio.create_task(self.sender_stream(gemini_ws))
                reconnect = False
                try:
                    async for message in self.receiver(gemini_ws):
                        if not self.connection_on:
                            break
                        await self.push_to_transcriber_queue(message)
                    # The generator returning means the server closed the stream (cap or finished).
                    reconnect = self.connection_on and not self._eos_received and not self.connection_error
                except ConnectionClosed as e:
                    logger.warning(f"Gemini websocket closed during streaming: {e}")
                    reconnect = self.connection_on and not self._eos_received
                except Exception as e:
                    logger.error(f"Error during Gemini streaming: {e}")
                    self.connection_error = str(e)
                finally:
                    if self.sender_task is not None:
                        self.sender_task.cancel()
                    try:
                        await gemini_ws.close()
                    except Exception:
                        pass
                    self.websocket_connection = None

                if reconnect:
                    flushed = self._flush_pending_final()
                    if flushed is not None:
                        await self.push_to_transcriber_queue(flushed)
                    self.audio_submitted = False
                    logger.info("Rotating Gemini Live session (cap reached or transient drop)")
                    continue
                break
        except Exception as e:
            logger.error(f"Unexpected error in Gemini transcribe: {e}")
            self.connection_error = str(e)
        finally:
            if self.utterance_timeout_task is not None:
                self.utterance_timeout_task.cancel()
            if self.sender_task is not None:
                self.sender_task.cancel()
            meta = dict(getattr(self, "meta_info", None) or {})
            meta["transcriber_duration"] = round(self.audio_duration_s, 4)
            if self.connection_error:
                meta["connection_error"] = self.connection_error
            await self.push_to_transcriber_queue(create_ws_data_packet("transcriber_connection_closed", meta))
