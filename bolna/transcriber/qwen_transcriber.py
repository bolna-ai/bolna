"""Qwen3-ASR speech-to-text, in two flavours behind one provider name.

"Qwen3-ASR" is two different products, and they need two different clients:

  stream=true  (default) — Qwen3-ASR-Flash-Realtime on Alibaba Model Studio. A hosted
      WebSocket carrying the OpenAI-Realtime event shape: a `session.update` handshake,
      base64 PCM in `input_audio_buffer.append`, and transcription events back. Endpointing
      is Qwen's own server VAD, which is the reason to pick it for a voice agent.

  stream=false — the Apache-2.0 open weights (Qwen3-ASR-1.7B / 0.6B), reached through any
      host's OpenAI-compatible /v1/audio/transcriptions: OpenRouter, DeepInfra, Azure AI
      Foundry, or a self-hosted `vllm serve Qwen/Qwen3-ASR-1.7B`. That endpoint is batch
      file transcription, so this path buffers an utterance, endpoints it locally on frame
      RMS, and POSTs a WAV. It emits NO interim results — see the caveat below.

Realtime turn lifecycle:
    input_audio_buffer.speech_started      -> "speech_started"
    ...transcription.text (text + stash)   -> "interim_transcript_received"
    input_audio_buffer.speech_stopped      -> arm the completion watchdog
    ...transcription.completed             -> "transcript"  (or "speech_ended" if empty)

Batch turn lifecycle:
    frame RMS crosses threshold            -> "speech_started"
    endpointing_ms of quiet (or the cap)   -> POST the buffer -> "transcript"

CAVEAT on the batch path: task_manager drives barge-in off interim_transcript_received, so
with no interims the agent cannot be interrupted mid-utterance — it only learns what the
caller said once they stop. Add a round trip (~1.8s observed on DeepInfra) on top. Good for
transcription, materially worse for conversational turn latency; prefer stream=true for live
calls. vLLM's /v1/realtime would in principle fix this, but it transcribes each 5s segment in
isolation and repeats speech across boundaries (vllm#35767, closed "not planned").

Neither path accepts mulaw, so Twilio and sip-trunk audio is decoded to linear16 first. Sample
rate is passed through (8k and 16k are both native to Qwen), so nothing resamples on the hot path.
"""

import asyncio
import audioop
import base64
import json
import os
import time
import traceback
from urllib.parse import urlencode

import aiohttp
import websockets
from dotenv import load_dotenv
from websockets.asyncio.client import ClientConnection
from websockets.exceptions import ConnectionClosed, ConnectionClosedError, InvalidHandshake

from .base_transcriber import BaseTranscriber
from bolna.constants import (
    QWEN_ASR_AUTO_LANGUAGE_VALUES,
    QWEN_ASR_COMPLETION_TIMEOUT_S,
    QWEN_ASR_DEFAULT_BASE_URL,
    QWEN_ASR_DEFAULT_BATCH_MODEL,
    QWEN_ASR_DEFAULT_MODEL,
    QWEN_ASR_HOST,
    QWEN_ASR_HTTP_TIMEOUT_S,
    QWEN_ASR_MAX_CORPUS_CHARS,
    QWEN_ASR_MAX_UTTERANCE_S,
    QWEN_ASR_MIN_SILENCE_DURATION_MS,
    QWEN_ASR_REALTIME_PATH,
    QWEN_ASR_SPEECH_RMS_THRESHOLD,
    QWEN_ASR_SUPPORTED_LANGUAGES,
    WEB_BASED_CALL_PROVIDER,
)
from bolna.enums import TelephonyProvider
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.ssl_context import get_ssl_context
from bolna.helpers.utils import create_ws_data_packet, pcm_to_wav_bytes, timestamp_ms

logger = configure_logger(__name__)
load_dotenv()

# Heartbeat cadence. Qwen defines no KeepAlive frame, so a bare WebSocket ping is the keepalive.
_HEARTBEAT_INTERVAL_S = 5
# How long the EOS drain waits for the last `.completed` before closing the socket.
_EOS_DRAIN_TIMEOUT_S = 5.0


class QwenTranscriber(BaseTranscriber):
    def __init__(
        self,
        telephony_provider,
        input_queue=None,
        model=QWEN_ASR_DEFAULT_MODEL,
        stream=True,
        language="en",
        endpointing="400",
        sampling_rate="16000",
        encoding="linear16",
        output_queue=None,
        keywords=None,
        process_interim_results="true",
        vad_threshold=None,
        **kwargs,
    ):
        super().__init__(input_queue)
        self.provider = telephony_provider
        self.model = model or QWEN_ASR_DEFAULT_MODEL
        self.stream = stream
        self.language = language
        self.encoding = encoding
        self.sampling_rate = int(sampling_rate) if isinstance(sampling_rate, (str, int)) else 16000
        self.keywords = keywords
        self.process_interim_results = process_interim_results
        self.transcriber_output_queue = output_queue
        self.connected_via_dashboard = kwargs.get("enforce_streaming", True)
        self.run_id = kwargs.get("run_id")

        # Bolna's `endpointing` is the silence the agent tolerates before replying; that is exactly
        # what Qwen's server VAD measures, so it maps straight onto silence_duration_ms.
        self.endpointing_ms = int(endpointing) if endpointing is not None else 400
        self.silence_duration_ms = max(QWEN_ASR_MIN_SILENCE_DURATION_MS, self.endpointing_ms)
        # threshold=0.0 is meaningful to Qwen (accept everything), so `is not None` — not truthiness.
        self.vad_threshold = float(vad_threshold) if vad_threshold is not None else None

        self.api_key = kwargs.get("transcriber_key") or os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
        self.qwen_host = kwargs.get("transcriber_host") or os.getenv("QWEN_ASR_HOST", QWEN_ASR_HOST)
        self.workspace_id = kwargs.get("transcriber_workspace") or os.getenv("DASHSCOPE_WORKSPACE_ID")

        # ── open-weights batch path (stream=false) ────────────────────────────────
        self.base_url = (kwargs.get("base_url") or os.getenv("QWEN_ASR_BASE_URL") or QWEN_ASR_DEFAULT_BASE_URL).rstrip(
            "/"
        )
        # A realtime model id posted to /v1/audio/transcriptions is a 404 nobody enjoys
        # debugging, so a config that only flipped `stream` still lands on a servable model.
        if not self.stream and self.model == QWEN_ASR_DEFAULT_MODEL:
            self.model = kwargs.get("batch_model") or os.getenv("QWEN_ASR_BATCH_MODEL") or QWEN_ASR_DEFAULT_BATCH_MODEL
            logger.info(f"Qwen ASR: stream=false with a realtime model id — using {self.model!r} instead")
        _rms = kwargs.get("speech_rms_threshold")
        self.speech_rms_threshold = int(_rms) if _rms is not None else QWEN_ASR_SPEECH_RMS_THRESHOLD
        self.max_utterance_s = float(kwargs.get("max_utterance_s") or QWEN_ASR_MAX_UTTERANCE_S)
        self.http_timeout_s = float(kwargs.get("http_timeout_s") or QWEN_ASR_HTTP_TIMEOUT_S)
        self._http_session = None
        self._utterance_buffer = bytearray()
        # Seconds of audio actually transcribed this connection. task_manager accumulates this
        # off the transcriber_connection_closed packet for ASR usage accounting, so leaving it
        # unset bills every qwen call as zero.
        self.total_audio_seconds = 0.0
        self._speech_active = False
        self._silence_started_at = None
        self._utterance_started_at = None

        # Connection + task state
        self.websocket_connection = None
        self.connection_authenticated = False
        self.connection_error = None
        self.transcription_task = None
        self.sender_task = None
        self.heartbeat_task = None
        self.completion_timeout_task = None

        # Per-stream audio bookkeeping (reset on each connect in transcribe())
        self.audio_submitted = False
        self.audio_submission_time = None
        self.num_frames = 0
        self.connection_start_time = None
        self.audio_frame_duration = 0.5
        self.meta_info = {}

        # Per-turn transcript state
        self.turn_counter = 0
        self.current_turn_id = None
        self.current_turn_start_time = None
        self._turn_first_speech_epoch_ms = None
        self.current_turn_interim_details = []
        self.last_interim_transcript = ""
        self.last_interim_time = None
        self._first_result_received = False
        # Set at speech_stopped; the watchdog force-finalizes if `.completed` never follows.
        self._speech_stopped_at = None
        # Wall-clock of speech_stopped, reported downstream as when the caller actually stopped.
        self._speech_stopped_wall = None
        self._last_detected_language = None
        self._last_detected_emotion = None
        # Lets the EOS drain in sender_stream wait for the trailing `.completed`.
        self._final_transcript_event = asyncio.Event()
        self.completion_timeout = float(kwargs.get("completion_timeout") or QWEN_ASR_COMPLETION_TIMEOUT_S)

        self._configure_audio_params()

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def _configure_audio_params(self):
        """Resolve encoding / sample rate / frame duration from the call's I/O provider.

        `self.encoding` describes what arrives on input_queue, not what Qwen receives — the
        mulaw case is decoded in _to_pcm16(). TranscriberPool also reads this attribute to pick
        the right silence byte for standby keepalives, so it must stay truthful to the input.
        """
        if self.provider in TelephonyProvider.telephony_values():
            self.encoding = "mulaw" if self.provider in TelephonyProvider.mulaw_values() else "linear16"
            self.sampling_rate = 8000
            self.audio_frame_duration = 0.2
        elif self.provider == WEB_BASED_CALL_PROVIDER:
            self.encoding = "linear16"
            self.sampling_rate = 16000
            self.audio_frame_duration = 0.256
        elif not self.connected_via_dashboard:
            self.encoding = "linear16"
            self.sampling_rate = 16000
            self.audio_frame_duration = 0.2
        else:
            self.audio_frame_duration = 0.2

        if self.provider == "playground":
            self.sampling_rate = 8000
            self.audio_frame_duration = 0.0

    def _resolve_language(self):
        """Concrete supported code -> pin it; auto/multi/unknown -> None (Qwen auto-detects)."""
        lang = (self.language or "").lower()
        if lang in QWEN_ASR_AUTO_LANGUAGE_VALUES:
            return None
        # Deepgram-style "multi-hi" means auto-detect biased to Hindi; Qwen has no such form,
        # so take the concrete half rather than pinning the literal "multi-hi".
        if lang.startswith("multi-"):
            lang = lang.split("-", 1)[1]
        # "en-US" / "zh-CN" -> the base code Qwen names.
        base = lang.split("-")[0]
        if base in QWEN_ASR_SUPPORTED_LANGUAGES:
            return base
        logger.warning(f"Qwen ASR: language {self.language!r} not in supported set — falling back to auto-detect")
        return None

    def _build_corpus(self):
        """Bolna keywords -> Qwen's biasing corpus, or None when there is nothing to bias."""
        if not self.keywords:
            return None
        terms = [kw.strip() for kw in self.keywords.split(",") if kw.strip()]
        if not terms:
            return None
        text = ", ".join(terms)
        if len(text) > QWEN_ASR_MAX_CORPUS_CHARS:
            # Truncate on a term boundary: a half-word left in the corpus biases toward a
            # string the caller will never say.
            text = text[:QWEN_ASR_MAX_CORPUS_CHARS].rsplit(",", 1)[0]
            logger.warning(f"Qwen ASR: biasing corpus truncated to {len(text)} chars")
        return {"text": text}

    def _build_session_config(self):
        """The `session.update` payload sent as the first frame after connecting."""
        transcription = {}
        language = self._resolve_language()
        if language:
            transcription["language"] = language
        corpus = self._build_corpus()
        if corpus:
            transcription["corpus"] = corpus

        turn_detection = {
            "type": "server_vad",
            "silence_duration_ms": self.silence_duration_ms,
        }
        if self.vad_threshold is not None:
            turn_detection["threshold"] = self.vad_threshold

        session = {
            "input_audio_format": "pcm",
            "sample_rate": int(self.sampling_rate),
            "turn_detection": turn_detection,
        }
        if transcription:
            session["input_audio_transcription"] = transcription
        return {"type": "session.update", "session": session}

    def get_qwen_ws_url(self):
        protocol = os.getenv("QWEN_ASR_HOST_PROTOCOL", "wss")
        params = {"model": self.model}
        return f"{protocol}://{self.qwen_host}{QWEN_ASR_REALTIME_PATH}?{urlencode(params)}"

    # ------------------------------------------------------------------
    # Audio
    # ------------------------------------------------------------------

    def _to_pcm16(self, audio_bytes):
        """Decode mulaw to linear16; pass PCM through untouched.

        Sample rate is never changed — Qwen takes 8k natively, so telephony audio needs no
        resampling and the frame stays byte-identical for the linear16 providers.
        """
        if self.encoding != "mulaw":
            return audio_bytes
        try:
            return audioop.ulaw2lin(audio_bytes, 2)
        except Exception as e:
            logger.error(f"Qwen ASR: mulaw decode failed, dropping frame: {e}")
            return None

    # ------------------------------------------------------------------
    # Turn bookkeeping
    # ------------------------------------------------------------------

    def _start_turn(self):
        self.turn_counter += 1
        self.current_turn_id = self.turn_counter
        now = timestamp_ms()
        self.current_turn_start_time = now
        self._turn_first_speech_epoch_ms = now
        self.current_turn_interim_details = []
        self.last_interim_transcript = ""
        self.last_interim_time = None
        self._first_result_received = False
        self._speech_stopped_at = None
        self._speech_stopped_wall = None
        self.is_transcript_sent_for_processing = False
        self._final_transcript_event.clear()
        self.turn_latencies.append(
            {
                "turn_id": self.current_turn_id,
                "asr_start_epoch_ms": self.current_turn_start_time,
                "asr_turn_start_epoch_ms": self._turn_first_speech_epoch_ms,
            }
        )
        logger.info(f"Qwen ASR: starting turn {self.current_turn_id}")

    def _reset_turn_state(self):
        self.current_turn_id = None
        self.current_turn_start_time = None
        self._turn_first_speech_epoch_ms = None
        self.current_turn_interim_details = []
        self.last_interim_transcript = ""
        self.last_interim_time = None
        self._speech_stopped_at = None
        self._speech_stopped_wall = None
        self.is_transcript_sent_for_processing = True

    def _mark_last_interim_final(self):
        if not self.current_turn_interim_details:
            return
        for entry in self.current_turn_interim_details:
            entry["is_final"] = False
        self.current_turn_interim_details[-1]["is_final"] = True

    def _build_finalized_turn_latency(self, final_transcript, force_finalized=False):
        self._mark_last_interim_final()
        first_interim_to_final_ms, last_interim_to_final_ms = self.calculate_interim_to_final_latencies(
            self.current_turn_interim_details
        )
        entry = {
            "turn_id": self.current_turn_id,
            "asr_start_epoch_ms": self.current_turn_start_time,
            "asr_turn_start_epoch_ms": self._turn_first_speech_epoch_ms,
            "asr_finalized_epoch_ms": timestamp_ms(),
            "final_transcript": final_transcript,
            "interim_details": self.current_turn_interim_details,
            "first_interim_to_final_ms": first_interim_to_final_ms,
            "last_interim_to_final_ms": last_interim_to_final_ms,
        }
        if self._speech_stopped_wall is not None:
            entry["user_speech_end_epoch_ms"] = self._speech_stopped_wall * 1000
        if force_finalized:
            entry["force_finalized"] = True
        self._upsert_turn_latency(entry)

    def _stamp_turn_meta(self):
        """Attach the per-turn signals the task manager reads off a final transcript."""
        # The caller stopped talking one silence-window before the turn closed, and the
        # interruption manager needs that offset to place the boundary. Which window depends
        # on who did the endpointing: Qwen's server VAD, or our local RMS gate.
        self.meta_info["user_stop_offset_ms"] = self.silence_duration_ms if self.stream else self.endpointing_ms
        if self._speech_stopped_wall is not None:
            self.meta_info["user_stop_ts_wall"] = self._speech_stopped_wall
            self.meta_info["last_vocal_frame_timestamp"] = self._speech_stopped_wall
        if self._last_detected_language:
            self.meta_info["transcriber_detected_language"] = self._last_detected_language
            # Only retarget TTS when no language was pinned — otherwise a single mis-tagged
            # turn would move the agent's voice off the language the agent was configured for.
            if self._resolve_language() is None:
                self.meta_info["detected_language_code"] = self._last_detected_language
        if self._last_detected_emotion:
            # Qwen returns one of seven emotions per turn; passed through as telemetry only.
            self.meta_info["transcriber_detected_emotion"] = self._last_detected_emotion

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    async def qwen_connect(self) -> ClientConnection:
        if not self.api_key:
            raise ValueError("Qwen ASR: no API key (set QWEN_API_KEY / DASHSCOPE_API_KEY or pass transcriber_key)")

        url = self.get_qwen_ws_url()
        headers = {"Authorization": f"Bearer {self.api_key}"}
        if self.workspace_id:
            headers["X-DashScope-WorkSpace"] = self.workspace_id

        try:
            logger.info(f"Connecting to Qwen ASR realtime: {url}")
            ws = await asyncio.wait_for(
                websockets.connect(url, additional_headers=headers, ssl=get_ssl_context(url)),
                timeout=10.0,
            )
            config = self._build_session_config()
            await ws.send(json.dumps(config))
            self.websocket_connection = ws
            self.connection_authenticated = True
            logger.info(
                f"Connected to Qwen ASR (model={self.model}, sample_rate={self.sampling_rate}, "
                f"input_encoding={self.encoding}, silence_duration_ms={self.silence_duration_ms}, "
                f"language={self._resolve_language() or 'auto'})"
            )
            return ws
        except asyncio.TimeoutError:
            raise ConnectionError("Timeout while connecting to Qwen ASR websocket")
        except InvalidHandshake as e:
            err = str(e)
            if "401" in err or "403" in err:
                raise ConnectionError(f"Qwen ASR auth failed: {e}")
            raise ConnectionError(f"Qwen ASR handshake failed: {e}")
        except ConnectionClosedError as e:
            raise ConnectionError(f"Qwen ASR websocket closed unexpectedly: {e}")
        except Exception as e:
            raise ConnectionError(f"Unexpected error connecting to Qwen ASR websocket: {e}")

    async def send_heartbeat(self, ws: ClientConnection):
        """Qwen defines no KeepAlive frame; a WebSocket ping is the keepalive."""
        try:
            while True:
                await asyncio.sleep(_HEARTBEAT_INTERVAL_S)
                try:
                    pong_waiter = await ws.ping()
                    await asyncio.wait_for(pong_waiter, timeout=10)
                except asyncio.TimeoutError:
                    logger.warning("Qwen ASR heartbeat ping timeout — connection may be stale")
                    break
                except ConnectionClosed:
                    break
                except Exception as e:
                    logger.error(f"Error sending Qwen ASR heartbeat: {e}")
                    break
        except asyncio.CancelledError:
            logger.info("Qwen ASR heartbeat task cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in Qwen ASR send_heartbeat: {e}")

    # ------------------------------------------------------------------
    # Sender
    # ------------------------------------------------------------------

    async def sender_stream(self, ws: ClientConnection):
        try:
            while True:
                ws_data_packet = await self.input_queue.get()

                if not self.audio_submitted:
                    packet_meta = ws_data_packet.get("meta_info")
                    # Standby keepalive frames carry an empty meta_info; adopting it would
                    # publish transcripts under a meta with no request_id.
                    if packet_meta:
                        self.meta_info = packet_meta
                        self.audio_submitted = True
                        self.audio_submission_time = time.time()
                        self.current_request_id = self.generate_request_id()
                        self.meta_info["request_id"] = self.current_request_id

                if (ws_data_packet.get("meta_info") or {}).get("eos") is True:
                    await self._drain_and_finish(ws)
                    break

                raw_audio = ws_data_packet.get("data")
                if not raw_audio:
                    continue

                pcm = self._to_pcm16(raw_audio)
                if not pcm:
                    continue

                self.num_frames += 1
                self.total_audio_seconds += len(pcm) / 2 / self.sampling_rate
                try:
                    await ws.send(
                        json.dumps(
                            {
                                "type": "input_audio_buffer.append",
                                "audio": base64.b64encode(pcm).decode(),
                            }
                        )
                    )
                except ConnectionClosed as e:
                    logger.error(f"Qwen ASR connection closed while sending audio: {e}")
                    break
                except Exception as e:
                    logger.error(f"Error sending audio to Qwen ASR: {e}")
                    break
        except asyncio.CancelledError:
            logger.info("Qwen ASR sender stream task cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in Qwen ASR sender_stream: {e}")
            raise

    async def _drain_and_finish(self, ws: ClientConnection):
        """On EOS: ask Qwen to finish, give the trailing final a moment to land, then close."""
        try:
            # Commit first so a turn still open at hangup is transcribed rather than discarded;
            # with server VAD the buffer is otherwise only flushed on the next silence.
            if self.current_turn_id is not None:
                await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
            await ws.send(json.dumps({"type": "session.finish"}))
        except Exception as e:
            logger.error(f"Error sending Qwen ASR session.finish: {e}")
            return

        if self.current_turn_id is not None:
            try:
                await asyncio.wait_for(self._final_transcript_event.wait(), timeout=_EOS_DRAIN_TIMEOUT_S)
            except asyncio.TimeoutError:
                logger.warning("Qwen ASR: timeout waiting for final transcript after EOS")

    # ------------------------------------------------------------------
    # Receiver
    # ------------------------------------------------------------------

    async def receiver(self, ws: ClientConnection):
        try:
            async for message in ws:
                try:
                    data = json.loads(message) if isinstance(message, str) else {}
                except Exception:
                    logger.error(f"Qwen ASR: non-JSON frame received: {str(message)[:200]}")
                    continue

                try:
                    event_type = data.get("type", "")

                    if self.connection_start_time is None:
                        self.connection_start_time = time.time() - (self.num_frames * self.audio_frame_duration)

                    if event_type == "input_audio_buffer.speech_started":
                        # A turn already open means Qwen re-opened without closing the last one;
                        # release it so the pending transcript isn't stranded.
                        if self.current_turn_id is not None:
                            async for packet in self._close_open_turn():
                                yield packet
                        self._start_turn()
                        yield create_ws_data_packet("speech_started", self.meta_info)

                    elif event_type == "conversation.item.input_audio_transcription.text":
                        if self.current_turn_id is None and self._interim_has_text(data):
                            # Interim with no preceding speech_started — open the turn here so
                            # downstream interruption handling still sees a turn boundary.
                            self._start_turn()
                            yield create_ws_data_packet("speech_started", self.meta_info)
                        packet = self._handle_interim(data)
                        if packet is not None:
                            yield packet

                    elif event_type == "input_audio_buffer.speech_stopped":
                        if self.current_turn_id is None:
                            logger.info("Qwen ASR: speech_stopped with no open turn — ignoring")
                        else:
                            self._speech_stopped_at = time.time()
                            self._speech_stopped_wall = self._speech_stopped_at - (self.silence_duration_ms / 1000.0)
                            logger.info(f"Qwen ASR: speech stopped on turn {self.current_turn_id}")

                    elif event_type == "conversation.item.input_audio_transcription.completed":
                        for packet in self._handle_completed(data):
                            yield packet

                    elif event_type == "conversation.item.input_audio_transcription.failed":
                        err = data.get("error", {}) or {}
                        logger.error(f"Qwen ASR transcription failed for item {data.get('item_id')}: {err}")
                        self._final_transcript_event.set()
                        # Close the turn so callee_speaking clears; a failed item has no text.
                        async for packet in self._close_open_turn():
                            yield packet

                    elif event_type == "error":
                        err = data.get("error", {}) or {}
                        self.connection_error = f"{err.get('code')}: {err.get('message')}"
                        logger.error(f"Qwen ASR error event: {err}")
                        # Session-level errors are terminal — the socket will not recover.
                        if err.get("type") == "invalid_request_error":
                            break

                    elif event_type == "session.finished":
                        logger.info(
                            f"Qwen ASR session finished — num_frames={self.num_frames} turns={self.turn_counter}"
                        )
                        break

                    elif event_type in (
                        "session.created",
                        "session.updated",
                        "input_audio_buffer.committed",
                        "conversation.item.created",
                    ):
                        logger.info(f"Qwen ASR session event: {event_type} | {json.dumps(data)[:400]}")

                    else:
                        logger.info(f"Qwen ASR unhandled event: {event_type} | {json.dumps(data)[:300]}")

                except Exception:
                    traceback.print_exc()

        except ConnectionClosedError as e:
            logger.error(f"Qwen ASR websocket closed during receiver: {e}")
            self.connection_error = str(e)
        except Exception:
            traceback.print_exc()

    @staticmethod
    def _running_utterance(data):
        """Qwen splits a partial into `text` (settled prefix) and `stash` (pre-recognized tail);
        the running utterance is their concatenation."""
        return ((data.get("text") or "") + (data.get("stash") or "")).strip()

    def _interim_has_text(self, data):
        return bool(self._running_utterance(data))

    def _handle_interim(self, data):
        """Build an interim packet from a `.text` event, or None if it carries no new text."""
        running = self._running_utterance(data)
        if not running:
            return None

        if data.get("language"):
            self._last_detected_language = data["language"]
        if data.get("emotion"):
            self._last_detected_emotion = data["emotion"]

        received_at = time.time()
        if not self._first_result_received and self.audio_submission_time:
            latency = received_at - self.audio_submission_time
            self.meta_info["transcriber_first_result_latency"] = latency
            self.meta_info["transcriber_latency"] = latency
            self._first_result_received = True

        # Qwen re-sends an unchanged partial while the stash settles; task_manager already
        # dedupes, but dropping it here keeps interim_details from inflating turn latencies.
        if running == self.last_interim_transcript:
            return None
        self.last_interim_transcript = running
        self.last_interim_time = received_at
        self.current_turn_interim_details.append(
            {
                "transcript": running,
                "received_at": received_at,
                "is_final": False,
            }
        )
        return create_ws_data_packet({"type": "interim_transcript_received", "content": running}, self.meta_info)

    def _handle_completed(self, data):
        """Turn a `.completed` event into the final transcript packet (or a bare turn close)."""
        transcript = (data.get("transcript") or "").strip()
        if data.get("language"):
            self._last_detected_language = data["language"]
        if data.get("emotion"):
            self._last_detected_emotion = data["emotion"]

        # Always unblock the EOS drain, even for an empty result.
        self._final_transcript_event.set()

        if self.is_transcript_sent_for_processing and self.current_turn_id is None:
            # Late duplicate after the watchdog already force-finalized this turn. Re-delivering
            # it would replay the user turn to the LLM.
            logger.info(f"Qwen ASR: dropping late completed event for already-closed turn: {transcript[:80]!r}")
            return

        if not transcript:
            logger.info("Qwen ASR: completed event with empty transcript — closing turn")
            yield create_ws_data_packet({"type": "speech_ended"}, self.meta_info)
            self._reset_turn_state()
            return

        item_id = data.get("item_id")
        if item_id:
            self.previous_request_id = self.current_request_id
            self.current_request_id = item_id
            self.meta_info["request_id"] = item_id
            self.meta_info["previous_request_id"] = self.previous_request_id

        self._build_finalized_turn_latency(transcript)
        self._stamp_turn_meta()
        logger.info(f"Qwen ASR: final transcript for turn {self.current_turn_id}: {transcript[:120]}")
        yield create_ws_data_packet({"type": "transcript", "content": transcript}, self.meta_info)
        self._reset_turn_state()

    async def _close_open_turn(self):
        """Release an open turn that will get no `.completed`, delivering any buffered text."""
        if self.current_turn_id is None:
            return
        transcript = self.last_interim_transcript.strip()
        if transcript:
            self._build_finalized_turn_latency(transcript, force_finalized=True)
            self._stamp_turn_meta()
            logger.warning(f"Qwen ASR: force-finalizing turn {self.current_turn_id}: {transcript[:120]}")
            yield create_ws_data_packet(
                {"type": "transcript", "content": transcript, "force_finalized": True}, self.meta_info
            )
        else:
            yield create_ws_data_packet({"type": "speech_ended"}, self.meta_info)
        self._reset_turn_state()

    # ------------------------------------------------------------------
    # Watchdog
    # ------------------------------------------------------------------

    def _completion_is_overdue(self, now):
        """True if speech_stopped fired on an open turn but `.completed` never arrived in time.

        The open-turn check is not redundant: _close_open_turn is a no-op without one, so a
        stopped-but-turnless state would re-fire this every tick for the rest of the call.
        """
        if self.current_turn_id is None or self._speech_stopped_at is None:
            return False
        if self.is_transcript_sent_for_processing:
            return False
        return (now - self._speech_stopped_at) > self.completion_timeout

    async def monitor_completion_timeout(self):
        """Force-close a turn Qwen stopped but never finalized.

        Without this a dropped `.completed` holds the agent's reply for the rest of the call:
        the task manager keeps waiting on a transcript that is never coming.
        """
        try:
            while True:
                await asyncio.sleep(0.2)
                if self._completion_is_overdue(time.time()):
                    logger.warning(
                        f"Qwen ASR: no completed event {self.completion_timeout}s after speech_stopped "
                        f"(turn {self.current_turn_id}) — force-closing turn"
                    )
                    self._final_transcript_event.set()
                    async for packet in self._close_open_turn():
                        await self.push_to_transcriber_queue(packet)
        except asyncio.CancelledError:
            logger.info("Qwen ASR completion timeout monitor cancelled")
            raise
        except Exception as e:
            logger.error(f"Qwen ASR completion timeout monitor error: {e}")
            raise

    # ------------------------------------------------------------------
    # Open-weights batch path (stream=false)
    # ------------------------------------------------------------------

    @property
    def transcriptions_url(self):
        return f"{self.base_url}/audio/transcriptions"

    @staticmethod
    def _rms(pcm):
        """Frame loudness. audioop raises on a partial sample, which a short tail can be."""
        try:
            return audioop.rms(pcm, 2) if len(pcm) >= 2 else 0
        except audioop.error:
            return 0

    def _batch_frame_is_speech(self, pcm):
        return self._rms(pcm) > self.speech_rms_threshold

    def _utterance_is_overlong(self, now):
        """True once a caller who never pauses has filled the cap — flush without a silence."""
        if self._utterance_started_at is None:
            return False
        return (now - self._utterance_started_at) > self.max_utterance_s

    async def _post_transcription(self, pcm):
        """POST buffered PCM as a WAV to the OpenAI-compatible endpoint; return text or None."""
        if self._http_session is None:
            self._http_session = aiohttp.ClientSession()

        wav = pcm_to_wav_bytes(bytes(pcm), sample_rate=self.sampling_rate)
        form = aiohttp.FormData()
        form.add_field("file", wav, filename="utterance.wav", content_type="audio/wav")
        form.add_field("model", self.model)
        form.add_field("response_format", "json")
        language = self._resolve_language()
        if language:
            form.add_field("language", language)

        started = time.time()
        try:
            async with self._http_session.post(
                self.transcriptions_url,
                data=form,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=aiohttp.ClientTimeout(total=self.http_timeout_s),
            ) as resp:
                body = await resp.text()
                if resp.status != 200:
                    self.connection_error = f"HTTP {resp.status}: {body[:200]}"
                    logger.error(f"Qwen ASR batch request failed — {self.connection_error}")
                    return None
                try:
                    payload = json.loads(body)
                    text = (payload.get("text") or "").strip()
                except Exception:
                    logger.error(f"Qwen ASR batch: unparseable response: {body[:200]}")
                    return None
        except asyncio.TimeoutError:
            # Record it: the non-200 branch sets connection_error, and without the same here a
            # timed-out turn vanishes from the call record with no trace of why.
            self.connection_error = f"transcription timed out after {self.http_timeout_s}s"
            logger.error(f"Qwen ASR batch request timed out after {self.http_timeout_s}s — utterance dropped")
            return None
        except Exception as e:
            logger.error(f"Qwen ASR batch request error: {e}")
            return None

        # Bill what the host says it processed when it reports it (OpenRouter returns
        # usage.seconds); otherwise the buffer length, which is the same audio.
        reported = (payload.get("usage") or {}).get("seconds") if isinstance(payload.get("usage"), dict) else None
        self.total_audio_seconds += float(reported) if reported else len(pcm) / 2 / self.sampling_rate

        latency = time.time() - started
        self.meta_info["transcriber_latency"] = latency
        logger.info(f"Qwen ASR batch: {len(pcm)} bytes → {latency:.2f}s → {text[:120]!r}")
        return text

    async def _flush_utterance(self):
        """Transcribe and emit the buffered utterance, then close the turn."""
        pcm, self._utterance_buffer = self._utterance_buffer, bytearray()
        self._speech_active = False
        self._silence_started_at = None
        self._utterance_started_at = None

        # Below ~200ms there is nothing a recogniser can do but hallucinate a word.
        if len(pcm) < self.sampling_rate // 5 * 2:
            logger.info("Qwen ASR batch: utterance too short to transcribe, closing turn")
            await self.push_to_transcriber_queue(create_ws_data_packet({"type": "speech_ended"}, self.meta_info))
            self._reset_turn_state()
            return

        text = await self._post_transcription(pcm)
        if not text:
            await self.push_to_transcriber_queue(create_ws_data_packet({"type": "speech_ended"}, self.meta_info))
            self._reset_turn_state()
            return

        self._build_finalized_turn_latency(text)
        self._stamp_turn_meta()
        await self.push_to_transcriber_queue(
            create_ws_data_packet({"type": "transcript", "content": text}, self.meta_info)
        )
        self._reset_turn_state()

    async def transcribe_http(self):
        """Buffer audio, endpoint it locally on frame RMS, and POST each utterance.

        The endpoint is batch, so this is where the turn boundary has to come from — the
        provider is not going to tell us. Mirrors the shape sarvam_transcriber uses.
        """
        logger.info(
            f"Qwen ASR batch mode: {self.transcriptions_url} model={self.model} "
            f"rate={self.sampling_rate} encoding={self.encoding} "
            f"endpointing={self.endpointing_ms}ms rms_threshold={self.speech_rms_threshold}"
        )
        if not self.api_key:
            self.connection_error = "Qwen ASR: no API key for the batch endpoint"
            logger.error(self.connection_error)

        try:
            while True:
                try:
                    packet = await asyncio.wait_for(self.input_queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    # No audio arriving still advances the silence clock, or a turn that ends
                    # exactly when the stream goes quiet would never be flushed.
                    if self._speech_active and self._silence_started_at is not None:
                        if (time.time() - self._silence_started_at) * 1000 >= self.endpointing_ms:
                            await self._flush_utterance()
                    continue

                meta = packet.get("meta_info")
                if meta and not self.audio_submitted:
                    self.meta_info = meta
                    self.audio_submitted = True
                    self.audio_submission_time = time.time()
                    self.current_request_id = self.generate_request_id()
                    self.meta_info["request_id"] = self.current_request_id

                if (meta or {}).get("eos") is True:
                    if self._speech_active:
                        logger.info("Qwen ASR batch: EOS with a turn open — flushing it")
                        await self._flush_utterance()
                    break

                raw = packet.get("data")
                if not raw:
                    continue
                pcm = self._to_pcm16(raw)
                if not pcm:
                    continue
                self.num_frames += 1

                now = time.time()
                if self._batch_frame_is_speech(pcm):
                    self._silence_started_at = None
                    if not self._speech_active:
                        self._speech_active = True
                        self._utterance_started_at = now
                        self._start_turn()
                        await self.push_to_transcriber_queue(create_ws_data_packet("speech_started", self.meta_info))
                    self._utterance_buffer.extend(pcm)
                elif self._speech_active:
                    # Trailing quiet belongs to the utterance — cutting it strands the last word.
                    self._utterance_buffer.extend(pcm)
                    if self._silence_started_at is None:
                        self._silence_started_at = now
                    elif (now - self._silence_started_at) * 1000 >= self.endpointing_ms:
                        await self._flush_utterance()
                        continue

                if self._speech_active and self._utterance_is_overlong(now):
                    logger.info(f"Qwen ASR batch: utterance hit the {self.max_utterance_s}s cap — flushing")
                    await self._flush_utterance()

        except asyncio.CancelledError:
            logger.info("Qwen ASR batch loop cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in Qwen ASR batch loop: {e}")
            self.connection_error = str(e)
            traceback.print_exc()
        finally:
            meta = dict(self.meta_info or {})
            meta["transcriber_duration"] = round(self.total_audio_seconds, 3)
            if self.connection_error:
                meta["connection_error"] = self.connection_error
            await self.push_to_transcriber_queue(create_ws_data_packet("transcriber_connection_closed", meta))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def push_to_transcriber_queue(self, data_packet):
        if self.transcriber_output_queue is not None:
            await self.transcriber_output_queue.put(data_packet)

    def get_meta_info(self):
        return self.meta_info

    async def toggle_connection(self):
        self.connection_on = False
        for task in (self.sender_task, self.heartbeat_task, self.completion_timeout_task):
            if task is not None:
                task.cancel()
        if self.websocket_connection is not None:
            try:
                await self.websocket_connection.close()
                logger.info("Qwen ASR websocket connection closed")
            except Exception as e:
                logger.error(f"Error closing Qwen ASR websocket: {e}")
            finally:
                self.websocket_connection = None
                self.connection_authenticated = False

        if self._http_session is not None:
            try:
                await self._http_session.close()
            except Exception as e:
                logger.error(f"Error closing Qwen ASR http session: {e}")
            finally:
                self._http_session = None

    async def cleanup(self):
        logger.info("Cleaning up Qwen ASR transcriber resources")
        self.connection_on = False
        for task_name, task in [
            ("sender_task", getattr(self, "sender_task", None)),
            ("heartbeat_task", getattr(self, "heartbeat_task", None)),
            ("completion_timeout_task", getattr(self, "completion_timeout_task", None)),
            ("transcription_task", getattr(self, "transcription_task", None)),
        ]:
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    logger.info(f"Qwen ASR {task_name} cancelled")
                except Exception as e:
                    logger.warning(f"Error cancelling Qwen ASR {task_name}: {e}")

        if self.websocket_connection is not None:
            try:
                await self.websocket_connection.close()
            except Exception as e:
                logger.error(f"Error closing Qwen ASR websocket: {e}")
            finally:
                self.websocket_connection = None
                self.connection_authenticated = False

        if self._http_session is not None:
            try:
                await self._http_session.close()
            except Exception as e:
                logger.error(f"Error closing Qwen ASR http session: {e}")
            finally:
                self._http_session = None

        self.current_turn_interim_details = []
        self._utterance_buffer = bytearray()

    async def run(self):
        try:
            # stream=true → the hosted realtime socket; stream=false → open weights over HTTP.
            runner = self.transcribe if self.stream else self.transcribe_http
            self.transcription_task = asyncio.create_task(runner())
        except Exception as e:
            logger.error(f"Error starting Qwen ASR transcriber: {e}")

    async def transcribe(self):
        ws = None
        self.num_frames = 0
        self.connection_start_time = None
        try:
            start_time = timestamp_ms()
            try:
                ws = await self.qwen_connect()
            except (ValueError, ConnectionError) as e:
                logger.error(f"Failed to establish Qwen ASR connection: {e}")
                self.connection_error = str(e)
                await self.toggle_connection()
                meta = dict(self.meta_info or {})
                meta["connection_error"] = self.connection_error
                await self.push_to_transcriber_queue(create_ws_data_packet("transcriber_connection_closed", meta))
                return

            if not self.connection_time:
                self.connection_time = round(timestamp_ms() - start_time)

            self.sender_task = asyncio.create_task(self.sender_stream(ws))
            self.heartbeat_task = asyncio.create_task(self.send_heartbeat(ws))
            self.completion_timeout_task = asyncio.create_task(self.monitor_completion_timeout())

            try:
                async for message in self.receiver(ws):
                    if self.connection_on:
                        await self.push_to_transcriber_queue(message)
                    else:
                        break
            except ConnectionClosedError as e:
                logger.error(f"Qwen ASR websocket closed during streaming: {e}")
                self.connection_error = str(e)
            except Exception as e:
                logger.error(f"Error during Qwen ASR streaming: {e}")
                self.connection_error = str(e)
                raise

        except (ValueError, ConnectionError) as e:
            logger.error(f"Connection error in Qwen ASR transcribe: {e}")
            self.connection_error = str(e)
            await self.toggle_connection()
        except Exception as e:
            logger.error(f"Unexpected error in Qwen ASR transcribe: {e}")
            self.connection_error = str(e)
            await self.toggle_connection()
        finally:
            if ws is not None:
                try:
                    await ws.close()
                except Exception as e:
                    logger.error(f"Error closing Qwen ASR websocket in finally: {e}")
                finally:
                    self.websocket_connection = None
                    self.connection_authenticated = False

            for task in (self.sender_task, self.heartbeat_task, self.completion_timeout_task):
                if task is not None:
                    task.cancel()

            meta = dict(self.meta_info or {})
            meta["transcriber_duration"] = round(self.total_audio_seconds, 3)
            if self.connection_error:
                meta["connection_error"] = self.connection_error
            await self.push_to_transcriber_queue(create_ws_data_packet("transcriber_connection_closed", meta))
