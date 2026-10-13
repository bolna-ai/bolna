import asyncio
import base64
import json
import os
import time
import traceback
import uuid
from collections import deque

import aiohttp
import websockets
from websockets.exceptions import InvalidHandshake

from .base_synthesizer import REJECTED_STATUSES, raise_if_rejected
from .stream_synthesizer import StreamSynthesizer
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.ssl_context import get_ssl_context
from bolna.helpers.utils import (
    StreamingPcmResampler,
    create_ws_data_packet,
    get_synth_audio_format,
    resample,
    wav_bytes_to_pcm,
)
from bolna.constants import (
    SARVAM_LOUDNESS_MAX,
    SARVAM_LOUDNESS_MIN,
    SARVAM_LOUDNESS_MODELS,
    SARVAM_MODEL_SAMPLING_RATE_MAPPING,
    SARVAM_STREAMING_WAV_HEADER_MODELS,
    SARVAM_TTS_SUPPORTED_LANGUAGES,
)

logger = configure_logger(__name__)

# Sarvam closes the socket after these (idle too long, a sentence it cannot speak); the close
# settles any open turn and monitor_connection redials.
SARVAM_RECOVERABLE_ERRORS = (400, 408, 422)


class SarvamSynthesizer(StreamSynthesizer):
    def __init__(
        self,
        voice_id,
        model,
        language,
        sampling_rate="8000",
        stream=False,
        buffer_size=400,
        speed=1.0,
        loudness=1.0,
        synthesizer_key=None,
        **kwargs,
    ):
        super().__init__(
            stream=stream,
            provider_name="sarvam",
            buffer_size=buffer_size,
            **kwargs,
        )
        self.api_key = os.environ["SARVAM_API_KEY"] if synthesizer_key is None else synthesizer_key
        self.voice_id = voice_id
        self.model = model
        self.stream = stream
        self.buffer_size = buffer_size
        if self.buffer_size < 30 or self.buffer_size > 200:
            self.buffer_size = 200

        self.sampling_rate = int(sampling_rate)
        self.original_sampling_rate = SARVAM_MODEL_SAMPLING_RATE_MAPPING.get(model, None)
        # Built on first raw-PCM chunk, once the stream header has confirmed the source rate.
        self.stream_resampler = None
        self.stream_resampler_rate = None
        # (socket, sequence_id) per flush sent. Sarvam has no per-turn cancel, so each flush ends in a
        # `final`, an abandoned turn's included, and the oldest entry owns the next one.
        self.flushed_sequences = deque()
        self.api_url = "https://api.sarvam.ai/text-to-speech"
        self.ws_url = f"wss://api.sarvam.ai/text-to-speech/ws?model={model}&send_completion_event=true"

        self.language = language
        self.loudness = float(loudness)
        if not SARVAM_LOUDNESS_MIN <= self.loudness <= SARVAM_LOUDNESS_MAX:
            raise ValueError(f"Sarvam loudness must be between {SARVAM_LOUDNESS_MIN} and {SARVAM_LOUDNESS_MAX}")
        self.pitch = 0.0
        self.pace = speed
        self.enable_preprocessing = True

    def get_sleep_time(self):
        return 0.01

    # ------------------------------------------------------------------
    # StreamSynthesizer hooks
    # ------------------------------------------------------------------

    def _get_audio_format(self):
        return "wav"

    def _unpack_receiver_message(self, item):
        return item if isinstance(item, tuple) else (item, {})

    def take_flush_owner(self, socket):
        """Sequence whose flush this `final` on `socket` answers. Flushes on a replaced socket never get one."""
        while self.flushed_sequences:
            flushed_socket, sequence_id = self.flushed_sequences.popleft()
            if flushed_socket is socket:
                return sequence_id
        return None

    def _process_audio_chunk(self, chunk):
        """WS chunk. Raw PCM after the stream header is one continuous stream, resampled as such;
        per-chunk resampling clicks at every chunk edge. Anything else converts on its own."""
        if self.original_sampling_rate is None or get_synth_audio_format(chunk) != "pcm":
            return self._process_audio_data(chunk)
        if self.stream_resampler is None or self.stream_resampler_rate != self.original_sampling_rate:
            self.stream_resampler = StreamingPcmResampler(self.original_sampling_rate, int(self.sampling_rate))
            self.stream_resampler_rate = self.original_sampling_rate
        try:
            return self.stream_resampler.process(chunk) or None
        except Exception as e:
            logger.error(f"Error in resampling audio: {e}")
            return None

    def _process_http_audio(self, audio):
        """A complete HTTP clip converts in one pass, outside the WS stream."""
        return self._process_audio_data(audio)

    def reset_stream_resampler(self):
        # A finished or abandoned turn's held-back samples must not lead the next turn.
        if self.stream_resampler is not None:
            self.stream_resampler.reset()

    async def handle_interruption(self):
        self.reset_stream_resampler()
        await super().handle_interruption()

    def _process_audio_data(self, audio):
        fmt = get_synth_audio_format(audio)

        if fmt == "wav" and self.model in SARVAM_STREAMING_WAV_HEADER_MODELS:
            received_sampling_rate = int.from_bytes(audio[24:28], byteorder="little")
            if self.original_sampling_rate != received_sampling_rate:
                logger.warning(
                    f"Expected sampling rate {self.original_sampling_rate} does not match "
                    f"received {received_sampling_rate} for model {self.model}. Using received."
                )
                self.original_sampling_rate = received_sampling_rate
            return None  # Header-only chunk

        try:
            resampled_audio = resample(
                audio,
                int(self.sampling_rate),
                format=fmt,
                original_sample_rate=self.original_sampling_rate,
            )
        except Exception as e:
            logger.error(f"Error in resampling audio: {e}")
            return None

        if fmt == "wav":
            return wav_bytes_to_pcm(resampled_audio)
        return resampled_audio

    # ------------------------------------------------------------------
    # sender / receiver
    # ------------------------------------------------------------------

    async def sender(self, text, sequence_id, end_of_llm_stream=False):
        try:
            if self.conversation_ended:
                return
            if not self.should_synthesize_response(sequence_id):
                logger.info(f"Not synthesizing: sequence_id {sequence_id} not current")
                return

            await self._wait_for_ws()

            if text != "":
                try:
                    if self.ws_send_time is None:
                        self.ws_send_time = time.perf_counter()
                    # Claim the socket before sending, not after: text lost into a socket that
                    # dies mid-send is still this turn's audio, and still has to settle.
                    self.current_turn_socket = self.websocket
                    await self._send_json({"type": "text", "data": {"text": text}})
                except Exception as e:
                    logger.error(f"Error sending chunk: {e}")
                    self.connection_error = str(e)
                    return

            if end_of_llm_stream:
                self.last_text_sent = True
                try:
                    self.flushed_sequences.append((self.websocket, sequence_id))
                    await self._send_json({"type": "flush"})
                except Exception as e:
                    logger.info(f"Error sending end-of-stream signal: {e}")
                    self.connection_error = str(e)

        except asyncio.CancelledError:
            logger.info("Sender task was cancelled.")
        except Exception as e:
            logger.error(f"Unexpected error in sender: {e}")

    async def receiver(self):
        not_connected_since = None
        draining_ws = None
        while True:
            try:
                if self.conversation_ended:
                    return
                if not self._is_ws_connected():
                    # A socket dying between recvs surfaces here, not below. self.websocket is
                    # still the dead one — once monitor_connection swaps it, this branch is skipped.
                    if self.has_unsettled_turn_on(self.websocket):
                        self.reset_stream_resampler()
                        yield b"\x00"
                    if self.connection_error:
                        return
                    now = time.perf_counter()
                    if not_connected_since is None:
                        not_connected_since = now
                    elif now - not_connected_since > 30:
                        logger.error("Sarvam receiver: WebSocket never connected after 30s, giving up.")
                        self.connection_error = self.connection_error or "WebSocket never connected"
                        return
                    logger.info("WebSocket is not connected, skipping receive.")
                    await asyncio.sleep(0.1)
                    continue
                else:
                    not_connected_since = None

                # A closing socket drains buffered frames first; they must not play as
                # the next turn, which the replacement socket is already serving.
                draining_ws = self.websocket
                response = await draining_ws.recv()
                if draining_ws is not self.websocket:
                    logger.info("Dropping frame from a replaced Sarvam socket")
                    continue
                data = json.loads(response)

                if data.get("type") == "audio":
                    chunk = base64.b64decode(data["data"]["audio"])
                    yield chunk

                event_type = data.get("data", {}).get("event_type") or data.get("event_type")
                if event_type == "final":
                    self.reset_stream_resampler()
                    owner = self.take_flush_owner(draining_ws)
                    if owner is not None and not self.should_synthesize_response(owner):
                        # An abandoned turn's final: the turn now streaming keeps its flag and its own eos.
                        yield b"\x00", {"retired_sequence_id": owner}
                        continue
                    self.last_text_sent = False
                    yield b"\x00"
                    continue

                if data.get("type") == "error":
                    logger.error(f"Sarvam TTS error response: {data}")
                    if data.get("data", {}).get("code") in SARVAM_RECOVERABLE_ERRORS:
                        continue
                    self.connection_error = json.dumps(data)
                    return

            except websockets.exceptions.ConnectionClosed:
                # Keep looping, never break: SynthesizerPool iterates generate() once, so
                # ending here leaves the call with no audio path at all.
                logger.info("Sarvam WebSocket connection closed, waiting for reconnect")
                if self.conversation_ended or self.connection_error:
                    return
                # A turn dying with the socket gets no 'final'; end it or playback stays
                # marked in progress for the rest of the call.
                if self.has_unsettled_turn_on(draining_ws):
                    logger.error("Sarvam WebSocket dropped mid-turn, ending the turn")
                    self.reset_stream_resampler()
                    yield b"\x00"
                await asyncio.sleep(0.05)
            except Exception as e:
                logger.error(f"Error occurred in receiver - {e}")

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def _config_message(self):
        data = {
            "target_language_code": self.language,
            "speaker": self.voice_id,
            "pace": self.pace,
            "enable_preprocessing": self.enable_preprocessing,
            "output_audio_codec": "wav",
            "output_audio_bitrate": "32k",
            "max_chunk_length": 250,
            "min_buffer_size": self.buffer_size,
        }
        if self.model in SARVAM_LOUDNESS_MODELS:
            data.update({"pitch": self.pitch, "loudness": self.loudness})
        return {"type": "config", "data": data}

    async def set_target_language(self, language):
        """Switch TTS output language on the live socket via a fresh config message (no reconnect)."""
        if not language or language == self.language:
            return
        if language not in SARVAM_TTS_SUPPORTED_LANGUAGES:
            logger.info(f"Sarvam TTS: detected language {language!r} not supported, keeping {self.language!r}")
            return
        self.language = language
        if self._is_ws_connected():
            try:
                await self._send_json(self._config_message())
                logger.info(f"Sarvam TTS: switched target_language_code -> {language}")
            except Exception as e:
                logger.error(f"Sarvam TTS: failed to send config update for {language!r}: {e}")

    async def establish_connection(self):
        try:
            start_time = time.perf_counter()
            websocket = await asyncio.wait_for(
                websockets.connect(
                    self.ws_url,
                    additional_headers={"api-subscription-key": self.api_key},
                    ssl=get_ssl_context(self.ws_url),
                ),
                timeout=10.0,
            )
            await websocket.send(json.dumps(self._config_message()))
            if not self.connection_time:
                self.connection_time = round((time.perf_counter() - start_time) * 1000)
            logger.info(f"Connected to {self.ws_url}")
            return websocket
        except asyncio.TimeoutError:
            logger.error("Timeout while connecting to Sarvam TTS websocket")
            return None
        except InvalidHandshake as e:
            error_msg = str(e)
            if "401" in error_msg or "403" in error_msg:
                logger.error(f"Sarvam TTS authentication failed: {e}")
            elif "404" in error_msg:
                logger.error(f"Sarvam TTS endpoint not found: {e}")
            else:
                logger.error(f"Sarvam TTS handshake failed: {e}")
            if getattr(getattr(e, "response", None), "status_code", None) in REJECTED_STATUSES:
                self.connection_error = str(e)
            return None
        except Exception as e:
            logger.error(f"Failed to connect to Sarvam TTS: {e}")
            return None

    # ------------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------------

    async def _send_payload(self, payload):
        headers = {"api-subscription-key": self.api_key, "Content-Type": "application/json"}
        async with aiohttp.ClientSession() as session:
            async with session.post(self.api_url, headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    if data and isinstance(data.get("audios", []), list) and data["audios"]:
                        return data["audios"][0]
                else:
                    detail = await response.text()
                    logger.error(f"Error: {response.status} - {detail}")
                    raise_if_rejected("Sarvam", response.status, detail)

    async def synthesize(self, text):
        return await self._generate_http(text)

    def _http_payload(self, text):
        payload = {
            "target_language_code": self.language,
            "text": text,
            "speaker": self.voice_id,
            "pace": self.pace,
            "speech_sample_rate": self.sampling_rate,
            "enable_preprocessing": self.enable_preprocessing,
            "model": self.model,
        }
        if self.model in SARVAM_LOUDNESS_MODELS:
            payload.update({"pitch": self.pitch, "loudness": self.loudness})
        return payload

    async def _generate_http(self, text):
        payload = self._http_payload(text)
        return await self._send_payload(payload)
