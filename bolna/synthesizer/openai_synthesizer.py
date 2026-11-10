import asyncio
import io
import os
import re
import time
from contextlib import aclosing

from dotenv import load_dotenv
from openai import AsyncOpenAI, DefaultAsyncHttpxClient

from .base_synthesizer import BaseSynthesizer
from bolna.constants import OPENAI_TTS_SPEED_MAX, OPENAI_TTS_SPEED_MIN
from bolna.helpers.function_calling_helpers import SSRFError, validate_outbound_url
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.utils import (
    StreamingPcmResampler,
    convert_audio_to_wav,
    create_ws_data_packet,
    pcm_to_ulaw,
    pcm_to_wav_bytes,
    resample,
)

logger = configure_logger(__name__)
load_dotenv()

_CUSTOM_ENDPOINT_TIMEOUT_SECONDS = 20
MULAW_SAMPLE_RATE = 8000


def _primary_language(language):
    """ "hi-IN" to "hi": speech servers take the bare language."""
    return re.split(r"[-_]", language.strip().lower(), maxsplit=1)[0] if isinstance(language, str) else language


class OPENAISynthesizer(BaseSynthesizer):
    def __init__(
        self,
        voice,
        audio_format="mp3",
        model="tts-1",
        stream=False,
        sampling_rate=8000,
        buffer_size=400,
        speed=1.0,
        endpoint=None,
        extra_body=None,
        response_format="mp3",
        pcm_sample_rate=24000,
        language=None,
        platform_endpoint=False,
        **kwargs,
    ):
        super().__init__(kwargs.get("task_manager_instance"), stream, buffer_size)
        self.voice = voice
        self.model = model
        self.sample_rate = int(sampling_rate) if isinstance(sampling_rate, str) else sampling_rate
        self.speed = float(speed)
        if not OPENAI_TTS_SPEED_MIN <= self.speed <= OPENAI_TTS_SPEED_MAX:
            raise ValueError(f"OpenAI speed must be between {OPENAI_TTS_SPEED_MIN} and {OPENAI_TTS_SPEED_MAX}")
        self.stream = False
        api_key = kwargs.get("synthesizer_key", os.getenv("OPENAI_API_KEY"))
        self.endpoint = endpoint
        # Only a customer's endpoint takes extra params; OpenAI's own API never gets them.
        self.extra_body = extra_body if endpoint else None
        # Only our own servers are told the language; a customer's server never asked for the field.
        self.platform_endpoint = platform_endpoint
        self.pcm_sample_rate = int(pcm_sample_rate)
        self.language = _primary_language(language) if platform_endpoint else None
        self._endpoint_checked = endpoint is None or platform_endpoint
        http_client = DefaultAsyncHttpxClient(follow_redirects=False) if endpoint else None
        limits = {"timeout": _CUSTOM_ENDPOINT_TIMEOUT_SECONDS, "max_retries": 1} if endpoint else {}
        self.async_client = AsyncOpenAI(api_key=api_key, base_url=endpoint, http_client=http_client, **limits)
        self._timed_turn_key = None
        self._timed_turn = None
        self._timed_turn_started = None
        # A customer's endpoint asked for raw PCM is played as it arrives; mp3 needs the whole clip to decode.
        self.streams_pcm = bool(endpoint) and response_format == "pcm"
        self.use_mulaw = kwargs.get("use_mulaw", False)
        if self.streams_pcm:
            target_rate = MULAW_SAMPLE_RATE if self.use_mulaw else self.sample_rate
            self.pcm_resampler = StreamingPcmResampler(self.pcm_sample_rate, target_rate)
            self._pcm_carry = b""
            self._interruptions = 0
            self._chunk_read = None

    def supports_websocket(self):
        return True

    def get_sleep_time(self):
        return 0.01 if self.streams_pcm else super().get_sleep_time()

    # ------------------------------------------------------------------
    # BaseSynthesizer hooks
    # ------------------------------------------------------------------

    def _process_http_audio(self, audio):
        # OpenAI always returns mp3 — convert + resample to target rate
        return resample(convert_audio_to_wav(audio, "mp3"), self.sample_rate, format="wav")

    async def _fetch_http_audio(self, text, meta_info=None):
        started = time.perf_counter()
        if meta_info is not None:
            self._begin_turn_timing(meta_info, started)
        audio = await super()._fetch_http_audio(text, meta_info)
        if meta_info is not None:
            self._finish_chunk_timing(meta_info, len(text), started, time.perf_counter())
        return audio

    # One record per turn, in the shape the streaming providers write and the call timeline reads.
    def _begin_turn_timing(self, meta_info, started):
        key = (meta_info.get("sequence_id"), meta_info.get("message_category"))
        if key == self._timed_turn_key:
            return
        self._timed_turn_key, self._timed_turn_started = key, started
        self._timed_turn = {
            "turn_id": meta_info.get("turn_id"),
            "sequence_id": key[0],
            "tts_start_ms": meta_info.get("tts_start_ms"),
            "message_category": key[1],
        }
        # Written before any audio, so a turn interrupted mid-request still shows its start.
        self._upsert_turn_latency(dict(self._timed_turn))

    def _finish_chunk_timing(self, meta_info, characters, started, finished):
        turn = self._timed_turn
        # Non-streaming: a turn's first audio is its first chunk's whole clip.
        turn.setdefault("first_result_latency_ms", round((finished - started) * 1000))
        turn["characters"] = turn.get("characters", 0) + characters
        turn["total_stream_duration_ms"] = round((finished - self._timed_turn_started) * 1000)
        self._upsert_turn_latency(dict(turn))
        if meta_info.get("end_of_llm_stream"):
            self._timed_turn_key = None

    def _request_extras(self):
        if self.language is None:
            return self.extra_body
        return {**(self.extra_body or {}), "language": self.language}

    async def set_target_language(self, language):
        if self.language is not None and language:
            self.language = _primary_language(language)

    async def _check_endpoint(self):
        if self._endpoint_checked:
            return
        try:
            await validate_outbound_url(self.endpoint)
        except SSRFError as e:
            logger.warning(f"Blocked custom TTS endpoint: {e}")
            reason = "it resolves to a non-public address" if "address" in str(e) else str(e)
            raise SSRFError(f"Custom TTS endpoint rejected: {reason}") from None
        self._endpoint_checked = True

    async def _generate_http(self, text, response_format="mp3"):
        await self._check_endpoint()
        spoken_response = await self.async_client.audio.speech.create(
            model=self.model,
            voice=self.voice,
            response_format=response_format,
            input=text,
            speed=self.speed,
            extra_body=self._request_extras(),
        )
        buffer = io.BytesIO()
        for chunk in spoken_response.iter_bytes(chunk_size=4096):
            buffer.write(chunk)
        buffer.seek(0)
        return buffer.getvalue()

    async def synthesize(self, text):
        if not self.platform_endpoint:
            return await self._generate_http(text)
        # Callers of our own servers get WAV, whose header carries the rate; they decode nothing else for it.
        audio = await self._generate_http(text, response_format="pcm")
        return pcm_to_wav_bytes(audio, sample_rate=self.pcm_sample_rate) if audio else None

    # ------------------------------------------------------------------
    # generate / push — use base _generate_http_loop
    # ------------------------------------------------------------------

    async def generate(self):
        loop = self._generate_pcm_stream_loop() if self.streams_pcm else self._generate_http_loop()
        async for packet in loop:
            yield packet

    # ------------------------------------------------------------------
    # Streamed PCM: one request per text piece, audio forwarded chunk by chunk
    # ------------------------------------------------------------------

    async def handle_interruption(self):
        if not self.streams_pcm:
            return
        self._interruptions += 1
        self._reset_pcm_stream()
        # Wakes the loop out of a read that may never return; it then sees the count has moved.
        if self._chunk_read is not None:
            self._chunk_read.cancel()

    async def monitor_connection(self):
        """Open the connection at call start, so the first reply does not pay for the handshake."""
        if not self.streams_pcm:
            return
        started = time.perf_counter()
        try:
            await self._check_endpoint()
            await self.async_client.models.list()
        except Exception as e:
            logger.info(f"Speech endpoint warm-up did not complete: {e}")
        self.connection_time = round((time.perf_counter() - started) * 1000)

    def _reset_pcm_stream(self):
        self.pcm_resampler.reset()
        self._pcm_carry = b""
        self.first_chunk_generated = False

    def _convert_pcm_chunk(self, chunk):
        """Resample a slice of the response body as part of one stream; mu-law for telephony."""
        # A network chunk can end mid-sample; the odd byte waits for the next one.
        data = self._pcm_carry + chunk
        whole = len(data) - len(data) % 2
        data, self._pcm_carry = data[:whole], data[whole:]
        audio = self.pcm_resampler.process(data) if data else b""
        if not audio:
            return None
        return pcm_to_ulaw(audio) if self.use_mulaw else audio

    async def _stream_pcm(self, text):
        await self._check_endpoint()
        async with self.async_client.audio.speech.with_streaming_response.create(
            model=self.model,
            voice=self.voice,
            response_format="pcm",
            input=text,
            speed=self.speed,
            extra_body=self._request_extras(),
        ) as response:
            async for chunk in response.iter_bytes():
                audio = self._convert_pcm_chunk(chunk)
                if audio:
                    yield audio
        # A body that ends mid-sample must not shift every sample of the next piece.
        self._pcm_carry = b""

    async def _next_audio(self, stream):
        """The stream's next audio, or None at its end; handle_interruption can cancel the wait."""

        async def read():
            try:
                return await stream.__anext__()
            except StopAsyncIteration:
                return None

        self._chunk_read = asyncio.ensure_future(read())
        try:
            return await self._chunk_read
        finally:
            self._chunk_read = None

    async def _generate_pcm_stream_loop(self):
        while True:
            message = await self.internal_queue.get()
            meta_info, text = message.get("meta_info"), message.get("data")
            sequence_id = meta_info.get("sequence_id")
            if not self.should_synthesize_response(sequence_id):
                logger.info(f"Not synthesizing: sequence_id {sequence_id} not current")
                return

            meta_info["format"] = "mulaw" if self.use_mulaw else "pcm"
            meta_info["text"] = text
            meta_info["is_cached"] = False

            if text and text.strip():
                self.synthesized_characters += len(text)
                started = time.perf_counter()
                self._begin_turn_timing(meta_info, started)
                # Canned speech shares sequence -1, which never retires, so a barge-in is seen by its count.
                interruptions = self._interruptions
                first_audio = True
                async with aclosing(self._stream_pcm(text)) as stream:
                    while True:
                        try:
                            audio = await self._next_audio(stream)
                        except asyncio.CancelledError:
                            # Only handle_interruption cancels the read alone; anything else is this task ending.
                            if self._interruptions == interruptions:
                                raise
                            audio = None
                        except Exception:
                            # A request nobody is waiting for any more must not end the call.
                            if self._interruptions == interruptions and self.should_synthesize_response(sequence_id):
                                raise
                            audio = None
                        if self._interruptions != interruptions:
                            logger.info(f"Stopping stream: sequence_id {sequence_id} was interrupted")
                            return
                        if not self.should_synthesize_response(sequence_id):
                            logger.info(f"Stopping stream: sequence_id {sequence_id} not current")
                            self._reset_pcm_stream()
                            # The consumer clears its playing flag on a retired sequence's end packet.
                            retired = {"sequence_id": sequence_id, "end_of_synthesizer_stream": True}
                            retired["format"] = meta_info["format"]
                            yield create_ws_data_packet(b"\x00", retired)
                            return
                        if audio is None:
                            break
                        # Only the end packet closes the turn; audio that says so releases a hangup wait early.
                        packet_meta = {**meta_info, "end_of_llm_stream": False}
                        if first_audio:
                            first_audio = False
                            if "first_result_latency_ms" not in self._timed_turn:
                                latency = time.perf_counter() - started
                                self._timed_turn["first_result_latency_ms"] = round(latency * 1000)
                                self._upsert_turn_latency(dict(self._timed_turn))
                                packet_meta["synthesizer_latency"] = latency
                        self._stamp_first_chunk(packet_meta)
                        self._stamp_mark_id(packet_meta)
                        yield create_ws_data_packet(audio, packet_meta)
                self._finish_chunk_timing(meta_info, len(text), started, time.perf_counter())

            if meta_info.get("end_of_llm_stream"):
                self._reset_pcm_stream()
                self._stamp_end_of_stream(meta_info)
                self._stamp_mark_id(meta_info)
                yield create_ws_data_packet(b"\x00", meta_info)
