import io
import time
import uuid
import asyncio
import re

from pydub import AudioSegment
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.utils import create_ws_data_packet

logger = configure_logger(__name__)


class BaseSynthesizer:
    def __init__(self, task_manager_instance=None, stream=True, buffer_size=40, event_loop=None):
        self.stream = stream
        self.buffer_size = buffer_size
        self.internal_queue = asyncio.Queue()
        self.task_manager_instance = task_manager_instance
        self.connection_time = None
        self.turn_latencies = []
        self.first_chunk_generated = False
        self.synthesized_characters = 0
        self.model = "default"
        self.current_turn_start_time = None
        self.current_turn_id = None

    # ------------------------------------------------------------------
    # Common accessors
    # ------------------------------------------------------------------

    def get_synthesized_characters(self):
        return self.synthesized_characters

    def get_engine(self):
        return self.model

    def supports_websocket(self):
        return True

    def get_sleep_time(self):
        return 0.2

    # ------------------------------------------------------------------
    # Queue helpers
    # ------------------------------------------------------------------

    def clear_internal_queue(self):
        logger.info("Clearing out internal queue")
        self.internal_queue = asyncio.Queue()

    def should_synthesize_response(self, sequence_id):
        return self.task_manager_instance.is_sequence_id_in_current_ids(sequence_id)

    async def push(self, message):
        meta_info = message.get("meta_info")
        self._stamp_turn_start(meta_info)

        self.internal_queue.put_nowait(message)

    # ------------------------------------------------------------------
    # Stubs for subclass override
    # ------------------------------------------------------------------

    async def flush_synthesizer_stream(self):
        pass

    async def generate(self):
        pass

    async def synthesize(self, text):
        pass

    async def monitor_connection(self):
        pass

    async def cleanup(self):
        pass

    async def handle_interruption(self):
        pass

    # ------------------------------------------------------------------
    # Meta-info helpers used by generate() implementations
    # ------------------------------------------------------------------

    def _stamp_first_chunk(self, meta_info):
        """Set is_first_chunk on meta_info and track first_chunk_generated state."""
        if not self.first_chunk_generated:
            meta_info["is_first_chunk"] = True
            self.first_chunk_generated = True
        else:
            meta_info["is_first_chunk"] = False

    def _stamp_end_of_stream(self, meta_info):
        """Mark end_of_synthesizer_stream when end_of_llm_stream is set."""
        if meta_info.get("end_of_llm_stream"):
            meta_info["end_of_synthesizer_stream"] = True
            self.first_chunk_generated = False

    def _stamp_mark_id(self, meta_info):
        meta_info["mark_id"] = str(uuid.uuid4())

    def _stamp_turn_start(self, meta_info):
        """Only stamp on the first push of a new turn (don't re-stamp on subsequent chunks)."""
        if self.current_turn_start_time is None:
            self.current_turn_start_time = time.perf_counter()
            logger.info(f"Push new_turn text_len={len(meta_info.get('text', '') or '')}")
        self.current_turn_id = meta_info.get("turn_id") or meta_info.get("sequence_id")

    def _record_turn_latency(self):
        """Append a latency record for the completed turn."""
        try:
            if self.current_turn_start_time is not None:
                total_stream_duration = time.perf_counter() - self.current_turn_start_time
                self.turn_latencies.append({
                    "turn_id": self.current_turn_id,
                    "sequence_id": self.current_turn_id,
                    "first_result_latency_ms": round(total_stream_duration * 1000),
                    "total_stream_duration_ms": round(total_stream_duration * 1000),
                })
                self.current_turn_start_time = None
                self.current_turn_id = None
        except Exception:
            logger.warning("Error recording turn latency", exc_info=True)
            pass

    # ------------------------------------------------------------------
    # HTTP generate loop (used by HTTP-only synths and dual-mode synths)
    # ------------------------------------------------------------------

    async def _generate_http(self, text):
        """Provider-specific HTTP TTS call. Return raw audio bytes."""
        raise NotImplementedError

    def _process_http_audio(self, audio):
        """Audio conversion for HTTP mode. Override per provider."""
        return audio

    def _get_http_audio_format(self):
        """Output format string for HTTP mode (e.g. 'wav', 'mulaw')."""
        return "wav"

    async def _generate_http_loop(self):
        """Standard HTTP (non-streaming) generate loop with caching support."""
        while True:
            message = await self.internal_queue.get()
            logger.info(f"Generating TTS response for message: {message}")
            meta_info, text = message.get("meta_info"), message.get("data")

            if not self.should_synthesize_response(meta_info.get("sequence_id")):
                logger.info(f"Not synthesizing: sequence_id {meta_info.get('sequence_id')} not current")
                continue

            audio = await self._fetch_http_audio(text, meta_info)
            audio = self._process_http_audio(audio)

            self._stamp_first_chunk(meta_info)
            self._stamp_end_of_stream(meta_info)

            meta_info["format"] = self._get_http_audio_format()
            meta_info["text"] = text
            meta_info["text_synthesized"] = f"{text} "
            self._stamp_mark_id(meta_info)

            self._record_turn_latency()

            yield create_ws_data_packet(audio, meta_info)

    async def _fetch_http_audio(self, text, meta_info=None):
        """Fetch audio via HTTP, with optional caching. Tracks synthesized_characters."""
        if getattr(self, "caching", False) and hasattr(self, "cache"):
            cached = self.cache.get(text)
            if cached:
                logger.info(f"Cache hit: {text}")
                if meta_info is not None:
                    meta_info["is_cached"] = True
                return cached
            logger.info("Not a cache hit")
            if meta_info is not None:
                meta_info["is_cached"] = False
            self.synthesized_characters += len(text)
            audio = await self._generate_http(text)
            if audio is not None and audio != b'\x00':
                self.cache.set(text, audio)
            return audio
        else:
            if meta_info is not None:
                meta_info["is_cached"] = False
            self.synthesized_characters += len(text)
            return await self._generate_http(text)

    # ------------------------------------------------------------------
    # Text utilities
    # ------------------------------------------------------------------

    def text_chunker(self, text):
        """Split text into chunks, ensuring to not break sentences."""
        splitters = (".", ",", "?", "!", ";", ":", "—", "-", "(", ")", "[", "]", "}", " ")

        buffer = ""
        for char in text:
            buffer += char
            if char in splitters:
                if buffer != " ":
                    yield buffer.strip() + " "
                buffer = ""

        if buffer:
            yield buffer.strip() + " "

    def normalize_text(self, s):
        return re.sub(r"\s+", " ", s.strip())

    # ------------------------------------------------------------------
    # Audio utilities
    # ------------------------------------------------------------------

    def resample(self, audio_bytes):
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
        if audio.frame_rate != 8000:
            audio = audio.set_frame_rate(8000)
        buffer = io.BytesIO()
        audio.export(buffer, format="wav")
        return buffer.getvalue()
