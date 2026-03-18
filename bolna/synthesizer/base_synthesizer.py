import io
import copy
import uuid
import time
import asyncio
import re
import traceback

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

    # ------------------------------------------------------------------
    # Stubs for subclass override
    # ------------------------------------------------------------------

    async def flush_synthesizer_stream(self):
        pass

    async def generate(self):
        pass

    async def push(self, message):
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
        return re.sub(r'\s+', ' ', s.strip())

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
