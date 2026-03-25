import io
import wave
from pydub import AudioSegment
from bolna.helpers.logger_config import configure_logger
import asyncio
import re

logger = configure_logger(__name__)

SSML_ROOT_RE = re.compile(r'<speak[\s>]', re.IGNORECASE)
SSML_INLINE_RE = re.compile(
    r'<(break|say-as|emphasis|phoneme|sub|prosody|mstts:|amazon:)', re.IGNORECASE
)
XML_TAG_RE = re.compile(r'<[^>]+>')


class BaseSynthesizer:
    def __init__(self, task_manager_instance=None, stream=True, buffer_size=40, event_loop=None):
        self.stream = stream
        self.buffer_size = buffer_size
        self.internal_queue = asyncio.Queue()
        self.task_manager_instance = task_manager_instance
        self.connection_time = None
        self.turn_latencies = []

    def clear_internal_queue(self):
        logger.info(f"Clearing out internal queue")
        self.internal_queue = asyncio.Queue()

    def should_synthesize_response(self, sequence_id):
        return self.task_manager_instance.is_sequence_id_in_current_ids(sequence_id)

    async def flush_synthesizer_stream(self):
        pass

    def generate(self):
        pass

    def push(self, text):
        pass
    
    def synthesize(self, text):
        pass

    def get_synthesized_characters(self):
        return 0

    async def monitor_connection(self):
        pass

    async def cleanup(self):
        pass

    async def handle_interruption(self):
        pass

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
    # SSML helpers — available to all synthesizer subclasses
    # ------------------------------------------------------------------

    def is_ssml(self, text: str) -> bool:
        """Return True if text is a complete SSML document (has a <speak> root)."""
        return bool(SSML_ROOT_RE.search(text.strip()))

    def has_inline_ssml(self, text: str) -> bool:
        """Return True if text contains inline SSML tags (break, say-as, emphasis …)."""
        return bool(SSML_INLINE_RE.search(text))

    def strip_ssml(self, text: str) -> str:
        """Remove all XML/SSML tags and return clean plain text."""
        cleaned = XML_TAG_RE.sub('', text)
        return re.sub(r'\s+', ' ', cleaned).strip()

    def resample(self, audio_bytes):
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
        if audio.frame_rate != 8000:
            audio = audio.set_frame_rate(8000)
        buffer = io.BytesIO()
        audio.export(buffer, format="wav")
        return buffer.getvalue()

    def get_engine(self):
        return "default"

    def supports_websocket(self):
        return True
    
    def get_sleep_time(self):
        return 0.2
