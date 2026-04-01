import io
import wave
from pydub import AudioSegment
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.ssml_config import PROVIDER_ALLOWED_TAGS, convert_markers_to_ssml
import asyncio
import re

logger = configure_logger(__name__)

_XML_TAG_RE = re.compile(r'<(/?)(\w[\w:-]*)([^>]*)/?>')


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
        """Split text into chunks, keeping XML/SSML tags as atomic standalone
        chunks (required for ElevenLabs SSML parsing in streaming mode).
        Paired tags like <say-as>content</say-as> are kept as single chunks."""
        splitters = (".", ",", "?", "!", ";", ":", "—", "-", "(", ")", "[", "]", "}", " ")

        buffer = ""
        inside_tag = False
        inside_element = False
        for char in text:
            buffer += char
            if char == "<":
                inside_tag = True
            elif char == ">" and inside_tag:
                inside_tag = False
                tag_text = buffer[buffer.rfind("<"):]
                if tag_text.startswith("</"):
                    inside_element = False
                elif not tag_text.rstrip().endswith("/>"):
                    inside_element = True
                continue
            if not inside_tag and not inside_element and char in splitters:
                if buffer != " ":
                    yield buffer.strip() + " "
                buffer = ""

        if buffer:
            yield buffer.strip() + " "

    def normalize_text(self, s):
        return re.sub(r'\s+', ' ', s.strip())

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

    @staticmethod
    def strip_unsupported_tags(text: str, provider: str) -> str:
        """
        Remove XML/SSML tags that the given provider does not support.
        Tags listed in PROVIDER_ALLOWED_TAGS for the provider are kept;
        everything else is stripped (tag removed, inner text preserved).
        If the provider has no entry at all, all tags are removed.
        """
        allowed = PROVIDER_ALLOWED_TAGS.get(provider, set())
        if not allowed:
            return re.sub(r'<[^>]+>', '', text).strip()

        def _replace(match):
            tag_name = match.group(2)
            if tag_name in allowed:
                return match.group(0)
            return ""

        cleaned = _XML_TAG_RE.sub(_replace, text)
        return re.sub(r'\s+', ' ', cleaned).strip()
