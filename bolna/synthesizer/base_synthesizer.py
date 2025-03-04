import io
import torchaudio
from bolna.helpers.logger_config import configure_logger
import asyncio

logger = configure_logger(__name__)


class BaseSynthesizer:
    def __init__(self, stream=True, buffer_size=40, event_loop=None):
        self.stream = stream
        self.buffer_size = buffer_size
        self.internal_queue = asyncio.Queue()

    def clear_internal_queue(self):
        logger.info(f"Clearing out internal queue")
        self.internal_queue = asyncio.Queue()
        
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
                yield buffer.strip() + " "
                buffer = ""

        if buffer:
            yield buffer.strip() + " "

    def resample(self, audio_bytes):
        audio_buffer = io.BytesIO(audio_bytes)
        waveform, orig_sample_rate = torchaudio.load(audio_buffer)
        resampler = torchaudio.transforms.Resample(orig_sample_rate, 8000)
        audio_waveform = resampler(waveform)
        audio_buffer = io.BytesIO()
        torchaudio.save(audio_buffer, audio_waveform, 8000, format="wav")
        audio_buffer.seek(0)
        audio_data = audio_buffer.read()
        return audio_data

    def get_engine(self):
        return "default"

    def supports_websocket(self):
        return True
