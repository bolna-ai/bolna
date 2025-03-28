import io
import uuid
import torchaudio
from bolna.helpers.logger_config import configure_logger
import asyncio
from bolna.helpers.utils import create_ws_data_packet

logger = configure_logger(__name__)


class BaseSynthesizer:
    def __init__(self, task_manager_instance=None, stream=True, buffer_size=40, event_loop=None, is_web_based_call=False,
                 is_precise_transcript_generation_enabled=True):
        self.stream = stream
        self.buffer_size = buffer_size
        self.internal_queue = asyncio.Queue()
        self.audio_chunks_sent = 0
        self.is_web_based_call = is_web_based_call
        self.is_precise_transcript_generation_enabled = is_precise_transcript_generation_enabled
        self.task_manager_instance = task_manager_instance

    def clear_internal_queue(self):
        logger.info(f"Clearing out internal queue")
        self.internal_queue = asyncio.Queue()

    def get_audio_chunks_sent(self):
        audio_chunks_sent = self.audio_chunks_sent
        self.audio_chunks_sent = 0
        return audio_chunks_sent

    def should_synthesize_response(self, sequence_id):
        return self.task_manager_instance.is_sequence_id_in_current_ids(sequence_id)

    async def flush_synthesizer_stream(self):
        pass

    async def break_audio_into_chunks(self, audio, slicing_range, meta_info, override_end_of_synthesizer_stream=False):
        is_first_chunk_sent = False
        try:
            audio_len = len(audio)
            for i in range(0, audio_len, slicing_range):
                is_last_iteration = (i + slicing_range >= len(audio))
                if is_first_chunk_sent:
                    meta_info["text_synthesized"] = ""
                is_first_chunk_sent = True
                meta_info["mark_id"] = str(uuid.uuid4())
                self.audio_chunks_sent += 1
                sliced_audio = audio[i: i + slicing_range]
                if override_end_of_synthesizer_stream:
                    meta_info["end_of_synthesizer_stream"] = is_last_iteration
                yield create_ws_data_packet(sliced_audio, meta_info)
        except Exception as e:
            logger.error(f"Error in break_audio_into_chunks - {e}")
            yield create_ws_data_packet(audio, meta_info)

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
                else:
                    logger.info(f"In else condition of text chunker where buffer = {buffer}")
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
