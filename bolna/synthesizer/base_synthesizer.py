import io
import torchaudio
import asyncio
import copy
import uuid
import aiohttp
from collections import deque
from typing import Any, Optional
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
        
        self.websocket_holder = {"websocket": None}
        self.conversation_ended = False
        self.sender_task = None
        
        self.first_chunk_generated = False
        self.last_text_sent = False
        self.text_queue = deque()
        self.meta_info = None
        self.synthesized_characters = 0
        self.current_text = ""

    def clear_internal_queue(self):
        logger.info(f"Clearing out internal queue")
        self.internal_queue = asyncio.Queue()

    def should_synthesize_response(self, sequence_id):
        if self.task_manager_instance:
            return self.task_manager_instance.is_sequence_id_in_current_ids(sequence_id)
        return True

    async def flush_synthesizer_stream(self):
        pass

    def generate(self):
        pass

    async def push(self, message):
        """Push message to internal queue, handling both streaming and non-streaming modes"""
        logger.info(f"Pushed message to internal queue {message}")
        if self.stream:
            meta_info, text = message.get("meta_info"), message.get("data")
            self.current_text = text
            self.synthesized_characters += len(text) if text is not None else 0
            end_of_llm_stream = "end_of_llm_stream" in meta_info and meta_info["end_of_llm_stream"]
            self.meta_info = copy.deepcopy(meta_info)
            meta_info["text"] = text
            
            self.update_streaming_context(meta_info)
            
            self.sender_task = asyncio.create_task(self.sender(text, meta_info.get("sequence_id"), end_of_llm_stream))
            self.text_queue.append(meta_info)
        else:
            self.internal_queue.put_nowait(message)
    
    def synthesize(self, text):
        pass

    def get_synthesized_characters(self):
        return 0

    async def monitor_connection(self):
        """Periodically check if the connection is still alive and re-establish if needed"""
        while True:
            if (self.websocket_holder["websocket"] is None or 
                (hasattr(self.websocket_holder["websocket"], 'state') and 
                 hasattr(self.websocket_holder["websocket"].state, 'name') and
                 self.websocket_holder["websocket"].state.name == 'CLOSED')):
                logger.info("Re-establishing connection...")
                self.websocket_holder["websocket"] = await self.establish_connection()
            await asyncio.sleep(1)

    async def cleanup(self):
        """Clean up websocket connections and tasks"""
        self.conversation_ended = True
        logger.info("cleaning synthesizer tasks")
        if self.sender_task:
            try:
                self.sender_task.cancel()
                await self.sender_task
            except asyncio.CancelledError:
                logger.info("Sender task was successfully cancelled during WebSocket cleanup.")

        websocket = self.websocket_holder.get("websocket")
        if websocket and hasattr(websocket, 'close') and callable(getattr(websocket, 'close', None)):
            try:
                await websocket.close()
            except Exception as e:
                logger.error(f"Error closing websocket: {e}")
        self.websocket_holder["websocket"] = None
        logger.info("WebSocket connection closed.")

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

    def set_first_chunk_metadata(self, meta_info):
        """Set metadata for first chunk"""
        if not self.first_chunk_generated:
            meta_info["is_first_chunk"] = True
            self.first_chunk_generated = True
        else:
            meta_info["is_first_chunk"] = False

    def set_end_of_stream_metadata(self, meta_info):
        """Set metadata for end of stream"""
        if "end_of_llm_stream" in meta_info and meta_info["end_of_llm_stream"]:
            meta_info["end_of_synthesizer_stream"] = True
            self.first_chunk_generated = False

    def create_audio_packet(self, audio, meta_info, text_synthesized=""):
        """Create standardized audio packet with metadata"""
        meta_info["text_synthesized"] = text_synthesized
        meta_info["mark_id"] = str(uuid.uuid4())
        return create_ws_data_packet(audio, meta_info)

    def update_streaming_context(self, meta_info):
        """Override in subclasses for provider-specific context management"""
        pass

    async def send_http_request(self, url, headers, payload=None, method="POST"):
        """Generic HTTP request method for synthesizer APIs"""
        async with aiohttp.ClientSession() as session:
            if method.upper() == "POST":
                if payload is not None:
                    async with session.post(url, headers=headers, json=payload) as response:
                        if response.status == 200:
                            return await response.read()
                        else:
                            logger.error(f"Error: {response.status} - {await response.text()}")
                            return None
                else:
                    logger.info("Payload was null")
                    return None

    async def establish_connection(self):
        """Override in subclasses for provider-specific connection logic"""
        pass

    async def sender(self, text, sequence_id, end_of_llm_stream=False):
        """Override in subclasses for provider-specific sender logic"""
        pass
