import asyncio
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)


class BaseVoiceToVoice(ABC):
    """Abstract base class for voice-to-voice providers."""

    def __init__(
        self,
        api_key: str,
        voice: str,
        instructions: str,
        task_manager_instance=None,
        **kwargs
    ):
        self.api_key = api_key
        self.voice = voice
        self.instructions = instructions
        self.task_manager = task_manager_instance
        self.kwargs = kwargs

        # Connection state
        self.ws = None
        self.connected = False
        self.session_id = None

        # Queues for audio flow
        self.audio_input_queue = asyncio.Queue()
        self.audio_output_queue = asyncio.Queue()

        # Conversation tracking
        self.conversation_history = []
        self.function_calls = []

        # Metrics
        self.connection_time = None
        self.turn_latencies = []
        self.audio_input_duration = 0
        self.audio_output_duration = 0

    @abstractmethod
    async def connect(self) -> bool:
        """Establish WebSocket connection to v2v provider."""
        pass

    @abstractmethod
    async def configure_session(self, config: Dict[str, Any]):
        """Configure session parameters (voice, instructions, tools, etc)."""
        pass

    @abstractmethod
    async def send_audio(self, audio_chunk: bytes, meta_info: Dict[str, Any]):
        """Send audio chunk to v2v provider."""
        pass

    @abstractmethod
    async def receive_audio(self) -> Optional[Dict[str, Any]]:
        """Receive audio chunk from v2v provider.

        Returns:
            Dict with 'audio', 'transcript', 'meta_info', or None
        """
        pass

    @abstractmethod
    async def handle_interruption(self):
        """Handle user interrupting assistant."""
        pass

    @abstractmethod
    async def close(self):
        """Close connection and cleanup."""
        pass

    async def get_conversation_transcript(self) -> str:
        """Get full conversation transcript."""
        transcript_parts = []
        for turn in self.conversation_history:
            role = turn.get('role', 'unknown')
            content = turn.get('content', '')
            transcript_parts.append(f"{role}: {content}")
        return "\n".join(transcript_parts)

    async def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return {
            "connection_time": self.connection_time,
            "turn_latencies": self.turn_latencies,
            "audio_input_duration": self.audio_input_duration,
            "audio_output_duration": self.audio_output_duration,
        }

