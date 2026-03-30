from abc import ABC, abstractmethod
from typing import AsyncGenerator, List, Optional


class BaseS2SProvider(ABC):
    """Base class for speech-to-speech providers.

    Any S2S provider (OpenAI Realtime, Gemini Live, etc.) implements this
    interface.  TaskManager only interacts through these methods, keeping
    the provider details isolated.
    """

    def __init__(
        self,
        *,
        system_prompt: str,
        voice: str,
        model: str,
        api_key: str,
        tools: Optional[List[dict]] = None,
        **kwargs,
    ):
        self.system_prompt = system_prompt
        self.voice = voice
        self.model = model
        self.api_key = api_key
        self.tools = tools or []
        self.connection_time: Optional[float] = None
        self.turn_latencies: list = []

    @abstractmethod
    async def connect(self) -> None:
        """Open the WebSocket / session to the provider."""
        ...

    @abstractmethod
    async def send_audio(self, pcm_24k_bytes: bytes) -> None:
        """Send PCM-16 24 kHz mono audio to the provider."""
        ...

    @abstractmethod
    async def receive_events(self) -> AsyncGenerator:
        """Yield provider-agnostic S2S events (AudioDelta, TranscriptDelta, etc.)."""
        ...  # pragma: no cover
        yield  # make it a generator  # noqa: E701

    @abstractmethod
    async def send_function_result(self, call_id: str, result: str) -> None:
        """Return the result of a function call back to the provider."""
        ...

    @abstractmethod
    async def trigger_response(self, instructions: Optional[str] = None) -> None:
        """Ask the provider to generate a response (e.g. welcome message)."""
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Cleanly close the connection."""
        ...
