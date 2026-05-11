from abc import ABC, abstractmethod
from typing import AsyncGenerator, List, Optional


class BaseS2SProvider(ABC):
    """Provider-agnostic interface for speech-to-speech models.

    TaskManager interacts only through these methods, so a new provider
    (Gemini Live, etc.) drops in by implementing this contract.
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
        self.first_audio_latencies: list = []

    @abstractmethod
    async def connect(self) -> None: ...

    @abstractmethod
    async def send_audio(self, pcm_24k_bytes: bytes) -> None: ...

    @abstractmethod
    async def receive_events(self) -> AsyncGenerator: ...

    @abstractmethod
    async def send_function_result(self, call_id: str, result: str) -> None: ...

    @abstractmethod
    async def commit_function_results(self) -> None: ...

    @abstractmethod
    async def trigger_response(self, instructions: Optional[str] = None) -> None: ...

    @abstractmethod
    async def disconnect(self) -> None: ...
