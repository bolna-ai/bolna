from abc import ABC, abstractmethod
from typing import AsyncGenerator, List, Optional


class BaseS2SProvider(ABC):
    """Provider-agnostic interface for speech-to-speech models.

    TaskManager interacts only through this contract, so a new provider drops in by
    subclassing and declaring its audio rates. Rates are declared per provider rather
    than assumed: OpenAI Realtime takes 24kHz input, Gemini Live takes 16kHz, and both
    emit 24kHz. Callers resample against these attributes instead of hardcoding.
    """

    input_sample_rate: int
    output_sample_rate: int = 24000

    # DTMF is telephony-only and not offered by every provider.
    supports_dtmf: bool = False

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
        self.usage_total: dict = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cached_tokens": 0,
            "input_audio_tokens": 0,
            "output_audio_tokens": 0,
            "input_text_tokens": 0,
            "output_text_tokens": 0,
        }

    def _accumulate_usage(self, usage: dict) -> None:
        for key, value in usage.items():
            if key in self.usage_total:
                self.usage_total[key] += value or 0

    @abstractmethod
    async def connect(self) -> None:
        """Open the session and block until the provider accepts the config."""

    @abstractmethod
    async def send_audio(self, pcm_bytes: bytes) -> None:
        """Send PCM-16 mono audio at input_sample_rate."""

    @abstractmethod
    async def receive_events(self) -> AsyncGenerator:
        """Yield provider-agnostic events until the session ends."""

    @abstractmethod
    async def send_function_result(self, call_id: str, name: str, result: str) -> None:
        """Queue one tool result for the model."""

    @abstractmethod
    async def commit_function_results(self) -> None:
        """Flush queued tool results and let the model continue."""

    @abstractmethod
    async def trigger_response(self, instructions: Optional[str] = None) -> None:
        """Ask the model to speak without new user audio, used for the welcome message."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Close the session."""

    async def send_dtmf(self, digits: str) -> None:
        """Forward telephony keypad digits to the model."""
        raise NotImplementedError(f"{type(self).__name__} does not support DTMF")
