from typing import Awaitable, Callable, Optional

from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)

# async def on_language(lang: str, confidence: Optional[float]) -> None
# confidence is None when the provider does not return a score (e.g. ElevenLabs Scribe streaming)
OnLanguageCallback = Callable[[str, Optional[float]], Awaitable[None]]


class LIDBackend:
    """Base class for all LID backends."""

    def __init__(self, on_language, config):
        self.on_language = on_language
        self.config = config

    async def start(self):
        raise NotImplementedError

    def feed(self, audio_bytes):
        raise NotImplementedError

    async def stop(self):
        raise NotImplementedError
