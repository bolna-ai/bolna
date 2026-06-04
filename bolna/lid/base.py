from typing import Awaitable, Callable, Optional

from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)

# async def on_language(lang: str, confidence: Optional[float]) -> None
# confidence is None when the provider does not return a score (e.g. ElevenLabs Scribe streaming)
OnLanguageCallback = Callable[[str, Optional[float]], Awaitable[None]]

# async def on_turn(transcript: str, detected_lang: Optional[str]) -> None
# Fired once per finalized user turn (end-of-speech) carrying the unbiased transcript
# plus the provider's per-turn detected language code (may be None).
OnTurnCallback = Callable[[str, Optional[str]], Awaitable[None]]


class LIDBackend:
    """Base class for all LID backends."""

    def __init__(self, on_language, config, on_turn=None):
        self.on_language = on_language
        self.on_turn = on_turn
        self.config = config

    async def start(self):
        raise NotImplementedError

    def feed(self, audio_bytes):
        raise NotImplementedError

    async def stop(self):
        raise NotImplementedError
