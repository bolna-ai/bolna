from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass
class FallbackEntry:
    """A provider to move to when the one in use fails, built only once the call needs it."""

    provider: str
    model: Optional[str]
    build: Callable[[], Any]
    # The entry's saved config, for what the call reads about the provider serving it.
    config: Optional[dict] = None


class ProviderUnhealthy(Exception):
    """A provider passed over at call start because the caller marked it as failing other calls."""


PROVIDER_UNHEALTHY = "provider_unhealthy"
