import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from bolna.enums import FallbackComponent, FallbackReason
from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)


@dataclass
class FallbackEntry:
    """A provider to move to when the one in use fails, built only once the call needs it."""

    provider: str
    model: Optional[str]
    build: Callable[[], Any]
    # The entry's saved config, for what the call reads about the provider serving it.
    config: Optional[dict] = None

    @property
    def key(self):
        return self.provider, self.model


class ProviderUnhealthy(Exception):
    """A provider passed over at call start because the caller marked it as failing other calls."""


def next_built(pending):
    """Pop entries off `pending` until one builds: (entry, instance), or None when none is left."""
    while pending:
        entry = pending.pop(0)
        try:
            return entry, entry.build()
        except Exception as e:
            logger.error(f"Fallback {entry.provider}/{entry.model} could not be built: {e}")
    return None


@dataclass
class StartChoice:
    """The entry a component starts on, its instance, and the fallbacks behind it."""

    entry: FallbackEntry
    instance: Any
    fallbacks: list


def start_on_healthy(own, entries, unhealthy):
    """Start on the agent's own provider unless it is marked unhealthy; then on the first fallback not marked so that
    builds, with the agent's own provider moved to the end of the fallbacks."""
    if own.key in unhealthy:
        built = next_built([entry for entry in entries if entry.key not in unhealthy])
        if built is not None:
            entry, instance = built
            return StartChoice(entry, instance, [*(other for other in entries if other is not entry), own])
    return StartChoice(own, own.build(), list(entries))


class FallbackRecorder:
    """The call's fallback events, each credited at the end of the call with what the provider it moved to served."""

    def __init__(self, started_at_ms, llm_turns):
        """llm_turns returns the call's LLM turn rows; it is read when needed, as the rows list can be replaced."""
        self.events: list[dict] = []
        self.started_at_ms = started_at_ms
        self.llm_turns = llm_turns
        # Events whose replacement reports its own usage, credited at the end of the call.
        self.replacements: list[tuple[dict, object]] = []
        # LLM events with the index of the first turn row the model taking over served.
        self.llm_starts: list[tuple[dict, int]] = []

    def record(self, component, failed, served, reason, error, meta_info=None, replacement=None):
        event = {
            # Plain strings: the event is sent with the call's data.
            "component": FallbackComponent(component).value,
            "from": failed,
            "to": served,
            "reason": getattr(reason, "value", reason),
            "error": error,
            "sequence_id": (meta_info or {}).get("sequence_id"),
            "turn_id": (meta_info or {}).get("turn_id"),
            "ts_ms": round(time.time() * 1000 - self.started_at_ms, 2),
        }
        self.events.append(event)
        if replacement is not None:
            self.replacements.append((event, replacement))
        elif component == FallbackComponent.LLM:
            turns = self.llm_turns()
            # A row opened for the turn that failed is finished by the model taking over, so it counts as its own.
            reruns_open_row = bool(turns) and turns[-1].get("sequence_id") == event["sequence_id"]
            self.llm_starts.append((event, len(turns) - reruns_open_row))

    def with_usage(self):
        """Each event with what the provider it moved to served for the rest of the call."""
        for event, replacement in self.replacements:
            if event["component"] == FallbackComponent.TRANSCRIBER:
                event["served_usage"] = {"audio_s": round(replacement.billed_audio_s(), 4)}
            else:
                event["served_usage"] = {"characters": replacement.get_synthesized_characters()}
        turns = self.llm_turns()
        for index, (event, start) in enumerate(self.llm_starts):
            # A switch re-runs the failed turn, so the entry serves every turn logged from then to the next switch.
            later = self.llm_starts[index + 1 :]
            usage = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0, "turns": 0}
            for turn in turns[start : later[0][1] if later else None]:
                for key in ("input_tokens", "output_tokens", "cached_tokens"):
                    usage[key] += turn.get(key) or 0
                usage["turns"] += 1
            event["served_usage"] = usage
        # A language that began on a fallback but never carried the call did not fall back.
        return [
            event
            for event in self.events
            if event["reason"] != FallbackReason.PROVIDER_UNHEALTHY or any((event.get("served_usage") or {}).values())
        ]
