from typing import Callable, Optional

from bolna.helpers.fallbacks import ProviderUnhealthy
from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)

# Bound to one model or backend, so a fallback never inherits them from the agent's own LLM.
MODEL_BOUND_LLM_SETTINGS = frozenset(
    {
        "llm_key",
        "base_url",
        "api_version",
        "overflow_llm",
        "service_tier",
        "aws_region_name",
        "reasoning_effort",
        "reasoning_summary",
        "verbosity",
        "thinking_budget",
        "extra_body",
        "use_responses_api",
        "compact_threshold",
    }
)


def fallback_llm_arguments(llm_config, kwargs, entry, credentials):
    """The config and kwargs a fallback entry is built with: the agent's LLM settings on the entry's model and backend."""
    config = {key: value for key, value in llm_config.items() if key not in MODEL_BOUND_LLM_SETTINGS}
    config.update({key: value for key, value in entry.items() if value is not None})
    entry_kwargs = {key: value for key, value in kwargs.items() if key not in MODEL_BOUND_LLM_SETTINGS}
    entry_kwargs.update({key: value for key, value in (credentials or {}).items() if value is not None})
    return config, entry_kwargs


def is_transient(error):
    """A fault the same model may not repeat: a dropped connection, a timeout or a server-side error."""
    status = getattr(error, "status_code", None)
    return status is None or status >= 500


class FallbackLLM:
    """The agent's LLM followed by its fallbacks.

    A turn that fails before yielding anything is re-run on the next entry, which then serves the rest of the
    call; a fault that may pass gets one more try on the same model first. Once a turn has yielded output it is
    never re-run: the caller may already be hearing it.
    Everything else is answered by whichever LLM is serving.
    """

    def __init__(self, primary, entries, on_switch: Optional[Callable] = None):
        self._active = primary
        self._pending = list(entries)
        self._retired = []
        self._on_switch = on_switch
        if self._pending:
            primary.fail_fast()

    def __getattr__(self, name):
        active = self.__dict__.get("_active")
        if active is None:
            raise AttributeError(name)
        return getattr(active, name)

    @property
    def active(self):
        return self._active

    async def generate_stream(self, messages, *args, **kwargs):
        retried = False
        while True:
            llm = self._active
            delivered = False
            try:
                async for chunk in llm.generate_stream(messages, *args, **kwargs):
                    delivered = True
                    yield chunk
                return
            except Exception as error:
                if delivered:
                    raise
                # Another stream running alongside this one already moved the call on; re-run there.
                if self._active is not llm:
                    continue
                # A passing fault gets one more try on the same model before the call moves off it for good.
                if self._pending and not retried and is_transient(error):
                    retried = True
                    logger.warning(f"LLM failed before any output, retrying once: {error!r}")
                    continue
                if not self._switch(error, kwargs.get("meta_info")):
                    raise
                retried = False

    def start_on_healthy(self, is_unhealthy, primary_entry):
        """Begin the call on the first entry not marked unhealthy; the agent's own model stays the last to try."""
        for entry in self._pending:
            if is_unhealthy(entry):
                continue
            try:
                llm = entry.build()
            except Exception as e:
                logger.error(f"LLM fallback {entry.provider}/{entry.model} could not be built: {e}")
                continue
            self._pending = [*(e for e in self._pending if e is not entry), primary_entry]
            self._move_to(entry, llm, ProviderUnhealthy(), None)
            return

    def _switch(self, error, meta_info):
        """Move to the next entry that builds; False when none is left."""
        while self._pending:
            entry = self._pending.pop(0)
            try:
                llm = entry.build()
            except Exception as e:
                logger.error(f"LLM fallback {entry.provider}/{entry.model} could not be built: {e}")
                continue
            logger.warning(f"LLM failed before any output, moving to {entry.provider}/{entry.model}: {error!r}")
            self._move_to(entry, llm, error, meta_info)
            return True
        return False

    def _move_to(self, entry, llm, error, meta_info):
        if self._pending:
            llm.fail_fast()
        self._retired.append(self._active)
        self._active = llm
        if self._on_switch is not None:
            try:
                self._on_switch(error, entry, meta_info)
            except Exception as e:
                logger.error(f"LLM fallback switch callback failed: {e}")

    async def close(self):
        # The agent's own model can be both retired and, as the last entry, active again.
        for llm in {id(llm): llm for llm in (*self._retired, self._active)}.values():
            try:
                await llm.close()
            except Exception as e:
                logger.error(f"Error closing LLM {getattr(llm, 'model', None)}: {e}")
