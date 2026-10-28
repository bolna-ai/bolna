from typing import Callable, Optional

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


class FallbackLLM:
    """The agent's LLM followed by its fallbacks.

    A turn that fails before yielding anything is re-run on the next entry, which then serves the rest of the
    call. Once a turn has yielded output it is never re-run: the caller may already be hearing it.
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
        while True:
            delivered = False
            try:
                async for chunk in self._active.generate_stream(messages, *args, **kwargs):
                    delivered = True
                    yield chunk
                return
            except Exception as error:
                if delivered or not self._switch(error, kwargs.get("meta_info")):
                    raise

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
            if self._pending:
                llm.fail_fast()
            self._retired.append(self._active)
            self._active = llm
            if self._on_switch is not None:
                try:
                    self._on_switch(error, entry, meta_info)
                except Exception as e:
                    logger.error(f"LLM fallback switch callback failed: {e}")
            return True
        return False

    async def close(self):
        for llm in (*self._retired, self._active):
            try:
                await llm.close()
            except Exception as e:
                logger.error(f"Error closing LLM {getattr(llm, 'model', None)}: {e}")
