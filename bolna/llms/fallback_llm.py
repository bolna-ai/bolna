import asyncio
import random
import time
from typing import Callable, Optional

import httpx
from google.genai import errors as genai_errors
from openai import APIConnectionError

from bolna.helpers.fallbacks import next_built
from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)

# A failure slower than this has already cost the caller a pause, so the turn moves on instead of trying again.
RETRY_WITHIN_S = 1.5

# Settings one model family reads, carried from the agent's LLM config onto its own model only.
REASONING_SETTINGS = ("reasoning_effort", "verbosity", "reasoning_summary", "thinking_budget", "extra_body")

# Bound to one model or backend, so a fallback never inherits them from the agent's own LLM.
MODEL_BOUND_LLM_SETTINGS = frozenset(
    {
        "llm_key",
        "base_url",
        "api_version",
        "overflow_llm",
        "service_tier",
        "aws_region_name",
        "use_responses_api",
        "compact_threshold",
        *REASONING_SETTINGS,
    }
)

# Statuses that say the provider cannot serve, whatever was asked: key, access, model, timeout or capacity.
PROVIDER_FAULT_STATUSES = frozenset({401, 403, 404, 408, 429})


def fallback_llm_arguments(llm_config, kwargs, entry, credentials):
    """The config and kwargs a fallback entry is built with: the agent's LLM settings on the entry's model and backend."""
    config = {key: value for key, value in llm_config.items() if key not in MODEL_BOUND_LLM_SETTINGS}
    config.update({key: value for key, value in entry.items() if value is not None})
    entry_kwargs = {key: value for key, value in kwargs.items() if key not in MODEL_BOUND_LLM_SETTINGS}
    entry_kwargs.update({key: value for key, value in (credentials or {}).items() if value is not None})
    return config, entry_kwargs


def status_of(error):
    """The HTTP status an SDK error carries, if any."""
    if isinstance(error, genai_errors.APIError):
        return error.code
    return getattr(error, "status_code", None)


def is_transient(error):
    """A fault the same model may not repeat: no connection, or a server-side error."""
    status = status_of(error)
    if status is not None:
        return status >= 500
    return isinstance(error, (APIConnectionError, httpx.TransportError, ConnectionError, asyncio.TimeoutError))


def is_provider_fault(error):
    """A failure of the provider rather than of the request, so it counts against the provider's health."""
    status = status_of(error)
    if status is not None:
        return status >= 500 or status in PROVIDER_FAULT_STATUSES
    return is_transient(error)


class FallbackLLM:
    """The agent's LLM followed by its fallbacks.

    A turn that fails before yielding anything is re-run on the next entry, which then serves the rest of the
    call; a fault that may pass gets one more try on the same model first. Once a turn has yielded output it is
    never re-run: the caller may already be hearing it.
    Everything else is answered by whichever LLM is serving.
    """

    def __init__(
        self,
        primary,
        entries,
        on_switch: Optional[Callable] = None,
        on_retry: Optional[Callable] = None,
        passed_over=None,
    ):
        """passed_over is the agent's own LLM when the call starts past it; it is closed with the rest."""
        self._active = primary
        self._pending = list(entries)
        self._retired = [passed_over] if passed_over is not None else []
        self._on_switch = on_switch
        self._on_retry = on_retry
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
            started = time.monotonic()
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
                    retried = False
                    continue
                failed_fast = time.monotonic() - started <= RETRY_WITHIN_S
                # A passing fault that came back at once gets one more try on the same model.
                if self._pending and not retried and failed_fast and is_transient(error):
                    retried = True
                    logger.warning(f"LLM failed before any output, retrying once: {error!r}")
                    self._notify(self._on_retry, error)
                    # The failed request may have used up what a chained request relies on.
                    llm.invalidate_response_chain()
                    await asyncio.sleep(random.uniform(0.05, 0.15))
                    continue
                if not self._switch(error, kwargs.get("meta_info")):
                    raise
                retried = False

    def _switch(self, error, meta_info):
        """Move to the next entry that builds; False when none is left."""
        built = next_built(self._pending)
        if built is None:
            return False
        entry, llm = built
        logger.warning(f"LLM failed before any output, moving to {entry.provider}/{entry.model}: {error!r}")
        self._move_to(entry, llm, error, meta_info)
        return True

    def _move_to(self, entry, llm, error, meta_info):
        if self._pending:
            llm.fail_fast()
        self._retired.append(self._active)
        self._active = llm
        self._notify(self._on_switch, error, entry, meta_info)

    def _notify(self, callback, *args):
        if callback is None:
            return
        try:
            callback(*args)
        except Exception as e:
            logger.error(f"LLM fallback callback failed: {e}")

    async def close(self):
        # The agent's own model can be both retired and, as the last entry, active again.
        for llm in {id(llm): llm for llm in (*self._retired, self._active)}.values():
            try:
                await llm.close()
            except Exception as e:
                logger.error(f"Error closing LLM {getattr(llm, 'model', None)}: {e}")
