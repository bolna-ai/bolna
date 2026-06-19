"""Helpers for safely scheduling fire-and-forget asyncio tasks.

``asyncio.create_task`` only stores a *weak* reference to the task it returns.
If the caller discards that reference (a fire-and-forget call such as
``asyncio.create_task(do_something())``), the task can be garbage-collected
before it finishes. That surfaces as the warning

    Task was destroyed but it is pending!

and the scheduled work is silently dropped. See the asyncio docs:
https://docs.python.org/3/library/asyncio-task.html#asyncio.create_task

``create_tracked_task`` keeps a strong reference to every task it schedules
until the task completes, then drops it via a done callback. It is a drop-in
replacement for ``asyncio.create_task`` at fire-and-forget call sites.
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any

from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)

# Strong references to in-flight fire-and-forget tasks. The event loop only
# holds a weak reference, so without this set the tasks can be collected
# mid-execution. Tasks remove themselves once done (see ``_on_task_done``).
_background_tasks: set[asyncio.Task] = set()


def _on_task_done(task: asyncio.Task) -> None:
    """Drop the strong reference and surface unexpected failures.

    Calling ``task.exception()`` here also retrieves the result, which prevents
    asyncio's "Task exception was never retrieved" noise for fire-and-forget
    tasks that raise.
    """
    _background_tasks.discard(task)
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error(f"Background task {task.get_name()!r} failed: {exc!r}")


def create_tracked_task(coro: Coroutine[Any, Any, Any], *, name: str | None = None) -> asyncio.Task:
    """Schedule ``coro`` and keep a strong reference until it finishes.

    Use this instead of ``asyncio.create_task`` whenever the returned task would
    otherwise be discarded, so the task cannot be garbage-collected while still
    pending. Must be called from within a running event loop.

    Args:
        coro: The coroutine to run.
        name: Optional task name, surfaced in logs and tracebacks.

    Returns:
        The scheduled ``asyncio.Task`` (callers may ignore it).
    """
    task = asyncio.create_task(coro, name=name)
    _background_tasks.add(task)
    task.add_done_callback(_on_task_done)
    return task


def pending_task_count() -> int:
    """Return the number of tracked tasks still in flight (handy in tests)."""
    return len(_background_tasks)
