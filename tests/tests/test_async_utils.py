"""Unit tests for bolna.helpers.async_utils.create_tracked_task.

These pin the fix for "Task was destroyed but it is pending!" (issue #462):
fire-and-forget tasks must keep a strong reference until they finish so the
garbage collector can't drop them mid-execution.
"""

import asyncio
import gc
import logging

import pytest

from bolna.helpers.async_utils import create_tracked_task, pending_task_count


@pytest.mark.asyncio
async def test_runs_coroutine_and_returns_task():
    result = {}

    async def work():
        result["ran"] = True
        return 42

    task = create_tracked_task(work())
    assert isinstance(task, asyncio.Task)
    assert await task == 42
    assert result["ran"] is True


@pytest.mark.asyncio
async def test_tracked_while_pending_then_dropped_when_done():
    started = asyncio.Event()
    release = asyncio.Event()

    async def work():
        started.set()
        await release.wait()

    before = pending_task_count()
    task = create_tracked_task(work())

    await started.wait()
    # While the task is pending, the helper holds a strong reference to it.
    assert pending_task_count() == before + 1
    assert task in _tracked_set()

    release.set()
    await task
    # The done callback runs on the next loop iteration; let it fire.
    await asyncio.sleep(0)
    assert pending_task_count() == before
    assert task not in _tracked_set()


@pytest.mark.asyncio
async def test_survives_garbage_collection_while_pending():
    """The core regression: a fire-and-forget task whose return value is
    discarded must still complete even if a GC pass runs while it is pending."""
    done = asyncio.Event()

    async def work():
        await asyncio.sleep(0.02)
        done.set()

    # Discard the returned task entirely — the only strong reference now lives
    # inside the helper's set, exactly the fire-and-forget shape from issue #462.
    create_tracked_task(work())

    # A GC pass here would collect an untracked task and raise
    # "Task was destroyed but it is pending!".
    gc.collect()

    await asyncio.wait_for(done.wait(), timeout=1.0)
    assert done.is_set()


@pytest.mark.asyncio
async def test_task_name_is_propagated():
    async def work():
        return None

    task = create_tracked_task(work(), name="my-task")
    assert task.get_name() == "my-task"
    await task


@pytest.mark.asyncio
async def test_exception_is_logged_and_reference_dropped(caplog):
    async def boom():
        raise ValueError("kaboom")

    before = pending_task_count()
    with caplog.at_level(logging.ERROR, logger="bolna.helpers.async_utils"):
        task = create_tracked_task(boom(), name="boom-task")
        with pytest.raises(ValueError, match="kaboom"):
            await task
        # Let the done callback run so it logs and drops the reference.
        await asyncio.sleep(0)

    assert pending_task_count() == before
    assert any("boom-task" in rec.message and "kaboom" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_cancelled_task_is_not_logged_as_error(caplog):
    async def work():
        await asyncio.sleep(10)

    before = pending_task_count()
    with caplog.at_level(logging.ERROR, logger="bolna.helpers.async_utils"):
        task = create_tracked_task(work(), name="cancel-me")
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0)

    assert pending_task_count() == before
    assert not any("cancel-me" in rec.message for rec in caplog.records)


def _tracked_set():
    """Access the helper's internal set for white-box assertions."""
    from bolna.helpers import async_utils

    return async_utils._background_tasks
