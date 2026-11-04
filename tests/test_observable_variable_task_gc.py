"""ObservableVariable must keep a strong reference to async-observer tasks.

asyncio only holds a weak reference to a bare task returned by loop.create_task, so a
fire-and-forget observer coroutine can be garbage-collected before it runs to completion.
The observers registered in TaskManager (agent hangup, end-of-call teardown, web-call init)
are exactly this pattern, so a dropped task means a dropped lifecycle event.
"""

import asyncio
import gc

from bolna.helpers.observable_variable import ObservableVariable


async def test_async_observer_task_is_held_until_it_finishes():
    ov = ObservableVariable(0)
    started = asyncio.Event()
    release = asyncio.Event()

    async def observer(_value):
        started.set()
        await release.wait()

    ov.add_observer(observer)
    ov.value = 1

    await asyncio.wait_for(started.wait(), timeout=1)
    # While the observer is still running, its task must be referenced.
    assert len(ov._pending_tasks) == 1

    release.set()
    for _ in range(100):
        if not ov._pending_tasks:
            break
        await asyncio.sleep(0)
    # Once it completes, the reference is released so nothing leaks.
    assert len(ov._pending_tasks) == 0


async def test_async_observer_runs_to_completion_even_after_gc():
    ov = ObservableVariable(0)
    ran = []
    done = asyncio.Event()

    async def observer(value):
        await asyncio.sleep(0.02)
        ran.append(value)
        done.set()

    ov.add_observer(observer)
    ov.value = 5
    # A collection cycle here would reap an unreferenced task; the strong ref must survive it.
    gc.collect()

    await asyncio.wait_for(done.wait(), timeout=1)
    assert ran == [5]


async def test_sync_observer_is_still_called_directly():
    ov = ObservableVariable(0)
    seen = []
    ov.add_observer(lambda value: seen.append(value))

    ov.value = 42

    assert seen == [42]
    assert ov._pending_tasks == set()


async def test_no_notification_when_value_is_unchanged():
    ov = ObservableVariable(7)
    seen = []
    ov.add_observer(lambda value: seen.append(value))

    ov.value = 7  # unchanged -> observers must not fire

    assert seen == []
