"""handle_language_switch cancels its own speculative follow-up, never a later turn's.

__run_language_switch stores the task on the shared slot self._spec_followup_task, then generates
outside language_switch_lock, so a second per-turn decision can overwrite that slot mid-flight.
Each handler therefore claims its task into a local the instant its decision unwinds and cancels
that local in finally, including on the exception path.
"""

import asyncio
from unittest.mock import MagicMock


from bolna.agent_manager.task_manager import TaskManager


def _make_tm():
    tm = MagicMock()
    tm.language_switch_lock = asyncio.Lock()
    tm._spec_followup_task = None
    return tm


def _handle(tm):
    return TaskManager.handle_language_switch.__get__(tm, TaskManager)


async def _never():
    await asyncio.sleep(3600)


async def test_concurrent_handlers_each_cancel_own_spec():
    """Two overlapping handlers must each cancel their OWN spec task.

    Reproduces the race: handler A spawns specA, releases the lock, and parks in
    follow-up generation; handler B then acquires the lock and overwrites the slot
    with specB. With the old shared-slot read, A cancelled specB and leaked specA.
    """
    tm = _make_tm()
    specA = asyncio.create_task(_never())
    specB = asyncio.create_task(_never())
    b_overwrote_slot = asyncio.Event()
    call_count = {"n": 0}

    async def fake_run(active_transcript, meta_info, spawn_language):
        call_count["n"] += 1
        if call_count["n"] == 1:  # handler A — spawns spec, returns a follow-up to generate
            tm._spec_followup_task = specA
            return ("A-followup",)
        # handler B — overwrites the slot, no follow-up so it unwinds promptly
        tm._spec_followup_task = specB
        b_overwrote_slot.set()
        return None

    async def fake_generate(*args):
        # A's generation runs outside the lock — hold here until B has overwritten the
        # shared slot AND fully unwound, reproducing the exact race window.
        await b_overwrote_slot.wait()
        await asyncio.sleep(0)

    tm._TaskManager__run_language_switch = fake_run
    tm._TaskManager__generate_switch_followup = fake_generate

    handle = _handle(tm)
    a = asyncio.create_task(handle())
    await asyncio.sleep(0)  # let A acquire+release the lock and park in generation
    b = asyncio.create_task(handle())
    await asyncio.gather(a, b)
    await asyncio.gather(specA, specB, return_exceptions=True)  # let cancellations settle

    assert specA.cancelled()  # old bug: leaked (B had nulled the slot before A's finally)
    assert specB.cancelled()  # cancelled by B, not stolen by A


async def test_single_handler_cancels_unconsumed_spec():
    tm = _make_tm()
    spec = asyncio.create_task(_never())

    async def fake_run(active_transcript, meta_info, spawn_language):
        tm._spec_followup_task = spec
        return None  # nothing consumed the spec

    tm._TaskManager__run_language_switch = fake_run
    await _handle(tm)()

    await asyncio.gather(spec, return_exceptions=True)  # let the cancellation settle
    assert spec.cancelled()
    assert tm._spec_followup_task is None


async def test_committed_spec_left_untouched():
    """When __run_language_switch consumes the spec (commit path), it nulls the slot;
    the handler must not cancel anything."""
    tm = _make_tm()
    spec = asyncio.create_task(_never())

    async def fake_run(active_transcript, meta_info, spawn_language):
        tm._spec_followup_task = spec
        tm._spec_followup_task = None  # commit path consumed + cleared it
        return ("committed-followup",)

    async def fake_generate(*args):
        return None

    tm._TaskManager__run_language_switch = fake_run
    tm._TaskManager__generate_switch_followup = fake_generate
    await _handle(tm)()

    assert not spec.cancelled()  # handler didn't touch the consumed task
    spec.cancel()
    await asyncio.gather(spec, return_exceptions=True)


async def test_spec_cancelled_when_decision_raises_after_spawn():
    """__run_language_switch can raise AFTER spawning the spec (decide/switch/gate);
    the inner finally still claims it, so the outer finally cancels it — and the
    handler swallows the error rather than propagating."""
    tm = _make_tm()
    spec = asyncio.create_task(_never())

    async def fake_run(active_transcript, meta_info, spawn_language):
        tm._spec_followup_task = spec
        raise RuntimeError("decide blew up after spawning speculation")

    tm._TaskManager__run_language_switch = fake_run
    await _handle(tm)()  # must not raise

    await asyncio.gather(spec, return_exceptions=True)  # let the cancellation settle
    assert spec.cancelled()
    assert tm._spec_followup_task is None
