"""Regression: the LID idle-flush watcher must never busy-spin the event loop.

Jul 2026 transcript-missing incident: on a multilingual (llm_language_switch) call,
when has_transfer / _end_call_in_progress was set, __run_language_switch abandoned the
decision PRE-drain, so the aged detector buffer stayed >= threshold and __lid_idle_watcher
re-fired it every iteration with no awaiting yield — a synchronous spin that pegged and
blocked the pod's event loop, starving co-tenant calls of media/TTS (silence, empty
transcript). Two guards fix it: (1) the loop-top skip now covers the full ignore-input
condition, not just hangup; (2) a spin-guard forces a yield whenever a fire leaves the
buffer undrained.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from bolna.agent_manager.task_manager import TaskManager
from bolna.transcriber.transcriber_pool import TranscriberPool


def _tm(*, has_transfer=False, end_call=False, hangup=False, buffer_age=5.0):
    """Minimal TaskManager whose idle watcher can run in isolation.

    buffer_age is a constant so the buffer looks perpetually aged-but-undrained — the exact
    state that produced the spin. handle_language_switch is a no-op mock (never drains).
    """
    tm = MagicMock()
    tm.conversation_ended = False
    tm.hangup_triggered = hangup
    tm._end_call_in_progress = end_call
    tm.has_transfer = has_transfer
    tm._pending_switch_turn = None
    tm.language = "hi"
    tm.handle_language_switch = AsyncMock()  # no-op: never drains the buffer

    pool = MagicMock(spec=TranscriberPool)
    pool.lid_buffer_age.return_value = buffer_age
    pool.lid_buffer_language.return_value = "hi"  # == active → mismatch False, threshold=idle_flush
    pool.lid_buffer_language_streak.return_value = 0
    pool.lid_buffer_event.return_value = None
    tm.tools = {"transcriber": pool}

    # Real methods under test / relied upon.
    tm._should_ignore_transcriber_input = TaskManager._should_ignore_transcriber_input.__get__(tm, TaskManager)
    tm._TaskManager__lid_idle_watcher = TaskManager._TaskManager__lid_idle_watcher.__get__(tm, TaskManager)
    return tm


async def _run_watcher_for(tm, seconds):
    task = asyncio.create_task(tm._TaskManager__lid_idle_watcher())
    await asyncio.sleep(seconds)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_no_fire_while_transfer_in_progress():
    # has_transfer True → loop-top guard skips; the decision is never fired (it would
    # abandon pre-drain anyway) and the watcher parks on a 0.5s sleep instead of spinning.
    tm = _tm(has_transfer=True)
    await _run_watcher_for(tm, 0.25)
    assert tm.handle_language_switch.await_count == 0


@pytest.mark.asyncio
async def test_no_fire_while_end_call_in_progress():
    tm = _tm(end_call=True)
    await _run_watcher_for(tm, 0.25)
    assert tm.handle_language_switch.await_count == 0


@pytest.mark.asyncio
async def test_undrained_fire_does_not_spin():
    # Normal state (guard passes) but the decision returns WITHOUT draining (buffer stays
    # aged). The spin-guard must force a yield so the loop is rate-limited (~10/s via the
    # 0.1s floor), NOT a millions-per-second busy-spin. Pre-fix this count was unbounded.
    tm = _tm()
    await _run_watcher_for(tm, 0.35)
    assert 0 < tm.handle_language_switch.await_count < 50
