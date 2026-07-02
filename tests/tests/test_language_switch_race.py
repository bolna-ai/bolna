"""The main reply must not race the switch decision onto the OLD voice: when the unbiased
detector tags the turn as another supported language, the reply is held until the decision
resolves — switched → dropped (the switch follow-up answers), stayed/timeout → played."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from bolna.agent_manager.task_manager import TaskManager
from bolna.synthesizer.synthesizer_pool import SynthesizerPool
from bolna.transcriber.transcriber_pool import TranscriberPool


def _tm(language="hi"):
    tm = MagicMock()
    tm.language = language
    tm.response_in_pipeline = True
    tm.function_call_in_flight = False
    tm._run_llm_task = AsyncMock()
    pool = MagicMock(spec=TranscriberPool)
    pool.lid_detection_events = []
    tm.tools = {"transcriber": pool}
    tm._TaskManager__record_lid_event = TaskManager._TaskManager__record_lid_event.__get__(tm, TaskManager)
    return tm


async def _run_hold(tm, decision, spawn_language="hi"):
    hold = TaskManager._TaskManager__run_llm_after_switch_decision.__get__(tm, TaskManager)
    await hold("pkg", decision, spawn_language)


@pytest.mark.asyncio
async def test_reply_dropped_when_decision_switches():
    tm = _tm()

    async def decide():
        tm.language = "te"  # the decision switched

    await _run_hold(tm, asyncio.create_task(decide()))
    tm._run_llm_task.assert_not_awaited()
    assert tm.response_in_pipeline is False
    events = tm.tools["transcriber"].lid_detection_events
    assert events and events[0]["type"] == "reply_hold" and events[0]["outcome"] == "dropped"


@pytest.mark.asyncio
async def test_reply_runs_despite_switch_when_tool_call_in_flight():
    # The in-flight-tool switch branch produces no follow-up — dropping here would
    # leave the caller unanswered, so the reply must run.
    tm = _tm()
    tm.function_call_in_flight = True

    async def decide():
        tm.language = "te"

    await _run_hold(tm, asyncio.create_task(decide()))
    tm._run_llm_task.assert_awaited_once_with("pkg")


@pytest.mark.asyncio
async def test_reply_runs_when_decision_stays():
    tm = _tm()

    async def decide():
        pass  # stay — language unchanged

    await _run_hold(tm, asyncio.create_task(decide()))
    tm._run_llm_task.assert_awaited_once_with("pkg")


@pytest.mark.asyncio
async def test_reply_runs_on_hold_timeout(monkeypatch):
    monkeypatch.setenv("LANGUAGE_SWITCH_MISMATCH_HOLD_S", "0.05")
    tm = _tm()
    stuck = asyncio.create_task(asyncio.sleep(10))
    try:
        await _run_hold(tm, stuck)
        tm._run_llm_task.assert_awaited_once_with("pkg")
    finally:
        stuck.cancel()


def _mismatch(tm):
    return TaskManager._TaskManager__detector_language_mismatch.__get__(tm, TaskManager)()


def test_detector_mismatch_gate():
    tm = _tm(language="hi")
    pool = MagicMock(spec=TranscriberPool)
    pool.lid_buffer_language = MagicMock(return_value="te")
    pool.labels = ["hi", "te"]
    synth = MagicMock(spec=SynthesizerPool)
    synth.labels = ["hi", "te"]
    tm.tools = {"transcriber": pool, "synthesizer": synth}
    assert _mismatch(tm) is True

    pool.lid_buffer_language = MagicMock(return_value="hi")  # same language
    assert _mismatch(tm) is False

    tm.language = "hi-IN"  # region-tagged active label still matches detector 'hi'
    assert _mismatch(tm) is False
    tm.language = "hi"

    pool.lid_buffer_language = MagicMock(return_value="ta")  # unsupported by ASR pool
    assert _mismatch(tm) is False

    pool.lid_buffer_language = MagicMock(return_value="te")
    synth.labels = ["hi"]  # unsupported by synth pool → half-switch, don't hold
    assert _mismatch(tm) is False

    tm.tools = {"transcriber": MagicMock()}  # not a pool (single-language call)
    assert _mismatch(tm) is False
