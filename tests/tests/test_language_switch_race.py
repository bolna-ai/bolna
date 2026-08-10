"""The reply-vs-switch race: generation runs immediately, only PLAYBACK waits for the decision.

The gate lives in __process_output_loop, which every call type runs, so the escapes matter more
than the hold: each one is checked inside the predicate because the loop's WAIT branch has no exit
of its own — it keeps the dequeued message and re-polls, so anything behind it waits too.
"""

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
    tm.task_config = {"tools_config": {}}
    tm.language_switcher = MagicMock()  # non-None = multilingual switch flow is on
    tm.hangup_triggered = False
    tm.conversation_ended = False
    tm._should_ignore_transcriber_input = MagicMock(return_value=False)
    tm.lid_playback_gate = None
    tm._TaskManager__buffered_language_evidence = TaskManager._TaskManager__buffered_language_evidence
    for name in ("arm_lid_playback_gate", "lid_playback_gate_holds", "release_lid_playback_gate"):
        attr = f"_TaskManager__{name}"
        setattr(tm, attr, getattr(TaskManager, attr).__get__(tm, TaskManager))
    for name in ("record_lid_event", "switch_settle_ms", "switch_decide_timeout_s"):
        attr = f"_TaskManager__{name}"
        setattr(tm, attr, getattr(TaskManager, attr).__get__(tm, TaskManager))
    return tm


def _arm(tm, sequence_id, task):
    tm._TaskManager__arm_lid_playback_gate(sequence_id, task)


def _holds(tm, sequence_id):
    return tm._TaskManager__lid_playback_gate_holds(sequence_id)


@pytest.mark.asyncio
async def test_gate_holds_while_the_decision_is_open_and_opens_when_it_resolves():
    tm = _tm()
    decide = asyncio.create_task(asyncio.sleep(5))
    try:
        _arm(tm, 7, decide)
        assert _holds(tm, 7) is True
    finally:
        decide.cancel()
    await asyncio.gather(decide, return_exceptions=True)
    assert _holds(tm, 7) is False  # task.done() releases it
    assert tm.lid_playback_gate is None  # one-shot: opened and forgotten


@pytest.mark.asyncio
async def test_gate_opens_on_the_wall_clock_deadline(monkeypatch):
    # The backstop that makes a wedge impossible: every other escape depends on state that could
    # in principle stop changing, this one cannot.
    monkeypatch.setenv("LANGUAGE_SWITCH_MAX_HOLD_S", "0.05")
    tm = _tm()
    stuck = asyncio.create_task(asyncio.sleep(10))
    try:
        _arm(tm, 7, stuck)
        assert _holds(tm, 7) is True
        await asyncio.sleep(0.08)
        assert _holds(tm, 7) is False
    finally:
        stuck.cancel()
        await asyncio.gather(stuck, return_exceptions=True)


@pytest.mark.asyncio
async def test_gate_never_delays_a_goodbye_or_transfer():
    for flag in ("hangup_triggered", "conversation_ended"):
        tm = _tm()
        stuck = asyncio.create_task(asyncio.sleep(10))
        try:
            _arm(tm, 7, stuck)
            setattr(tm, flag, True)
            assert _holds(tm, 7) is False, flag
        finally:
            stuck.cancel()
            await asyncio.gather(stuck, return_exceptions=True)

    tm = _tm()  # _should_ignore_transcriber_input covers _end_call_in_progress / has_transfer
    stuck = asyncio.create_task(asyncio.sleep(10))
    try:
        _arm(tm, 7, stuck)
        tm._should_ignore_transcriber_input.return_value = True
        assert _holds(tm, 7) is False
    finally:
        stuck.cancel()
        await asyncio.gather(stuck, return_exceptions=True)


@pytest.mark.asyncio
async def test_gate_only_applies_to_its_own_turn():
    tm = _tm()
    stuck = asyncio.create_task(asyncio.sleep(10))
    try:
        _arm(tm, 7, stuck)
        assert _holds(tm, 8) is False  # a later turn must not wait behind this one
        assert _holds(tm, None) is False
        assert _holds(tm, -1) is False  # handoff/system sequence is always sendable
    finally:
        stuck.cancel()
        await asyncio.gather(stuck, return_exceptions=True)


@pytest.mark.asyncio
async def test_system_sequences_are_never_armed():
    tm = _tm()
    stuck = asyncio.create_task(asyncio.sleep(10))
    try:
        _arm(tm, -1, stuck)
        assert tm.lid_playback_gate is None
        _arm(tm, None, stuck)
        assert tm.lid_playback_gate is None
    finally:
        stuck.cancel()
        await asyncio.gather(stuck, return_exceptions=True)


@pytest.mark.asyncio
async def test_gate_is_inert_without_the_switch_flow():
    # Legacy-flow and single-language calls DO reach the output loop; language_switcher is the
    # only thing keeping them out of the gate.
    tm = _tm()
    stuck = asyncio.create_task(asyncio.sleep(10))
    try:
        _arm(tm, 7, stuck)
        tm.language_switcher = None
        assert _holds(tm, 7) is False
    finally:
        stuck.cancel()
        await asyncio.gather(stuck, return_exceptions=True)


@pytest.mark.asyncio
async def test_gate_release_is_recorded_with_outcome_and_duration():
    # The generation hold this replaced wrote reply_hold records. Without an equivalent there is no
    # way to tell a gate that worked (decided, inside the deadline) from one that expired and let
    # old-language audio through — which is also the played/dropped ratio question.
    tm = _tm()
    decide = asyncio.create_task(asyncio.sleep(5))
    try:
        _arm(tm, 7, decide)
        assert _holds(tm, 7) is True
        assert tm.tools["transcriber"].lid_detection_events == []  # nothing recorded while held
    finally:
        decide.cancel()
    await asyncio.gather(decide, return_exceptions=True)
    assert _holds(tm, 7) is False
    rec = [e for e in tm.tools["transcriber"].lid_detection_events if e.get("type") == "playback_gate"]
    assert len(rec) == 1
    assert rec[0]["outcome"] == "decided"
    assert rec[0]["sequence_id"] == 7
    assert rec[0]["from_language"] == "hi"
    assert rec[0]["held_ms"] >= 0
    assert _holds(tm, 7) is False  # one-shot: no duplicate record
    assert len(rec) == 1


@pytest.mark.asyncio
async def test_expired_gate_is_recorded_as_expired(monkeypatch):
    monkeypatch.setenv("LANGUAGE_SWITCH_MAX_HOLD_S", "0.05")
    tm = _tm()
    stuck = asyncio.create_task(asyncio.sleep(10))
    try:
        _arm(tm, 7, stuck)
        await asyncio.sleep(0.08)
        assert _holds(tm, 7) is False
        rec = [e for e in tm.tools["transcriber"].lid_detection_events if e.get("type") == "playback_gate"]
        assert rec and rec[0]["outcome"] == "expired"  # old-language audio played anyway
    finally:
        stuck.cancel()
        await asyncio.gather(stuck, return_exceptions=True)


@pytest.mark.asyncio
async def test_teardown_release_is_recorded_as_teardown():
    tm = _tm()
    stuck = asyncio.create_task(asyncio.sleep(10))
    try:
        _arm(tm, 7, stuck)
        tm.conversation_ended = True
        assert _holds(tm, 7) is False
        rec = [e for e in tm.tools["transcriber"].lid_detection_events if e.get("type") == "playback_gate"]
        assert rec and rec[0]["outcome"] == "teardown"
    finally:
        stuck.cancel()
        await asyncio.gather(stuck, return_exceptions=True)


def test_default_gate_is_a_class_attribute():
    # __process_output_loop reads this on EVERY call; if it were only set in some conditional
    # __init__ branch the shared loop would raise AttributeError and the loop would exit for good.
    assert TaskManager.lid_playback_gate is None


def _mismatch(tm):
    return TaskManager._TaskManager__detector_language_mismatch.__get__(tm, TaskManager)()


def test_detector_mismatch_gate():
    tm = _tm(language="hi")
    pool = MagicMock(spec=TranscriberPool)
    pool.lid_buffer_segments = MagicMock(return_value=[{"lang": "te", "audio_s": 2.0}])
    pool.labels = ["hi", "te"]
    synth = MagicMock(spec=SynthesizerPool)
    synth.labels = ["hi", "te"]
    tm.tools = {"transcriber": pool, "synthesizer": synth}
    assert _mismatch(tm) is True

    # A long active turn must not lend its duration to a mis-tagged fragment.
    pool.lid_buffer_segments = MagicMock(return_value=[{"lang": "hi", "audio_s": 3.0}, {"lang": "te", "audio_s": 0.15}])
    assert _mismatch(tm) is False

    pool.lid_buffer_segments = MagicMock(return_value=[{"lang": "hi", "audio_s": 2.0}])  # same language
    assert _mismatch(tm) is False

    tm.language = "hi-IN"  # region-tagged active label still matches detector 'hi'
    assert _mismatch(tm) is False
    tm.language = "hi"

    pool.lid_buffer_segments = MagicMock(return_value=[{"lang": "ta", "audio_s": 2.0}])  # unsupported by ASR pool
    assert _mismatch(tm) is False

    pool.lid_buffer_segments = MagicMock(return_value=[{"lang": "te"}])
    synth.labels = ["hi"]  # unsupported by synth pool → half-switch, don't hold
    assert _mismatch(tm) is False

    synth.labels = ["hi", "te"]
    pool.lid_buffer_max_segment_seconds = MagicMock(return_value=0.5)
    assert _mismatch(tm) is False  # acknowledgment-length mis-tag → decide would gate; skip the hold

    tm.tools = {"transcriber": MagicMock()}  # not a pool (single-language call)
    assert _mismatch(tm) is False


@pytest.mark.asyncio
async def test_spawn_arms_the_gate_so_the_eager_path_is_covered():
    # The gate is armed inside _spawn_language_switch_decision, not at the call sites: the eager
    # (Flux) turn path spawns the decision too, and arming only at the turn boundary left every
    # eager turn playing the old-language reply and then truncating it.
    tm = _tm(language="hi")
    pool = MagicMock(spec=TranscriberPool)
    pool.lid_buffer_segments = MagicMock(return_value=[{"lang": "te", "audio_s": 2.0}])
    pool.labels = ["hi", "te"]
    synth = MagicMock(spec=SynthesizerPool)
    synth.labels = ["hi", "te"]
    tm.tools = {"transcriber": pool, "synthesizer": synth}
    tm.handle_language_switch = AsyncMock()
    tm._TaskManager__detector_language_mismatch = TaskManager._TaskManager__detector_language_mismatch.__get__(
        tm, TaskManager
    )
    spawn = TaskManager._spawn_language_switch_decision.__get__(tm, TaskManager)

    task = spawn("mala samajla nahi", {"sequence_id": 11})
    try:
        assert task is not None
        assert tm.lid_playback_gate is not None
        assert tm.lid_playback_gate["sequence_id"] == 11
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_spawn_does_not_arm_when_the_detector_agrees():
    tm = _tm(language="hi")
    pool = MagicMock(spec=TranscriberPool)
    pool.lid_buffer_segments = MagicMock(return_value=[{"lang": "hi"}])
    pool.lid_buffer_max_segment_seconds = MagicMock(return_value=2.0)
    pool.labels = ["hi", "te"]
    tm.tools = {"transcriber": pool}
    tm.handle_language_switch = AsyncMock()
    tm._TaskManager__detector_language_mismatch = TaskManager._TaskManager__detector_language_mismatch.__get__(
        tm, TaskManager
    )
    spawn = TaskManager._spawn_language_switch_decision.__get__(tm, TaskManager)
    task = spawn("haan ji", {"sequence_id": 12})
    try:
        assert tm.lid_playback_gate is None  # no mismatch → no reason to delay audio
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
