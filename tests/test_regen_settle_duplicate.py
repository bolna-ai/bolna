"""A regen must not fire once the turn it meant to replace has already been spoken.

Incident 9b14c2b2 (Spinny): the caller's utterance arrived as two finals. The first turn's reply was
committed and marked ~2ms before the debounce was armed, so the regen replaced nothing and appended a
near-identical second utterance — the opening line was spoken twice with no interrupting user turn.
"""

import asyncio

import pytest

from bolna.agent_manager.task_manager import TaskManager


class _Stub:
    """Only the attributes arm_regen_settle / __regen_after_settle touch."""


class _FakeTask:
    """Stands in for a live debounce timer so arm_regen_settle takes its re-arm branch."""

    def cancel(self):
        pass

    def done(self):
        return False


def _manager(sent_audio_sequences=(), armed=False):
    tm = _Stub()
    tm.regen_settle_task = _FakeTask() if armed else None
    tm.regen_settle_payload = None
    tm.regen_settle_supersedes = None
    tm._sent_audio_sequences = set(sent_audio_sequences)
    tm.kicked_off = []
    tm.kickoff_llm_generation = lambda msg, meta: tm.kicked_off.append((msg, meta))
    tm.regen_settle_armed = lambda: TaskManager.regen_settle_armed(tm)

    async def _noop():
        return None

    # arm_regen_settle schedules this; the tests drive the real coroutine themselves.
    tm._TaskManager__regen_after_settle = _noop
    return tm


def _arm(tm, text, seq, turn):
    TaskManager.arm_regen_settle(tm, text, {"sequence_id": seq, "turn_id": turn})


async def _fire(tm):
    await TaskManager._TaskManager__regen_after_settle(tm)


@pytest.fixture(autouse=True)
def _instant_settle(monkeypatch):
    """Collapse the 0.7s window; real asyncio.sleep stays intact for everything else."""
    monkeypatch.setattr("bolna.agent_manager.task_manager.LLM_REGEN_SETTLE_S", 0)


async def test_regen_is_skipped_when_the_superseded_turn_was_already_spoken():
    # The 9b14c2b2 shape: seq=1's audio shipped, then the second final armed a regen as seq=2.
    tm = _manager(sent_audio_sequences={1})
    _arm(tm, "हेलो। हाँ, बताइए।", seq=2, turn=2)
    await _fire(tm)
    assert tm.kicked_off == []


async def test_regen_still_fires_when_nothing_has_been_spoken():
    # The healthy supersede: the first turn is still in the pipeline, so replacing it is correct.
    tm = _manager(sent_audio_sequences=set())
    _arm(tm, "हेलो। हाँ, बताइए।", seq=2, turn=2)
    await _fire(tm)
    assert [m for m, _ in tm.kicked_off] == ["हेलो। हाँ, बताइए।"]


async def test_an_unrelated_earlier_turn_does_not_block_the_regen():
    # seq=1 was spoken turns ago; this episode supersedes seq=3, which has not been spoken.
    tm = _manager(sent_audio_sequences={1})
    _arm(tm, "merged", seq=4, turn=4)
    await _fire(tm)
    assert len(tm.kicked_off) == 1


async def test_a_burst_of_finals_keeps_the_original_supersede_anchor():
    """Re-arms must measure against the response actually in flight, not the newest neighbour.

    Without the episode anchor the guard would test seq=3 (never spoken) and the duplicate would
    still reach the caller.
    """
    tm = _manager(sent_audio_sequences={1})
    _arm(tm, "हेलो।", seq=2, turn=2)
    assert tm.regen_settle_supersedes == 1
    tm.regen_settle_task = _FakeTask()  # episode still armed
    _arm(tm, "हेलो। हाँ,", seq=3, turn=3)
    _arm(tm, "हेलो। हाँ, बताइए।", seq=4, turn=4)
    assert tm.regen_settle_supersedes == 1
    await _fire(tm)
    assert tm.kicked_off == []


async def test_the_anchor_is_cleared_after_firing():
    tm = _manager(sent_audio_sequences=set())
    _arm(tm, "merged", seq=2, turn=2)
    await _fire(tm)
    assert tm.regen_settle_supersedes is None


async def test_no_payload_is_a_no_op():
    tm = _manager()
    tm.regen_settle_payload = None
    await _fire(tm)
    assert tm.kicked_off == []
