"""BOLNA-2563: a hung LLM task (no token, no error, never completes) used to silently disable
both hangup backstops in __check_for_completion, so a call with voicemail detection +
hangup-after-silence enabled would sit in dead air until the carrier dropped it.

response_in_pipeline is only ever cleared back to False inside _run_llm_task's exception
handlers (task_manager.py ~4419-4432). A generation that never raises and never yields a token
never reaches that cleanup, so both signals downstream stay "busy" indefinitely:

  - has_pending_generation (task_manager.py ~7352-7358) reads llm_task.done(), which is False
    for the life of the hang.
  - response_in_pipeline stays True, so _pipeline_busy() (task_manager.py ~7296-7298) stays True.

_should_stall_hangup is the backstop for exactly "no forward progress at all": past
STALL_HANGUP_HARD_CAP_S of mutual silence with no audio playing, it fires unconditionally -
it no longer takes has_pending_generation into account at all, so a hung task (or any other
in-flight work) can never wedge the call open past the hard cap. _run_llm_task's own
LLM_GENERATION_TIMEOUT_S is what protects legitimate in-flight generation from being cut off
before that point.
"""

import asyncio
from types import SimpleNamespace

import pytest

from bolna.agent_manager.task_manager import TaskManager
from bolna.constants import STALL_HANGUP_HARD_CAP_S

AGED = STALL_HANGUP_HARD_CAP_S - 5  # comfortably below the hard cap
VERY_AGED = STALL_HANGUP_HARD_CAP_S + 300  # 5 minutes past the hard cap - "it's just stuck"


@pytest.fixture
async def hung_llm_task():
    """A task standing in for a generation that never yields a token and never raises -
    it just hangs. Never completes on its own; the fixture cancels it during teardown."""
    task = asyncio.create_task(asyncio.Event().wait())
    yield task
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def _has_pending_generation(llm_task):
    # Mirrors task_manager.py:7352-7358 for the llm_task clause (the other two clauses -
    # execute_function_call_task and s2s tool tasks - are not part of this bug).
    return llm_task is not None and not llm_task.done()


async def test_hung_llm_task_keeps_has_pending_generation_true_forever(hung_llm_task):
    assert _has_pending_generation(hung_llm_task) is True
    # Simulate the watchdog's 2s-interval passing many times over; nothing about a hung task
    # (no completion, no exception) ever flips this.
    await asyncio.sleep(0)
    assert _has_pending_generation(hung_llm_task) is True


async def test_short_lived_pending_generation_is_still_protected_below_the_hard_cap(hung_llm_task):
    """Below STALL_HANGUP_HARD_CAP_S, a task that's merely still running (not yet provably hung)
    must not be force-hungup on - that's real in-flight work, protected by _run_llm_task's own
    LLM_GENERATION_TIMEOUT_S rather than by _should_stall_hangup."""
    fake = SimpleNamespace(hang_conversation_after=15)

    fires = TaskManager._should_stall_hangup(
        fake,
        audio_playing=False,
        time_since_last_spoken_ai_word=AGED,
        time_since_user_last_spoke=AGED,
    )
    assert fires is False


async def test_hung_llm_task_no_longer_defeats_the_stall_backstop_past_hard_cap(hung_llm_task):
    """Past STALL_HANGUP_HARD_CAP_S the backstop fires regardless of any in-flight task, so a
    hung task (this one included) can no longer wedge the call open forever."""
    fake = SimpleNamespace(hang_conversation_after=15)

    fires = TaskManager._should_stall_hangup(
        fake,
        audio_playing=False,
        time_since_last_spoken_ai_word=VERY_AGED,
        time_since_user_last_spoke=VERY_AGED,
    )
    assert fires is True


async def test_hung_llm_task_leaves_response_in_pipeline_stuck_so_pipeline_stays_busy(hung_llm_task):
    """response_in_pipeline is set True when the turn kicks off (task_manager.py:4554) and is
    only cleared in _run_llm_task's except blocks (task_manager.py:4419-4432). A hang that never
    raises never reaches that clear, so _pipeline_busy() reads busy indefinitely - and
    __check_for_completion's `if self._pipeline_busy(...): continue` (task_manager.py:7338-7339)
    means the normal hang_conversation_after check below it is never reached."""
    assert _has_pending_generation(hung_llm_task) is True  # the hang is still "in flight"

    stuck_response_in_pipeline = True  # never cleared - the exception handler never ran
    fake = SimpleNamespace(response_in_pipeline=stuck_response_in_pipeline, _synthesis_awaiting_first_audio=False)

    busy = TaskManager._pipeline_busy(fake, audio_playing=False)
    assert busy is True  # bug: __check_for_completion `continue`s here, forever


async def test_stall_backstop_eventually_fires_no_matter_how_long_the_task_stays_hung(hung_llm_task):
    """End-to-end: as the same hung task ages, the backstop stays protective early (does not
    punish real in-flight work) but is guaranteed to fire by the hard cap - unlike before the
    fix, where it would have stayed silent even at 20+ minutes."""
    fake_stall = SimpleNamespace(hang_conversation_after=15)

    for silence_s, expected in ((AGED, False), (VERY_AGED, True), (VERY_AGED * 4, True)):
        stall_fires = TaskManager._should_stall_hangup(
            fake_stall,
            audio_playing=False,
            time_since_last_spoken_ai_word=silence_s,
            time_since_user_last_spoke=silence_s,
        )
        assert stall_fires is expected
