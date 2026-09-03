"""BOLNA-2563: a hung LLM task (no token, no error, never completes) used to silently disable
both hangup backstops in __check_for_completion, so a call with voicemail detection +
hangup-after-silence enabled would sit in dead air until the carrier dropped it.

response_in_pipeline is cleared in several places (the generation path, the audio-playback path,
teardown, and _run_llm_task's own exception handlers) - but all of them require the in-flight
work to actually finish or raise. A generation that never does either reaches none of them, so
both signals downstream stay "busy" indefinitely:

  - has_pending_generation reads llm_task.done(), which is False for the life of the hang.
  - response_in_pipeline stays True, so _pipeline_busy() stays True.

_should_stall_hangup is the backstop for exactly "no forward progress at all": past
STALL_HANGUP_HARD_CAP_S of mutual silence with no audio playing, it fires unconditionally - it
does not take has_pending_generation into account at all, so a hung task (or any other in-flight
work) can never wedge the call open past the hard cap. _run_llm_task's own LLM_GENERATION_TIMEOUT_S
(scoped to __do_llm_generation) is what protects legitimate in-flight generation from being cut
off before that point, and from being confused with unrelated work that runs after it.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import bolna.agent_manager.task_manager as task_manager_module
from bolna.agent_manager.task_manager import TaskManager
from bolna.constants import STALL_HANGUP_HARD_CAP_S

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


async def test_hung_llm_task_leaves_response_in_pipeline_stuck_so_pipeline_stays_busy(hung_llm_task):
    """response_in_pipeline is set True when a turn kicks off and only clears once that turn
    finishes or raises. A hang that never does either never reaches that clear, so
    _pipeline_busy() reads busy indefinitely - and __check_for_completion's pipeline-busy gate
    means the normal hang_conversation_after check below it is never reached."""
    has_pending_generation = hung_llm_task is not None and not hung_llm_task.done()
    assert has_pending_generation is True  # the hang is still "in flight"

    stuck_response_in_pipeline = True  # never cleared - the turn never finished or raised
    fake = SimpleNamespace(response_in_pipeline=stuck_response_in_pipeline, _synthesis_awaiting_first_audio=False)

    busy = TaskManager._pipeline_busy(fake, audio_playing=False)
    assert busy is True  # bug: __check_for_completion `continue`s here, forever


async def test_hung_generation_still_times_out_via_the_wrapper(monkeypatch):
    """The timeout now lives on __do_llm_generation itself, not around the whole conversation
    task - confirm a hung implementation still raises within LLM_GENERATION_TIMEOUT_S instead
    of hanging forever."""
    monkeypatch.setattr(task_manager_module, "LLM_GENERATION_TIMEOUT_S", 0.05)
    tm = TaskManager.__new__(TaskManager)

    async def _hang(*args, **kwargs):
        await asyncio.Event().wait()

    tm._TaskManager__do_llm_generation_impl = AsyncMock(side_effect=_hang)

    # A 1.0s outer bound only guards the test itself against hanging; asserting elapsed time
    # stays well under it proves LLM_GENERATION_TIMEOUT_S (0.05s) is what actually fired.
    start = asyncio.get_event_loop().time()
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(TaskManager._TaskManager__do_llm_generation(tm, None, None, None), timeout=1.0)
    assert asyncio.get_event_loop().time() - start < 0.5


async def test_slow_work_after_generation_is_no_longer_capped(monkeypatch):
    """_run_llm_task no longer wraps the whole conversation task - work that runs after
    generation (the hangup-decision call, or hangup teardown) can legitimately outlast
    LLM_GENERATION_TIMEOUT_S without being cut off mid-way. BOLNA-2563."""
    monkeypatch.setattr(task_manager_module, "LLM_GENERATION_TIMEOUT_S", 0.05)
    tm = TaskManager.__new__(TaskManager)
    tm.task_config = {"task_type": "conversation"}

    async def _slow(*args, **kwargs):
        await asyncio.sleep(0.15)  # deliberately past the patched timeout above

    tm._process_conversation_task = AsyncMock(side_effect=_slow)
    tm._end_call_on_component_error = AsyncMock()

    await asyncio.wait_for(tm._run_llm_task({}), timeout=1.0)

    tm._end_call_on_component_error.assert_not_awaited()
