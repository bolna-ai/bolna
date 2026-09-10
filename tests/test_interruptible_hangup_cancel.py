"""When interruptible_hangup_message is on, a barge-in during the goodbye cancels the pending
hangup and resumes the call. _cancel_pending_hangup is the synchronous reset that makes the
resume safe: it stops the goodbye's llm task and the detached hangup task before either can
commit the disconnect, and clears every hangup flag.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from bolna.agent_manager.task_manager import TaskManager


def _target(llm_task=None, hangup_task=None):
    return SimpleNamespace(
        _hangup_interruptible_window=True,
        _hangup_cancelled=False,
        conversation_ended=False,
        conversation_config={"check_if_user_online": True},
        check_if_user_online=False,
        llm_task=llm_task,
        _end_call_hangup_task=hangup_task,
        hangup_triggered=True,
        _end_call_in_progress=True,
        hangup_message_queued=True,
        hangup_triggered_at=123.0,
        hangup_decision_at=123.0,
        _hangup_processing=True,
        _end_of_conversation_in_progress=True,
        hangup_detail="END_CALL_TOOL",
    )


async def _forever():
    await asyncio.sleep(100)


async def test_resets_every_hangup_flag():
    tm = _target()
    TaskManager._cancel_pending_hangup(tm)

    assert tm._hangup_interruptible_window is False
    assert tm.hangup_triggered is False
    assert tm._end_call_in_progress is False
    assert tm.hangup_message_queued is False
    assert tm.hangup_triggered_at is None
    assert tm.hangup_decision_at is None
    assert tm._hangup_processing is False
    assert tm._end_of_conversation_in_progress is False
    assert tm.hangup_detail is None


async def test_cancels_goodbye_and_detached_hangup_tasks():
    llm_task = asyncio.ensure_future(_forever())
    hangup_task = asyncio.ensure_future(_forever())
    await asyncio.sleep(0)  # let both start running

    tm = _target(llm_task=llm_task, hangup_task=hangup_task)
    TaskManager._cancel_pending_hangup(tm)

    assert tm.llm_task is None
    assert tm._end_call_hangup_task is None

    for t in (llm_task, hangup_task):
        try:
            await t
        except asyncio.CancelledError:
            pass
    assert llm_task.cancelled()
    assert hangup_task.cancelled()


async def test_marks_the_hangup_cancelled_so_end_call_bails_out():
    """Task cancellation alone does not stop the end_call branch: handle_interruption clears the
    mark dict, so its playout wait can return normally and arm the teardown anyway. The branch
    re-checks this flag, so setting it is what actually keeps the rescued call alive.
    """
    tm = _target()
    TaskManager._cancel_pending_hangup(tm)
    assert tm._hangup_cancelled is True


async def test_does_not_mark_cancelled_once_conversation_ended():
    # The disconnect already committed, so the end_call branch must not be told to bail out.
    tm = _target()
    tm.conversation_ended = True
    TaskManager._cancel_pending_hangup(tm)
    assert tm._hangup_cancelled is False


async def test_restores_the_user_online_check_on_resume():
    # __execute_function_call clears it on entry and the end_call branch returns without restoring,
    # so without this a resumed call never asks "are you still there" again.
    tm = _target()
    TaskManager._cancel_pending_hangup(tm)
    assert tm.check_if_user_online is True


async def test_noop_once_conversation_ended():
    # A committed disconnect must survive an admitted-late barge-in: the hangup task keeps running.
    hangup_task = asyncio.ensure_future(_forever())
    await asyncio.sleep(0)
    tm = _target(hangup_task=hangup_task)
    tm.conversation_ended = True

    TaskManager._cancel_pending_hangup(tm)

    assert tm._hangup_interruptible_window is False
    assert tm.hangup_triggered is True
    assert tm._end_call_hangup_task is hangup_task
    assert not hangup_task.cancelled()
    hangup_task.cancel()
    await asyncio.gather(hangup_task, return_exceptions=True)


async def test_never_cancels_the_current_task():
    # _cancel_pending_hangup runs on the transcriber task's stack; it must not cancel a task it
    # is itself executing inside, or the resume unwinds before it finishes.
    current = asyncio.current_task()
    tm = _target(llm_task=current, hangup_task=current)
    TaskManager._cancel_pending_hangup(tm)
    assert not current.cancelled()


def _cleanup_target():
    """A TaskManager with just enough wired up to run the real __cleanup_downstream_tasks."""
    tm = TaskManager.__new__(TaskManager)
    tm._hangup_interruptible_window = True
    tm._hangup_cancelled = False
    tm.conversation_ended = False
    tm.conversation_config = {"check_if_user_online": True}
    tm.check_if_user_online = False
    tm._end_call_hangup_task = None
    tm.hangup_triggered = True
    tm._end_call_in_progress = True
    tm.hangup_message_queued = True
    tm.hangup_triggered_at = 1.0
    tm.hangup_decision_at = 1.0
    tm._hangup_processing = True
    tm._end_of_conversation_in_progress = True
    tm.hangup_detail = "END_CALL_TOOL"

    tm._cancel_in_flight_llm_response = MagicMock()
    tm.regen_settle_armed = MagicMock(return_value=False)
    tm.regen_settle_payload = None
    tm.sync_history = AsyncMock()
    tm._drop_all_staged_assistant_history = MagicMock()
    tm.response_in_pipeline = True
    tm._synthesis_awaiting_first_audio = True
    tm.output_task = None
    tm.eager_llm_task = None
    tm.first_message_task = None
    tm.synthesizer_tasks = []
    tm.started_transmitting_audio = True
    tm.last_transmitted_timestamp = 0
    tm.buffered_output_queue = asyncio.Queue()
    tm._turn_audio_flushed = asyncio.Event()

    tm.interruption_manager = MagicMock()
    tm.mark_event_meta_data = MagicMock()
    tm.mark_event_meta_data.fetch_cleared_mark_event_data = MagicMock(return_value={})
    tm.voicemail_handler = SimpleNamespace(cancel_task=lambda: None)

    output = SimpleNamespace(handle_interruption=AsyncMock())
    synth = SimpleNamespace(handle_interruption=AsyncMock(), flush_synthesizer_stream=AsyncMock())
    inp = SimpleNamespace(
        welcome_message_played=MagicMock(return_value=True),
        reset_response_heard_by_user=MagicMock(),
        is_welcome_message_played=True,
    )
    tm.tools = {"output": output, "synthesizer": synth, "input": inp}
    tm._TaskManager__process_output_loop = AsyncMock()
    return tm


async def test_cleanup_cancels_goodbye_before_it_can_rearm_the_disconnect():
    """The keystone race: a barge-in reaches __cleanup_downstream_tasks while the goodbye task is
    parked in its playout wait. The goodbye must be cancelled before this cleanup's first await,
    or it resumes during those awaits and arms the disconnect after we chose to keep the call.
    """
    tm = _cleanup_target()
    tm._rearmed = False

    async def _goodbye_that_would_rearm():
        # Stand-in for the inline wait_for_current_message resuming and arming the detached hangup.
        for _ in range(10):
            await asyncio.sleep(0)
        tm._rearmed = True

    goodbye = asyncio.ensure_future(_goodbye_that_would_rearm())
    tm.llm_task = goodbye
    await asyncio.sleep(0)  # let the goodbye task start and park

    await TaskManager._TaskManager__cleanup_downstream_tasks(tm)
    # Gathering would run the goodbye to completion (setting _rearmed) had it not been cancelled.
    await asyncio.gather(goodbye, return_exceptions=True)

    assert goodbye.cancelled()
    assert tm._rearmed is False, "goodbye task resumed and armed the disconnect after the barge-in"
    assert tm.hangup_triggered is False
    assert tm._end_call_in_progress is False
    assert tm._hangup_interruptible_window is False
