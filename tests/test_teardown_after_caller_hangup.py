"""Teardown after the caller hangs up must not wait on playout that can never be acknowledged.

A caller who hangs up during the goodbye leaves marks pending forever. The playout wait used to
sit out its full deadline, and a detached end_call teardown then ran after run() had released
the tools and raised KeyError on the input handler.
"""

import asyncio
import time
from types import SimpleNamespace

from bolna.agent_manager.task_manager import TaskManager
from bolna.helpers.mark_event_meta_data import MarkEventMetaData
from bolna.input_handlers.default import DefaultInputHandler


def _task_manager_with_pending_goodbye():
    marks = MarkEventMetaData()
    marks.update_data(
        "goodbye-mark",
        {"type": "agent_hangup", "text_synthesized": "bye", "duration": 5.0, "sent_ts": time.time()},
    )
    input_handler = DefaultInputHandler(queues={"transcriber": asyncio.Queue()}, mark_event_meta_data=marks)

    task_manager = TaskManager.__new__(TaskManager)
    task_manager.conversation_ended = False
    task_manager.hangup_mark_event_timeout = 10
    task_manager._turn_audio_flushed = asyncio.Event()
    task_manager._turn_audio_flushed.set()
    task_manager.mark_event_meta_data = marks
    task_manager.tools = {"input": input_handler}
    return task_manager, input_handler


async def test_playout_wait_ends_as_soon_as_the_caller_hangs_up():
    task_manager, input_handler = _task_manager_with_pending_goodbye()

    wait = asyncio.create_task(task_manager.wait_for_current_message())
    await asyncio.sleep(0.05)
    assert not wait.done(), "the goodbye is still playing, so the wait should block"

    started = time.monotonic()
    input_handler._end_input_stream("plivo")
    await asyncio.wait_for(wait, timeout=1)
    assert time.monotonic() - started < 1


async def test_playout_wait_returns_immediately_once_the_caller_is_gone():
    task_manager, input_handler = _task_manager_with_pending_goodbye()
    input_handler._end_input_stream("plivo")

    await asyncio.wait_for(task_manager.wait_for_current_message(), timeout=0.5)


async def test_teardown_finishing_after_run_released_the_tools_does_not_raise():
    task_manager = TaskManager.__new__(TaskManager)
    task_manager._end_of_conversation_in_progress = False
    task_manager.conversation_ended = False
    task_manager.hangup_triggered = True
    task_manager.hangup_message_queued = False
    task_manager.turn_based_conversation = False
    task_manager.llm_task = None
    task_manager.tools = {}
    task_manager.voicemail_handler = SimpleNamespace(cancel_task=lambda: None)

    await task_manager._TaskManager__process_end_of_conversation()

    assert task_manager.conversation_ended


async def test_no_goodbye_is_rendered_for_a_caller_who_already_hung_up():
    task_manager, input_handler = _task_manager_with_pending_goodbye()
    input_handler._end_input_stream("plivo")
    task_manager.hangup_decision_at = None
    task_manager._hangup_processing = False
    task_manager.call_hangup_message_config = "Thank you, goodbye"
    task_manager.language = "en"
    task_manager.voicemail_handler = SimpleNamespace(detected=False)
    task_manager._TaskManager__is_s2s = lambda: False
    ended = []

    async def _end_of_conversation():
        ended.append(True)

    async def _synthesize(_packet):
        raise AssertionError("goodbye rendered for a disconnected caller")

    task_manager._TaskManager__process_end_of_conversation = _end_of_conversation
    task_manager._synthesize = _synthesize

    await task_manager.process_call_hangup()

    assert ended == [True]
    assert task_manager.hangup_message_queued is False
