"""Teardown after the caller hangs up must not wait on playout that can never be acknowledged.

A caller who hangs up during the goodbye leaves marks pending forever. The playout wait used to
sit out its full deadline, and a detached end_call teardown then ran after run() had released
the tools and raised KeyError on the input handler.
"""

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bolna.agent_manager.task_manager import TaskManager
from bolna.enums import HangupReason
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


@pytest.mark.parametrize("hangs_up_during_last_reply", [False, True])
async def test_no_goodbye_is_rendered_for_a_caller_who_hung_up(hangs_up_during_last_reply):
    task_manager, input_handler = _task_manager_with_pending_goodbye()
    if not hangs_up_during_last_reply:
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

    hangup = asyncio.create_task(task_manager.process_call_hangup())
    await asyncio.sleep(0.05)
    if hangs_up_during_last_reply:
        assert not hangup.done(), "the last reply is still playing, so the goodbye should wait"
        input_handler._end_input_stream("plivo")
    await asyncio.wait_for(hangup, timeout=1)

    assert ended == [True]
    assert task_manager.hangup_message_queued is False


async def test_a_tool_call_is_not_sent_after_the_caller_hung_up():
    task_manager, input_handler = _task_manager_with_pending_goodbye()
    input_handler._end_input_stream("plivo")
    task_manager.run_id = "run-1"
    task_manager.hangup_triggered = False
    task_manager.kwargs = {"api_tools": {"tools_params": {}}}
    task_manager.fire_pre_call_webhook = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("pre-call webhook fired for a disconnected caller")
    )

    with patch("bolna.agent_manager.task_manager.trigger_api", new=AsyncMock()) as trigger_api:
        await task_manager._TaskManager__execute_function_call(
            "https://crm.example/book",
            "POST",
            "{}",
            None,
            None,
            {},
            {"turn_id": 1, "sequence_id": 1},
            "llm",
            "book_appointment",
            tool_call_id="tc-1",
            textual_response=None,
        )

    trigger_api.assert_not_awaited()


async def test_a_caller_hanging_up_during_the_end_call_goodbye_is_not_recorded_as_our_hangup():
    task_manager, input_handler = _task_manager_with_pending_goodbye()
    task_manager.run_id = "run-1"
    task_manager.check_if_user_online = True
    task_manager.interruptible_hangup_message = False
    task_manager._hangup_interruptible_window = False
    task_manager._hangup_cancelled = False
    task_manager._end_call_in_progress = False
    task_manager.hangup_detail = None
    task_manager.conversation_history = MagicMock()
    task_manager._enter_hangup_state = MagicMock()
    task_manager.process_call_hangup = AsyncMock()

    with patch("bolna.agent_manager.task_manager.convert_to_request_log"):
        end_call = asyncio.create_task(
            task_manager._TaskManager__execute_function_call(
                None,
                None,
                None,
                None,
                None,
                {},
                {"turn_id": 1, "sequence_id": 1},
                "llm",
                "end_call",
                tool_call_id="tc-1",
                textual_response="Thank you, bye.",
                model_response=[],
            )
        )
        await asyncio.sleep(0.05)
        input_handler._end_input_stream("plivo")
        await asyncio.wait_for(end_call, timeout=1)
    await asyncio.sleep(0)

    task_manager.process_call_hangup.assert_not_awaited()
    assert task_manager.hangup_detail is None


def _task_manager_asked_to_switch_language(language_switcher):
    task_manager, input_handler = _task_manager_with_pending_goodbye()
    task_manager.run_id = "run-1"
    task_manager.language = "en"
    task_manager.hangup_triggered = False
    task_manager.language_switcher = language_switcher
    task_manager.switch_language = AsyncMock()
    task_manager._TaskManager__do_llm_generation = AsyncMock()
    return task_manager, input_handler


async def _call_switch_language(task_manager):
    await task_manager._TaskManager__execute_function_call(
        None,
        None,
        None,
        None,
        None,
        {},
        {"turn_id": 1, "sequence_id": 1},
        "llm",
        "switch_language",
        language="hi",
        tool_call_id="tc-1",
        model_response=[],
    )


@pytest.mark.parametrize("language_switcher", [object(), None], ids=["lid-flow", "legacy-flow"])
async def test_the_language_is_not_switched_for_a_caller_who_already_hung_up(language_switcher):
    task_manager, input_handler = _task_manager_asked_to_switch_language(language_switcher)
    input_handler._end_input_stream("plivo")

    await _call_switch_language(task_manager)

    task_manager.switch_language.assert_not_awaited()
    task_manager._TaskManager__do_llm_generation.assert_not_awaited()


async def test_the_language_is_not_switched_when_the_caller_hangs_up_during_the_handoff():
    task_manager, input_handler = _task_manager_asked_to_switch_language(None)
    task_manager.switch_handoff_messages = {"en": "One moment, switching to {language}"}
    task_manager._get_voice_name_for_label = lambda label: "Asha"
    task_manager.context_data = {}
    task_manager.tools["output"] = SimpleNamespace(get_provider=lambda: "plivo")
    handoffs = []

    async def _synthesize(packet):
        handoffs.append(packet["data"])
        input_handler._end_input_stream("plivo")

    task_manager._synthesize = _synthesize

    await asyncio.wait_for(_call_switch_language(task_manager), timeout=1)

    assert handoffs == ["One moment, switching to Hindi"]
    task_manager.switch_language.assert_not_awaited()
    task_manager._TaskManager__do_llm_generation.assert_not_awaited()


def _task_manager_ending_the_call(reason, caller_on_line):
    task_manager, input_handler = _task_manager_with_pending_goodbye()
    if not caller_on_line:
        input_handler._end_input_stream("plivo")
    input_handler.stop_handler = AsyncMock()
    task_manager.hangup_detail = reason
    task_manager._caller_on_line_at_hangup = False
    task_manager._end_of_conversation_in_progress = False
    task_manager.hangup_triggered = True
    task_manager.hangup_message_queued = False
    task_manager.turn_based_conversation = False
    task_manager.llm_task = None
    task_manager.wait_for_current_message = AsyncMock()
    task_manager.voicemail_handler = SimpleNamespace(cancel_task=lambda: None)
    return task_manager


AGENT_DECIDED = [HangupReason.END_CALL_TOOL, HangupReason.LLM_PROMPTED_HANGUP]


@pytest.mark.parametrize("reason", AGENT_DECIDED)
async def test_an_agent_reason_is_dropped_when_the_caller_hung_up_before_we_did(reason):
    task_manager = _task_manager_ending_the_call(reason, caller_on_line=False)

    await task_manager._TaskManager__process_end_of_conversation()
    task_manager._settle_hangup_detail()

    assert task_manager.hangup_detail is None


@pytest.mark.parametrize("reason", AGENT_DECIDED)
async def test_an_agent_reason_stands_when_we_hung_up_on_a_caller_still_on_the_line(reason):
    task_manager = _task_manager_ending_the_call(reason, caller_on_line=True)

    await task_manager._TaskManager__process_end_of_conversation()
    task_manager._settle_hangup_detail()

    assert task_manager.hangup_detail == reason


@pytest.mark.parametrize("reason", AGENT_DECIDED)
async def test_a_caller_hanging_up_during_the_hangup_message_is_not_recorded_as_our_hangup(reason):
    # The caller's hangup ends the run, so our teardown never commits to disconnecting.
    task_manager = _task_manager_ending_the_call(reason, caller_on_line=False)
    task_manager.hangup_message_queued = True

    task_manager._settle_hangup_detail()

    assert task_manager.hangup_detail is None


@pytest.mark.parametrize(
    "reason",
    [
        HangupReason.INACTIVITY_TIMEOUT,
        HangupReason.VOICEMAIL_DETECTED,
        HangupReason.MAX_DURATION_REACHED,
        HangupReason.TRANSCRIBER_ERROR,
    ],
)
async def test_a_reason_about_the_call_itself_is_kept_when_the_caller_hung_up_first(reason):
    task_manager = _task_manager_ending_the_call(reason, caller_on_line=False)

    await task_manager._TaskManager__process_end_of_conversation()
    task_manager._settle_hangup_detail()

    assert task_manager.hangup_detail == reason


@pytest.mark.parametrize("reason", AGENT_DECIDED)
async def test_an_agent_reason_stands_when_our_own_hangup_closed_the_stream_first(reason):
    # sip-trunk sends HANGUP the moment the hangup message has played, so Asterisk can close
    # the media stream before our teardown gets to look at it.
    task_manager = _task_manager_ending_the_call(reason, caller_on_line=False)
    task_manager.tools["output"] = MagicMock(hangup_sent=MagicMock(return_value=True))

    await task_manager._TaskManager__process_end_of_conversation()
    task_manager._settle_hangup_detail()

    assert task_manager.hangup_detail == reason
