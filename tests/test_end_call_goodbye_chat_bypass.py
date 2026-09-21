"""The goodbye generated after end_call must reach a turn-based chat client.

Chat text is delivered only through _handle_llm_output's bypass-synth branch (the dashboard input handler
stamps meta_info["bypass_synth"] = True). The end_call follow-up generation dropped that flag, so the goodbye
went to a synthesizer queue nothing consumes and the socket closed before the client saw a word.
"""

from unittest.mock import AsyncMock, MagicMock, patch

from bolna.agent_manager.task_manager import TaskManager

MOD = "bolna.agent_manager.task_manager"


def _make_tm():
    tm = TaskManager.__new__(TaskManager)
    tm.run_id = "run-1"
    tm.interruptible_hangup_message = False
    tm._hangup_interruptible_window = False
    tm._hangup_cancelled = False
    tm.hangup_detail = None
    tm.call_hangup_message_config = None
    tm.check_if_user_online = True
    tm.llm_config = {"model": "gpt-4.1-mini"}
    tm.conversation_history = MagicMock()
    tm.conversation_history.get_copy.return_value = [{"role": "user", "content": "यह गलत नंबर है"}]
    tm._spawn_followup_meta_info = lambda meta_info: dict(meta_info)
    tm._TaskManager__do_llm_generation = AsyncMock()
    tm._enter_hangup_state = MagicMock()
    tm.wait_for_current_message = AsyncMock()
    tm.process_call_hangup = AsyncMock()
    tm._TaskManager__log_detached_hangup_exception = MagicMock()
    return tm


async def _end_call(tm, meta_info):
    with patch(f"{MOD}.convert_to_request_log"), patch(f"{MOD}.format_messages", return_value=""):
        await tm._TaskManager__execute_function_call(
            url=None,
            method=None,
            param=None,
            api_token=None,
            headers=None,
            model_args=None,
            meta_info=meta_info,
            next_step="synthesizer",
            called_fun="end_call",
            reason="wrong number",
            model_response={},
            tool_call_id="tool-1",
        )
        await tm._end_call_hangup_task


async def test_chat_goodbye_follow_up_bypasses_the_synthesizer():
    tm = _make_tm()
    await _end_call(tm, {"bypass_synth": True, "turn_id": 3, "response_uid": "r3", "sequence_id": 3})
    tm._TaskManager__do_llm_generation.assert_awaited_once()
    kwargs = tm._TaskManager__do_llm_generation.await_args.kwargs
    assert kwargs["should_bypass_synth"] is True and kwargs["should_trigger_function_call"] is False
    tm._enter_hangup_state.assert_called_once()
    tm.process_call_hangup.assert_awaited_once()


async def test_voice_goodbye_follow_up_still_goes_to_the_synthesizer():
    tm = _make_tm()
    await _end_call(tm, {"turn_id": 3, "response_uid": "r3", "sequence_id": 3})
    assert tm._TaskManager__do_llm_generation.await_args.kwargs["should_bypass_synth"] is False


async def test_goodbye_in_the_same_turn_skips_the_follow_up():
    tm = _make_tm()
    with patch(f"{MOD}.convert_to_request_log"):
        await tm._TaskManager__execute_function_call(
            url=None,
            method=None,
            param=None,
            api_token=None,
            headers=None,
            model_args=None,
            meta_info={"bypass_synth": True, "turn_id": 1, "response_uid": "r1", "sequence_id": 1},
            next_step="synthesizer",
            called_fun="end_call",
            reason="bye",
            model_response={},
            tool_call_id="t",
            textual_response="Goodbye!",
        )
        await tm._end_call_hangup_task
    tm._TaskManager__do_llm_generation.assert_not_awaited()
    assert tm._end_call_hangup_task.done()  # the teardown was still armed
    tm.process_call_hangup.assert_awaited_once()
