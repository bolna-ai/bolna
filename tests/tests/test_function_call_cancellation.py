"""Tests for the cancellation + history-record behavior of __execute_function_call.

These cover the production bug where user interruption mid-trigger_api would
leave no trace of the attempted tool call in conversation history, causing the
LLM to re-emit the same call indefinitely on subsequent turns.

The fix records the tool_call attempt BEFORE awaiting trigger_api with a
pending placeholder, overwrites with the real body on success, or with an
"interrupted_by_user" marker on CancelledError.
"""

import asyncio
import copy
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from bolna.agent_manager.task_manager import TaskManager
from bolna.helpers.conversation_history import ConversationHistory


def _build_task_manager_mock(conversation_history: ConversationHistory) -> MagicMock:
    """Build a TaskManager-shaped mock that uses a real ConversationHistory."""
    tm = MagicMock()
    tm.conversation_ended = False
    tm.hangup_triggered = False
    tm.run_id = "run-test"
    tm.check_if_user_online = True
    tm.context_data = None
    tm.conversation_history = conversation_history
    tm.conversation_config = {}
    tm.llm_config = {"model": "gpt-4o"}
    tm.generate_precise_transcript = False
    tm.tools = {"output": MagicMock(), "llm_agent": MagicMock()}
    tm.execute_function_call_task = None
    tm._TaskManager__is_graph_agent = MagicMock(return_value=False)
    tm.wait_for_current_message = AsyncMock()
    tm._commit_staged_assistant_history = MagicMock()
    tm._spawn_followup_meta_info = MagicMock(return_value={"sequence_id": 2, "turn_id": 2})
    tm._start_api_call_detail = MagicMock(return_value={})
    tm._finalize_api_call_detail = MagicMock()
    tm._extract_api_call_runtime_args = MagicMock(return_value={})
    # Name-mangled private method must be set explicitly because MagicMock
    # auto-attributes return MagicMock instances that cannot be awaited.
    tm._TaskManager__do_llm_generation = AsyncMock()
    return tm


class TestFunctionCallCancellation:
    @pytest.mark.asyncio
    async def test_history_has_pending_placeholder_before_trigger_api(self):
        """attach_tool_calls + append_tool_result(pending) run BEFORE trigger_api.

        This is the key invariant: if anything later raises CancelledError,
        history already has the assistant.tool_calls and a tool message.
        """
        ch = ConversationHistory()
        ch.append_user("track my order")
        ch.append_assistant("एक मिनट दीजिए", turn_id=1)

        tm = _build_task_manager_mock(ch)

        observed_history_at_trigger_api: list = []

        async def slow_trigger_api(**kwargs):
            # Snapshot history at the moment trigger_api starts so we can
            # confirm the placeholder was already written. Deep-copy
            # because the tool message gets mutated in place after this
            # function returns.
            observed_history_at_trigger_api.extend(copy.deepcopy(ch.messages))
            return {"body": '{"order": "shipped"}', "status_code": 200, "content_type": "application/json"}

        model_response = [{"id": "call_X", "function": {"name": "advanced_track_order"}}]

        with patch("bolna.agent_manager.task_manager.trigger_api", side_effect=slow_trigger_api):
            with patch(
                "bolna.agent_manager.task_manager.computed_api_response",
                new_callable=AsyncMock,
                return_value=([], []),
            ):
                with patch(
                    "bolna.agent_manager.task_manager.TaskManager._TaskManager__do_llm_generation",
                    new_callable=AsyncMock,
                ):
                    with patch("bolna.agent_manager.task_manager.convert_to_request_log"):
                        with patch("bolna.agent_manager.task_manager.format_messages", return_value=""):
                            await TaskManager._TaskManager__execute_function_call(
                                tm,
                                url="https://api.example.com/tools",
                                method="post",
                                param=None,
                                api_token=None,
                                headers=None,
                                model_args={},
                                meta_info={"sequence_id": 1, "turn_id": 1, "bypass_synth": False},
                                next_step=None,
                                called_fun="advanced_track_order",
                                model_response=model_response,
                                tool_call_id="call_X",
                                textual_response=None,
                            )

        # At the moment trigger_api was entered, history must already contain
        # the assistant.tool_calls + a tool message (pending placeholder).
        assistant_with_tool_calls = [
            m for m in observed_history_at_trigger_api if m.get("role") == "assistant" and m.get("tool_calls")
        ]
        tool_messages = [m for m in observed_history_at_trigger_api if m.get("role") == "tool"]
        assert len(assistant_with_tool_calls) == 1, "Assistant.tool_calls must be set before trigger_api"
        assert assistant_with_tool_calls[0]["tool_calls"][0]["id"] == "call_X"
        assert len(tool_messages) == 1, "A pending tool placeholder must exist before trigger_api"
        assert tool_messages[0]["tool_call_id"] == "call_X"
        assert "pending" in tool_messages[0]["content"]

        # After success, the placeholder should have been overwritten with
        # the real response body, not still "pending".
        final_tools = [m for m in ch.messages if m.get("role") == "tool"]
        assert len(final_tools) == 1
        assert "pending" not in final_tools[0]["content"]
        assert "shipped" in final_tools[0]["content"]

    @pytest.mark.asyncio
    async def test_cancellation_mid_trigger_api_marks_tool_as_interrupted(self):
        """When CancelledError lands inside trigger_api, the pending tool
        placeholder is converted to an 'interrupted_by_user' marker so the
        LLM next turn sees the attempt happened."""
        ch = ConversationHistory()
        ch.append_user("track my order")
        ch.append_assistant("एक मिनट दीजिए", turn_id=1)

        tm = _build_task_manager_mock(ch)

        # trigger_api blocks long enough for us to cancel it mid-flight.
        async def hanging_trigger_api(**kwargs):
            await asyncio.sleep(10)
            return {"body": "never reached"}

        model_response = [{"id": "call_Y", "function": {"name": "advanced_track_order"}}]

        with patch("bolna.agent_manager.task_manager.trigger_api", side_effect=hanging_trigger_api):
            with patch(
                "bolna.agent_manager.task_manager.computed_api_response",
                new_callable=AsyncMock,
                return_value=([], []),
            ):
                with patch(
                    "bolna.agent_manager.task_manager.TaskManager._TaskManager__do_llm_generation",
                    new_callable=AsyncMock,
                ):
                    with patch("bolna.agent_manager.task_manager.convert_to_request_log"):
                        task = asyncio.create_task(
                            TaskManager._TaskManager__execute_function_call(
                                tm,
                                url="https://api.example.com/tools",
                                method="post",
                                param=None,
                                api_token=None,
                                headers=None,
                                model_args={},
                                meta_info={"sequence_id": 1, "turn_id": 1, "bypass_synth": False},
                                next_step=None,
                                called_fun="advanced_track_order",
                                model_response=model_response,
                                tool_call_id="call_Y",
                                textual_response=None,
                            )
                        )

                        # Wait for the task to enter the hanging trigger_api,
                        # then cancel — simulating the user interruption.
                        await asyncio.sleep(0.05)
                        task.cancel()
                        with pytest.raises(asyncio.CancelledError):
                            await task

        # History must show the cancellation as a tool message, not vanish.
        tool_messages = [m for m in ch.messages if m.get("role") == "tool"]
        assert len(tool_messages) == 1
        assert tool_messages[0]["tool_call_id"] == "call_Y"
        assert "interrupted_by_user" in tool_messages[0]["content"]

        # And the assistant must keep its tool_calls field so the pair stays
        # well-formed on the next get_copy().
        assistants_with_calls = [m for m in ch.messages if m.get("role") == "assistant" and m.get("tool_calls")]
        assert len(assistants_with_calls) == 1
        assert assistants_with_calls[0]["tool_calls"][0]["id"] == "call_Y"

        # get_copy must keep the well-formed pair intact (no orphan removal,
        # no synthetic insertion needed because we already have a real tool
        # message adjacent to its parent).
        clean = ch.get_copy()
        roles = [m["role"] for m in clean]
        assert "assistant" in roles and "tool" in roles
        # Assistant with tool_calls must be immediately followed by its tool.
        asst_idx = next(i for i, m in enumerate(clean) if m.get("role") == "assistant" and m.get("tool_calls"))
        assert clean[asst_idx + 1]["role"] == "tool"
        assert clean[asst_idx + 1]["tool_call_id"] == "call_Y"

    @pytest.mark.asyncio
    async def test_subsequent_llm_turn_sees_the_interrupted_attempt(self):
        """End-to-end: after a cancelled call, the next LLM-bound history
        copy shows the interrupted record. The LLM has evidence the call
        happened and won't blindly re-emit."""
        ch = ConversationHistory()
        ch.append_user("track my order")
        ch.append_assistant("एक मिनट दीजिए", turn_id=1)

        tm = _build_task_manager_mock(ch)

        async def hanging_trigger_api(**kwargs):
            await asyncio.sleep(10)

        model_response = [{"id": "call_Z", "function": {"name": "advanced_track_order"}}]

        with patch("bolna.agent_manager.task_manager.trigger_api", side_effect=hanging_trigger_api):
            with patch("bolna.agent_manager.task_manager.convert_to_request_log"):
                task = asyncio.create_task(
                    TaskManager._TaskManager__execute_function_call(
                        tm,
                        url="https://api.example.com/tools",
                        method="post",
                        param=None,
                        api_token=None,
                        headers=None,
                        model_args={},
                        meta_info={"sequence_id": 1, "turn_id": 1, "bypass_synth": False},
                        next_step=None,
                        called_fun="advanced_track_order",
                        model_response=model_response,
                        tool_call_id="call_Z",
                        textual_response=None,
                    )
                )
                await asyncio.sleep(0.05)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task

        # Simulate the next user turn appending a new user message.
        ch.append_user("हाँ कोई बात नहीं")
        # And the LLM-bound copy
        messages = ch.get_copy()
        # Verify the interrupted attempt is preserved in the history sent to
        # the next LLM invocation.
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert "interrupted_by_user" in tool_msgs[0]["content"]
        # The assistant.tool_calls field is preserved as well.
        asst_msgs = [m for m in messages if m.get("role") == "assistant" and m.get("tool_calls")]
        assert len(asst_msgs) == 1
        assert asst_msgs[0]["tool_calls"][0]["id"] == "call_Z"
