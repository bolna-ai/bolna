"""A cancelled end_call must leave no trace in history.

Prod replay of run 5c5df9f7 showed the model re-invokes end_call 0/8 times while the cancelled
tool call and its "Call is ending now" result remain, and 8/8 once they are dropped, so the
caller could not get the agent to hang up.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from bolna.agent_manager.task_manager import TaskManager
from bolna.enums import ChatRole
from bolna.helpers.conversation_history import ConversationHistory

TOOL_RESULT = '{"status": "success", "message": "Call is ending now. Say a brief goodbye to the user."}'


def _history_with_end_call(call_id="call_1", spoken=None):
    history = ConversationHistory()
    history.setup_system_prompt({"role": ChatRole.SYSTEM, "content": "You are Pooja."})
    history.append_user("no no thank you thank you thank you")
    history.append_assistant(spoken, turn_id=1)
    history.attach_tool_calls_to_turn(
        1, [{"id": call_id, "type": "function", "function": {"name": "end_call", "arguments": "{}"}}]
    )
    history.append_tool_result(call_id, TOOL_RESULT)
    return history


def test_drops_the_tool_call_and_its_result():
    history = _history_with_end_call()
    assert history.drop_tool_call("call_1") is True
    assert not any(m.get("tool_call_id") == "call_1" for m in history.messages)
    assert not any(m.get("tool_calls") for m in history.messages)


def test_keeps_a_goodbye_the_agent_actually_spoke():
    history = _history_with_end_call(spoken="Thank you, goodbye!")
    history.drop_tool_call("call_1")
    assert history.last_assistant_content() == "Thank you, goodbye!"
    assert not any(m.get("tool_calls") for m in history.messages)


def test_drops_the_stub_turn_that_only_carried_the_tool_call():
    history = _history_with_end_call(spoken=None)
    before = len(history.messages)
    history.drop_tool_call("call_1")
    assert len(history.messages) == before - 2


def test_leaves_a_sibling_tool_call_on_the_same_turn_alone():
    history = ConversationHistory()
    history.append_assistant(None, turn_id=1)
    history.attach_tool_calls_to_turn(1, [{"id": "a"}, {"id": "b"}])
    history.append_tool_result("a", "res-a")
    history.append_tool_result("b", "res-b")
    history.drop_tool_call("a")
    assert [c["id"] for m in history.messages for c in (m.get("tool_calls") or [])] == ["b"]
    assert any(m.get("tool_call_id") == "b" for m in history.messages)


def test_is_a_noop_for_an_unknown_or_blank_id():
    history = _history_with_end_call()
    assert history.drop_tool_call("") is False
    assert history.drop_tool_call("nope") is False
    assert history.drop_tool_call("call_1") is True
    assert history.drop_tool_call("call_1") is False


def test_repair_also_applies_to_the_interim_copy():
    history = _history_with_end_call()
    history.sync_interim(history.get_copy())
    history.drop_tool_call("call_1")
    assert not any(m.get("tool_call_id") == "call_1" for m in history.interim)


def _cancel_target(history):
    return SimpleNamespace(
        _hangup_interruptible_window=True,
        _hangup_cancelled=False,
        conversation_ended=False,
        conversation_config={"check_if_user_online": True},
        check_if_user_online=False,
        llm_task=None,
        _end_call_hangup_task=None,
        _end_call_tool_call_id="call_1",
        conversation_history=history,
        hangup_triggered=True,
        _end_call_in_progress=True,
        hangup_message_queued=True,
        hangup_triggered_at=1.0,
        hangup_decision_at=1.0,
        _hangup_processing=True,
        _end_of_conversation_in_progress=True,
        hangup_detail="END_CALL_TOOL",
    )


async def test_cancelling_the_hangup_repairs_history():
    history = _history_with_end_call(spoken="Thank you, goodbye!")
    target = _cancel_target(history)
    TaskManager._cancel_pending_hangup(target)

    assert not any(m.get("tool_calls") for m in history.messages)
    assert not any(m.get("tool_call_id") == "call_1" for m in history.messages)
    assert history.last_assistant_content() == "Thank you, goodbye!"
    assert target._end_call_tool_call_id is None


async def test_cancel_survives_a_missing_tool_call_id():
    target = _cancel_target(_history_with_end_call())
    target._end_call_tool_call_id = ""
    target.conversation_history = MagicMock()
    TaskManager._cancel_pending_hangup(target)
    assert target.hangup_triggered is False
