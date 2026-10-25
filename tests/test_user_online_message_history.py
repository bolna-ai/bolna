"""The "are you still there" nudge reaches the LLM's history only when the agent opts in."""

import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

from bolna.helpers.conversation_history import ConversationHistory
from bolna.models import ConversationConfig
from tests.test_check_completion_requires_mark_ack import _drive, _telephony_task_manager

NUDGE = "Hey, are you still there"


def silent_caller(in_llm_history):
    """A call where both sides have been quiet past the nudge trigger but short of the silence hangup."""
    tm = _telephony_task_manager(time.time() - 8, hang_conversation_after=60, age=8)
    tm.check_if_user_online = True
    tm.check_user_online_message_config = NUDGE
    tm.language = "en"
    tm.should_record = False
    tm.user_online_message_in_llm_history = in_llm_history
    tm.conversation_history = ConversationHistory()
    tm._synthesize = AsyncMock()
    tm._TaskManager__is_s2s = lambda: False
    tm.tools = {
        "input": SimpleNamespace(
            is_audio_being_played_to_user=lambda: False,
            reset_response_heard_by_user=lambda: None,
            update_is_audio_being_played=lambda *args: None,
        ),
        "output": SimpleNamespace(get_provider=lambda: "plivo", handle_interruption=AsyncMock()),
    }
    return tm


def nudges(messages):
    return [m for m in messages if m.get("content") == NUDGE]


async def test_the_nudge_stays_out_of_the_llm_history_by_default(monkeypatch):
    tm = silent_caller(in_llm_history=False)
    await _drive(tm, monkeypatch)

    assert tm.asked_if_user_is_still_there
    assert len(nudges(tm.conversation_history.messages)) == 1
    assert nudges(tm.conversation_history.get_copy()) == []


async def test_an_opted_in_agent_sends_the_nudge_to_the_llm(monkeypatch):
    tm = silent_caller(in_llm_history=True)
    await _drive(tm, monkeypatch)

    assert [m["role"] for m in nudges(tm.conversation_history.get_copy())] == ["assistant"]


def test_the_opt_in_survives_agent_validation():
    """The OSS server stores the validated config's model_dump(), so the key must be a declared field."""
    assert (
        ConversationConfig(user_online_message_in_llm_history=True).model_dump()["user_online_message_in_llm_history"]
        is True
    )
    assert ConversationConfig().model_dump()["user_online_message_in_llm_history"] is False
