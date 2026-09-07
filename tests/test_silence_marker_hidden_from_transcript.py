"""The silence nudge is a control token for the LLM, not something the caller said.

_inject_and_run_llm appends "[silence] User was silent for N seconds" as a user turn so the
agent re-prompts. It has to stay in the LLM's history to do that job, but it must never surface
as a caller utterance — and the N it prints is the configured threshold, not a measured gap.
"""

from unittest.mock import AsyncMock, MagicMock, patch

from bolna.agent_manager.task_manager import TaskManager
from bolna.helpers.conversation_history import ConversationHistory
from bolna.helpers.utils import format_messages
from bolna.llms.message_models import INTERNAL_MESSAGE_KEYS, strip_internal_keys

MARKER = "[silence] User was silent for 12.0 seconds"


def _history_with_marker():
    h = ConversationHistory()
    h.append_assistant("Can you provide your Serial Number?")
    h.append_user(MARKER, exclude_from_transcript=True)
    h.append_assistant("Sure, take your time.")
    h.append_user("VI 2.6 37080.")
    return h


def test_the_marker_is_absent_from_the_transcript():
    transcript = format_messages(_history_with_marker().messages)
    assert MARKER not in transcript
    assert "[silence]" not in transcript
    assert "user: VI 2.6 37080." in transcript  # real caller speech is untouched


def test_the_llm_still_sees_the_marker():
    """Dropping it from history would remove the nudge that makes the agent re-prompt."""
    copy = _history_with_marker().get_copy()
    assert any(m.get("content") == MARKER for m in copy)


def test_the_marker_key_never_reaches_a_provider():
    sent = strip_internal_keys(_history_with_marker().get_copy())
    assert all("exclude_from_transcript" not in m for m in sent)
    assert "exclude_from_transcript" in INTERNAL_MESSAGE_KEYS


def test_an_ordinary_user_turn_is_never_filtered():
    h = ConversationHistory()
    h.append_user("I was silent for a while, sorry")
    assert "user: I was silent for a while, sorry" in format_messages(h.messages)


async def test_the_injection_marks_the_turn():
    tm = TaskManager.__new__(TaskManager)
    tm.conversation_history = ConversationHistory()
    tm.tools = {"output": MagicMock()}
    tm.task_config = {"tools_config": {"output": {"format": "pcm"}}}
    tm.response_in_pipeline = False
    tm._TaskManager__get_updated_meta_info = MagicMock(return_value={})
    tm._run_llm_task = AsyncMock()

    with patch("bolna.agent_manager.task_manager.asyncio.create_task", lambda coro: AsyncMock()()):
        await TaskManager._inject_and_run_llm(tm, MARKER)

    row = tm.conversation_history.messages[-1]
    assert row["content"] == MARKER
    assert row["exclude_from_transcript"] is True
    assert MARKER not in format_messages(tm.conversation_history.messages)
