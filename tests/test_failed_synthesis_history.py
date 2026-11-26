"""A reply whose synthesis failed partway keeps in history only the text the synthesizer rendered, whether the reply
was already committed when the failure arrived or the LLM was still writing it."""

from bolna.agent_manager.task_manager import TaskManager
from bolna.helpers.conversation_history import ConversationHistory

FULL = "First sentence. Second sentence. Third sentence."
META = {"sequence_id": 5, "turn_id": 3, "response_uid": "uid-1"}


def _task_manager():
    history = ConversationHistory()
    history.append_user("hello")
    tm = TaskManager.__new__(TaskManager)
    tm.conversation_history = history
    tm._turn_msg_map = {}
    tm._last_spoken_assistant = None
    tm._last_spoken_user_input = None
    tm._pending_assistant_history = {}
    tm._pending_user_input = None
    tm._response_captures = {5: object()}
    tm._sent_audio_sequences = {5}
    tm._blocked_sequences = set()
    tm._committed_assistant_sequences = set()
    tm._rendered_before_failure = {}
    return tm, history


def _fail(tm, rendered):
    tm._TaskManager__keep_rendered_text({**META, "synthesis_failed": True, "text_rendered": rendered})


def _assistant(history):
    return [m["content"] for m in history.messages if m["role"] == "assistant"]


def test_a_committed_reply_is_cut_to_the_rendered_text():
    tm, history = _task_manager()
    tm._stage_assistant_history(META, FULL)
    _fail(tm, "First sentence.")
    assert _assistant(history) == ["First sentence."]
    assert tm._last_spoken_assistant == (3, "First sentence.")
    assert 5 not in tm._response_captures


def test_a_reply_the_llm_finishes_after_the_failure_is_committed_cut():
    tm, history = _task_manager()
    _fail(tm, "First sentence.")
    tm._stage_assistant_history(META, FULL)
    assert _assistant(history) == ["First sentence."]
    assert tm._last_spoken_assistant == (3, "First sentence.")


def test_requests_joined_with_single_spaces_still_match_the_reply():
    tm, history = _task_manager()
    _fail(tm, "First sentence. Second sentence.")
    tm._stage_assistant_history(META, "First sentence.\nSecond  sentence. Third sentence.")
    assert _assistant(history) == ["First sentence.\nSecond  sentence."]


def test_a_reply_none_of_whose_requests_ended_is_not_committed():
    tm, history = _task_manager()
    _fail(tm, "")
    tm._stage_assistant_history(META, FULL)
    assert _assistant(history) == []
    assert tm._last_spoken_assistant is None
