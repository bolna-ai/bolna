"""Barge-in truncation: history must record what the caller actually heard, not what was generated.

A trim is addressed by turn_id or response_uid so a late barge-in can never rewrite an older
assistant turn — a missing or unmatched address skips the trim rather than guessing. When nothing
was heard the assistant turn is removed outright, along with the tool result it would otherwise
orphan, so the next LLM request carries no record of speech the caller never received.
"""

from bolna.enums import ChatRole
from bolna.helpers.conversation_history import ConversationHistory

FULL = "We are open from nine to five, Monday through Friday."


def _heard_prefix(original, heard):
    """Stand-in for the production update_fn: keep only what the caller heard."""
    return heard or ""


def _history():
    history = ConversationHistory([{"role": ChatRole.SYSTEM, "content": "sys"}])
    history.append_user("what are your hours")
    history.append_assistant(FULL, turn_id=4, response_uid="r4")
    return history


def _contents(history):
    return [m.get("content") for m in history.get_copy()]


def test_trim_by_turn_id_keeps_only_the_heard_prefix():
    history = _history()
    history.sync_turn_after_interruption(4, "We are open from nine", _heard_prefix)
    assert _contents(history) == ["sys", "what are your hours", "We are open from nine"]


def test_trim_by_response_uid_keeps_only_the_heard_prefix():
    history = _history()
    history.sync_response_after_interruption("r4", "We are open", _heard_prefix)
    assert _contents(history) == ["sys", "what are your hours", "We are open"]


def test_trim_without_a_turn_id_leaves_history_alone():
    """An unaddressed turn is exactly what a None lookup would match, so the guard must refuse.

    The staged assistant placeholder carries no turn_id, so without the guard a None address
    trims that placeholder instead of declining.
    """
    history = _history()
    history.append_assistant("staged, not yet addressed")
    before = _contents(history)

    history.sync_turn_after_interruption(None, "We are open", _heard_prefix)

    assert _contents(history) == before


def test_trim_without_a_response_uid_leaves_history_alone():
    history = _history()
    history.append_assistant("staged, not yet addressed")
    before = _contents(history)

    history.sync_response_after_interruption(None, "We are open", _heard_prefix)

    assert _contents(history) == before


def test_trim_for_an_unmatched_turn_leaves_history_alone():
    history = _history()
    history.sync_turn_after_interruption(99, "We are open", _heard_prefix)
    assert _contents(history) == ["sys", "what are your hours", FULL]


def test_nothing_heard_removes_the_assistant_turn():
    history = _history()
    history.sync_turn_after_interruption(4, "", _heard_prefix)
    assert _contents(history) == ["sys", "what are your hours"]


def test_nothing_heard_takes_the_orphaned_tool_result_with_it():
    """Leaving the tool result behind would strand it with no assistant turn to answer."""
    history = ConversationHistory([{"role": ChatRole.SYSTEM, "content": "sys"}])
    history.append_user("book me in")
    history.append_assistant("Let me check that.", tool_calls=[{"id": "t1"}], turn_id=7, response_uid="r7")
    history.append_tool_result("t1", "slot found")

    history.sync_turn_after_interruption(7, "", _heard_prefix)

    assert _contents(history) == ["sys", "book me in"]


def test_trim_last_assistant_needs_no_address():
    history = _history()
    history.sync_after_interruption("We are open from", _heard_prefix)
    assert _contents(history) == ["sys", "what are your hours", "We are open from"]


def test_interim_trim_does_not_touch_the_committed_history():
    """The interim list tracks the in-flight turn; the real transcript must not move with it."""
    history = _history()
    history.sync_interim()

    history.sync_interim_turn_after_interruption(4, "We are open", _heard_prefix)

    assert [m.get("content") for m in history.interim] == ["sys", "what are your hours", "We are open"]
    assert _contents(history) == ["sys", "what are your hours", FULL]
