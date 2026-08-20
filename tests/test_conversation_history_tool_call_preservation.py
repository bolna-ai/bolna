"""Regression tests for BLT-018 — a tool call being emitted/executed twice.

Root cause: attach_tool_calls_to_turn creates a uid-less assistant *placeholder* carrying the
tool_calls; if the heard-audio materialization (sync_history -> upsert_assistant_for_response)
then can't find that turn by response_uid, it used to APPEND a duplicate plain-text assistant.
That duplicate lands between the tool-call assistant and its tool result, so _sanitize_tool_messages
(run inside get_copy() before every LLM request) strips the tool_calls and drops the tool result.
The follow-up generation then sees no record the tool ran and re-issues it.

The fix: upsert_assistant_for_response adopts the uid-less placeholder for the same turn_id
(filling content + response_uid, preserving tool_calls) instead of duplicating the turn.
"""

from bolna.helpers.conversation_history import ConversationHistory
from bolna.enums import ChatRole


TOOL_CALLS = [
    {
        "id": "call_1",
        "type": "function",
        "function": {"name": "reschedule_call", "arguments": '{"mentionedTime":"later"}'},
    }
]


def _assistant_tool_call_pair_intact(llm_messages):
    """True iff some assistant carries tool_calls AND is immediately followed by a tool
    message for each of its call ids (exactly what OpenAI requires and what _sanitize keeps)."""
    for i, m in enumerate(llm_messages):
        if m.get("role") in (ChatRole.ASSISTANT, "assistant") and m.get("tool_calls"):
            expected = {tc["id"] for tc in m["tool_calls"]}
            got = set()
            j = i + 1
            while j < len(llm_messages) and llm_messages[j].get("role") in (ChatRole.TOOL, "tool"):
                got.add(llm_messages[j].get("tool_call_id"))
                j += 1
            if expected.issubset(got):
                return True
    return False


def test_tool_call_survives_heard_audio_materialization_race():
    # BLT-018 repro: staged text dropped -> attach makes a uid-less placeholder ->
    # heard-audio upsert runs BEFORE the tool result is appended.
    h = ConversationHistory([{"role": ChatRole.SYSTEM, "content": "sys"}])
    h.append_user("please reschedule")
    h.attach_tool_calls_to_turn(8, TOOL_CALLS)  # placeholder: content=None, tool_calls, turn 8, no uid
    h.upsert_assistant_for_response("uid8", "Your call has been successfully rescheduled.", turn_id=8)
    h.append_tool_result("call_1", "ok")

    llm = h.get_copy()
    assert _assistant_tool_call_pair_intact(llm), (
        "tool_calls + tool result must survive sanitize so the model doesn't re-issue the tool call"
    )
    # And exactly one assistant for turn 8 (no duplicate materialized).
    assert sum(1 for m in h.messages if m.get("role") == ChatRole.ASSISTANT and m.get("turn_id") == 8) == 1


def test_clean_ordering_preserves_tool_call():
    # Sanity: the non-racy ordering was never broken and still isn't.
    h = ConversationHistory([{"role": ChatRole.SYSTEM, "content": "sys"}])
    h.append_user("please reschedule")
    h.append_assistant("Your call has been successfully rescheduled.", turn_id=8, response_uid="uid8")
    h.attach_tool_calls_to_turn(8, TOOL_CALLS)
    h.append_tool_result("call_1", "ok")
    h.upsert_assistant_for_response("uid8", "Your call has been successfully rescheduled.", turn_id=8)

    assert _assistant_tool_call_pair_intact(h.get_copy())


def test_uidless_placeholder_adopted_not_duplicated():
    # The placeholder is filled in place: content + response_uid set, tool_calls preserved.
    h = ConversationHistory()
    h.attach_tool_calls_to_turn(8, TOOL_CALLS)
    h.upsert_assistant_for_response("uid8", "heard text", turn_id=8)

    assistants = [m for m in h.messages if m.get("role") == ChatRole.ASSISTANT]
    assert len(assistants) == 1
    assert assistants[0]["content"] == "heard text"
    assert assistants[0]["response_uid"] == "uid8"
    assert assistants[0].get("tool_calls") == TOOL_CALLS


def test_separate_response_with_own_uid_not_clobbered():
    # The turn_id fallback must only adopt a *uid-less* placeholder — a real, distinct response
    # that already owns a response_uid in the same turn must not be overwritten/merged.
    h = ConversationHistory()
    h.append_assistant("first response", turn_id=5, response_uid="a")
    h.upsert_assistant_for_response("b", "second response", turn_id=5)

    assistants = [m for m in h.messages if m.get("role") == ChatRole.ASSISTANT]
    assert len(assistants) == 2
    assert [a["content"] for a in assistants] == ["first response", "second response"]
    assert [a["response_uid"] for a in assistants] == ["a", "b"]
