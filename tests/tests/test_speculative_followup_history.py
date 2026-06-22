"""Regression tests for __speculative_followup_text's history construction.

The speculation runs the main LLM over a deep COPY of history. It must mirror the real
switch path's history correction so the committed speculation matches what the real path
would have produced:
  * turn-boundary: replace the garbled (wrong-language) user turn — found content-guarded,
    like replace_last_user — with the unbiased detector transcript, even when the main
    reply already trails it, and even though an idle-flush would leave an OLDER user turn
    last. Never append a second consecutive user message (alternating-role LLMs reject it).
  * idle-flush (no active_transcript): append the detector transcript.
  * newer turn arrived (no content match): leave history untouched, answer the newest turn.
"""

import types
from unittest.mock import MagicMock

import pytest

from bolna.agent_manager.task_manager import TaskManager
from bolna.helpers.conversation_history import ConversationHistory


def _msg(data="", end=False, fc=False):
    return types.SimpleNamespace(data=data, end_of_stream=end, is_function_call=fc)


def _make_tm(history, generate):
    tm = MagicMock()
    tm.conversation_history = history
    tm.multilingual_prompts = {"en": "You are a helpful agent.", "hi": "Hindi prompt"}
    # Bind the real note builder so the speculative system prompt mirrors production.
    tm._TaskManager__switch_context_note = TaskManager._TaskManager__switch_context_note.__get__(tm, TaskManager)
    tm.tools = {"llm_agent": MagicMock()}
    tm.tools["llm_agent"].generate = generate
    return tm


def _spec(tm):
    return TaskManager._TaskManager__speculative_followup_text.__get__(tm, TaskManager)


def _capturing_generate(captured):
    async def generate(messages, synthesize=False, meta_info=None):
        captured["messages"] = messages
        captured["synthesize"] = synthesize
        yield _msg(data="reply text", end=True)

    return generate


@pytest.mark.asyncio
async def test_replaces_garbled_trailing_user_turn():
    captured = {}
    h = ConversationHistory(initial_history=[{"role": "system", "content": "base prompt"}])
    h.append_assistant("Hello, how can I help?")
    h.append_user("Kana Rohit Prakaran")  # garbled wrong-language ASR of the Tamil turn

    text = await _spec(_make_tm(h, _capturing_generate(captured)))(
        "en", "I want to talk in English", "Kana Rohit Prakaran"
    )

    assert text == "reply text"
    msgs = captured["messages"]
    assert captured["synthesize"] is False
    user_turns = [m for m in msgs if m.get("role") == "user"]
    assert len(user_turns) == 1
    assert user_turns[0]["content"] == "I want to talk in English"
    assert msgs[-1]["role"] == "user"  # ends on the user turn the LLM must answer
    # Clean replacement dict — no stale identity keys carried from the garbled turn.
    assert set(user_turns[0].keys()) == {"role", "content"}
    # Target-language directive installed into the system prompt.
    assert msgs[0]["role"] == "system"
    assert "You are a helpful agent." in msgs[0]["content"]
    assert "Language note" in msgs[0]["content"]
    # Real history is untouched (get_copy deep-copies).
    real_users = [m for m in h.get_copy() if m.get("role") == "user"]
    assert real_users[-1]["content"] == "Kana Rohit Prakaran"


@pytest.mark.asyncio
async def test_replaces_garbled_even_when_assistant_reply_already_trails():
    """The race finding: the main reply was committed before this snapshot, so the garbled
    user turn is no longer last. Content-guarded scan must still replace it (not append a
    polluting second user turn)."""
    captured = {}
    h = ConversationHistory(initial_history=[{"role": "system", "content": "base prompt"}])
    h.append_user("Kana Rohit Prakaran")  # garbled current turn
    h.append_assistant("Wrong-language reply that already played")  # main reply landed first

    await _spec(_make_tm(h, _capturing_generate(captured)))("en", "I want English", "Kana Rohit Prakaran")

    msgs = captured["messages"]
    user_turns = [m for m in msgs if m.get("role") == "user"]
    assert len(user_turns) == 1
    assert user_turns[0]["content"] == "I want English"
    # The garbled wrong-language text is gone entirely — no pollution.
    assert all("Kana Rohit Prakaran" != m.get("content") for m in msgs)
    # No two consecutive user turns anywhere.
    roles = [m.get("role") for m in msgs]
    assert not any(roles[i] == "user" and roles[i + 1] == "user" for i in range(len(roles) - 1))


@pytest.mark.asyncio
async def test_appends_when_idle_flush_no_active_transcript():
    """Idle-flush: no garbled turn was appended (active_transcript empty) → append, and do
    NOT overwrite the older legitimate user turn that happens to be last-of-its-role."""
    captured = {}
    h = ConversationHistory(initial_history=[{"role": "system", "content": "base prompt"}])
    h.append_user("an earlier, legitimate turn")
    h.append_assistant("the agent's earlier reply")

    await _spec(_make_tm(h, _capturing_generate(captured)))("en", "switch me to english", "")

    msgs = captured["messages"]
    user_turns = [m for m in msgs if m.get("role") == "user"]
    assert len(user_turns) == 2  # the old turn is preserved, the detector turn appended
    assert user_turns[0]["content"] == "an earlier, legitimate turn"
    assert user_turns[1]["content"] == "switch me to english"
    assert msgs[-1]["role"] == "user"


@pytest.mark.asyncio
async def test_newer_turn_during_decide_leaves_history_untouched():
    """A newer user turn arrived during the decide (content no longer matches the garbled
    turn) → leave history untouched and answer the newest turn, mirroring the real path's
    transcript_corrected=False branch. Crucially, do NOT append a second user turn."""
    captured = {}
    h = ConversationHistory(initial_history=[{"role": "system", "content": "base prompt"}])
    h.append_user("Kana Rohit Prakaran")  # the garbled turn that was switched on
    h.append_assistant("reply")
    h.append_user("a brand new question")  # newer turn landed during the decide

    await _spec(_make_tm(h, _capturing_generate(captured)))("en", "detector", "Kana Rohit Prakaran")

    msgs = captured["messages"]
    # detector_transcript was NOT injected; the newest turn stands.
    assert all(m.get("content") != "detector" for m in msgs)
    assert msgs[-1]["role"] == "user"
    assert msgs[-1]["content"] == "a brand new question"
    roles = [m.get("role") for m in msgs]
    assert not any(roles[i] == "user" and roles[i + 1] == "user" for i in range(len(roles) - 1))


@pytest.mark.asyncio
async def test_no_consecutive_user_turns_ever_sent():
    """The original defect was two back-to-back user messages; assert it can't happen."""
    captured = {}
    h = ConversationHistory(initial_history=[{"role": "system", "content": "base prompt"}])
    h.append_user("garbled turn")

    await _spec(_make_tm(h, _capturing_generate(captured)))("en", "detector transcript", "garbled turn")

    msgs = captured["messages"]
    roles = [m.get("role") for m in msgs]
    assert not any(roles[i] == "user" and roles[i + 1] == "user" for i in range(len(roles) - 1))


@pytest.mark.asyncio
async def test_aborts_and_returns_empty_on_function_call():
    """A function-call chunk must abort speculation and return "" — even if it carries text,
    that text must NOT leak into the reply (so the chunk carries non-empty data here, which
    would otherwise be accumulated if the abort branch were missing)."""

    async def generate(messages, synthesize=False, meta_info=None):
        yield _msg(data="leaked tool argument text", fc=True)

    h = ConversationHistory(initial_history=[{"role": "system", "content": "base prompt"}])
    h.append_user("garbled")

    text = await _spec(_make_tm(h, generate))("en", "detector text", "garbled")
    assert text == ""
