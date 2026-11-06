"""The language_switch_turn_note flag sends the history as prompt text for the first replies after a switch.

Some models keep the language of the chat history over every language instruction, so after a switch
they reply in the old language. The earlier turns go into the system prompt's Transcript section and
only the caller's latest turn stays a message. The LLM request changes only: history is untouched."""

import types
from unittest.mock import AsyncMock, MagicMock, patch

from bolna.agent_manager.task_manager import TaskManager
from bolna.agent_types.graph_agent import GraphAgent
from bolna.constants import LANGUAGE_SWITCH_TRANSCRIPT_REPLIES
from bolna.helpers.conversation_history import ConversationHistory

PROMPT = "## Agent Prompt:\n\nBe helpful.\n\n## Transcript:\n\n\n## Language note:\nreply in English"
CONTINUE = "Continue it from exactly where it left off"


def holder(flag=True, switcher=True, replies=0):
    tm = MagicMock()
    tm.task_config = {"tools_config": {"language_switch_turn_note": True} if flag else {}}
    tm.language_switcher = MagicMock() if switcher else None
    tm.replies_since_language_switch = replies
    return tm


def transcript(tm, messages, speculative=False):
    return TaskManager._TaskManager__with_switch_transcript(tm, messages, speculative=speculative)


def history():
    return [
        {"role": "system", "content": PROMPT},
        {"role": "assistant", "content": "வணக்கம், நான் நவ்யா."},
        {"role": "user", "content": "ஆமாங்க."},
        {"role": "assistant", "content": "இத பத்தி சொல்லட்டுமா?"},
        {"role": "user", "content": "Yeah tell me about it."},
    ]


def test_flag_off_sends_the_messages_unchanged():
    messages = history()
    assert transcript(holder(flag=False), messages) is messages


def test_without_a_language_switcher_the_messages_are_unchanged():
    messages = history()
    assert transcript(holder(switcher=False), messages) is messages


def test_before_any_switch_the_messages_are_unchanged():
    messages = history()
    assert transcript(holder(replies=None), messages) is messages


def test_after_a_switch_earlier_turns_move_into_the_prompt():
    messages = history()

    sent = transcript(holder(), messages)

    assert [m["role"] for m in sent] == ["system", "user"]
    assert sent[1] == messages[-1]
    prompt = sent[0]["content"]
    assert "## Transcript:\nAgent: வணக்கம், நான் நவ்யா.\nCaller: ஆமாங்க.\nAgent: இத பத்தி சொல்லட்டுமா?\n" in prompt
    assert CONTINUE in prompt
    assert prompt.endswith("## Language note:\nreply in English")
    assert messages == history()  # the caller's list is not modified


def test_only_the_first_replies_after_a_switch_get_the_transcript():
    tm = holder()
    for _ in range(LANGUAGE_SWITCH_TRANSCRIPT_REPLIES):
        assert len(transcript(tm, history())) == 2

    messages = history()
    assert transcript(tm, messages) is messages
    assert tm.replies_since_language_switch == LANGUAGE_SWITCH_TRANSCRIPT_REPLIES


def test_the_speculative_reply_gets_it_without_using_up_the_window():
    tm = holder(replies=None)
    assert len(transcript(tm, history(), speculative=True)) == 2
    assert tm.replies_since_language_switch is None


def test_tool_turns_after_the_callers_turn_stay_messages_and_earlier_ones_become_text():
    messages = history()
    messages[2:2] = [
        {"role": "assistant", "content": None, "tool_calls": [{"id": "t0", "function": {"name": "lookup"}}]},
        {"role": "tool", "tool_call_id": "t0", "content": '{"tier": "gold"}'},
    ]
    messages += [
        {"role": "assistant", "content": None, "tool_calls": [{"id": "t1", "function": {"name": "transfer"}}]},
        {"role": "tool", "tool_call_id": "t1", "content": "{}"},
    ]

    sent = transcript(holder(), messages)

    assert sent[1:] == messages[-3:]
    assert 'Agent called: lookup\nTool result: {"tier": "gold"}\n' in sent[0]["content"]


def test_a_non_text_caller_turn_is_left_alone():
    messages = history()
    messages[-1] = {"role": "user", "content": [{"type": "text", "text": "hi"}]}
    assert transcript(holder(), messages) is messages


def test_a_prompt_without_a_transcript_section_gets_one():
    messages = history()
    messages[0] = {"role": "system", "content": "Be helpful."}

    prompt = transcript(holder(), messages)[0]["content"]

    assert prompt.startswith("Be helpful.\n\n## Transcript:\nAgent: வணக்கம், நான் நவ்யா.")


async def test_the_speculative_reply_is_asked_with_the_transcript_and_history_is_untouched():
    captured = {}

    async def generate(messages, synthesize=False, meta_info=None):
        captured["messages"] = messages
        yield types.SimpleNamespace(data="Sure.", end_of_stream=True, is_function_call=False)

    h = ConversationHistory(initial_history=[{"role": "system", "content": "base prompt"}])
    h.append_assistant("வணக்கம்")
    h.append_user("कैन यू स्पीक इन इंग्लिश")
    tm = holder(replies=None)
    tm.conversation_history = h
    tm.multilingual_prompts = {"en": "English prompt\n\n## Transcript:\n", "hi": "Hindi prompt"}
    tm._TaskManager__language_directive = TaskManager._TaskManager__language_directive.__get__(tm, TaskManager)
    tm._TaskManager__with_switch_transcript = TaskManager._TaskManager__with_switch_transcript.__get__(tm, TaskManager)
    tm.tools = {"llm_agent": MagicMock()}
    tm.tools["llm_agent"].generate = generate

    await TaskManager._TaskManager__speculative_followup_text(
        tm, "en", "Can you speak in English?", "कैन यू स्पीक इन इंग्लिश"
    )

    sent = captured["messages"]
    assert [m["role"] for m in sent] == ["system", "user"]
    assert sent[1]["content"] == "Can you speak in English?"
    assert "## Transcript:\nAgent: வணக்கம்\n" in sent[0]["content"]
    assert h.get_copy()[-1]["content"] == "कैन यू स्पीक इन इंग्लिश"


async def test_a_reply_after_a_switch_gets_the_transcript_but_history_keeps_every_turn():
    captured = {}

    async def generate(messages, **kwargs):
        captured["messages"] = messages
        yield {"static_message": "Sure, English it is.", "static_audio_hash": "hash"}

    agent = GraphAgent.__new__(GraphAgent)
    agent.current_node_id = "greet"
    agent.get_node_by_id = lambda node_id: None
    agent.generate = generate
    tm = TaskManager.__new__(TaskManager)
    tm.tools = {"input": MagicMock(), "llm_agent": agent}
    tm.conversation_history = ConversationHistory()
    tm.conversation_history.current_node = tm.graph_node
    tm._pending_assistant_history = {}
    tm._pending_user_input = None
    tm._sent_audio_sequences = set()
    tm._committed_assistant_sequences = set()
    tm._blocked_sequences = set()
    tm._last_spoken_assistant = None
    tm._last_spoken_user_input = None
    tm._turn_msg_map = {}
    tm.on_turn_usage = None
    tm.on_overflow = None
    tm.routing_latencies = {}
    tm._usage_tasks = set()
    tm.repeat_after_silence_seconds = None
    tm.hangup_triggered = False
    tm.conversation_ended = False
    tm.run_id = "run"
    tm.language = "en"
    tm.stream = True
    tm.turn_based_conversation = False
    tm.llm_config = {"model": "sarvam-105b-conversations", "provider": "sarvam"}
    tm.task_config = {"tools_config": {"language_switch_turn_note": True}}
    tm.language_switcher = MagicMock()
    tm.replies_since_language_switch = 0
    tm._inject_language_instruction = lambda messages: messages
    tm._apply_node_interruption_threshold = lambda node: None
    tm._synthesize = AsyncMock()
    messages = history()

    with patch("bolna.agent_manager.task_manager.convert_to_request_log"):
        await TaskManager._TaskManager__do_llm_generation_impl(
            tm, messages, {"turn_id": 2, "sequence_id": 3, "response_uid": "r1"}, "synthesizer"
        )

    assert [m["role"] for m in captured["messages"]] == ["system", "user"]
    assert tm.replies_since_language_switch == 1
    assert messages[: len(history())] == history()  # every turn stays a message in history
