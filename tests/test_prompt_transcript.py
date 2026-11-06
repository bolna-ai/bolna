"""The language_switch_prompt_transcript flag sends the earlier turns as prompt text right after a switch.

Some models keep the language of the chat history over every language instruction, so after a switch
they reply in the old language. The earlier turns go into the system prompt as a transcript and only
the caller's latest turn stays a message. The LLM request changes only: history is untouched, and
graph agents keep routing on the full history."""

import types
from unittest.mock import AsyncMock, MagicMock, patch

from bolna.agent_manager.task_manager import TaskManager
from bolna.agent_types.graph_agent import GraphAgent
from bolna.constants import LANGUAGE_SWITCH_TRANSCRIPT_REPLIES
from bolna.helpers.conversation_history import ConversationHistory
from bolna.helpers.prompt_transcript import CONTINUE_LINE, with_prompt_transcript

PROMPT = "## Agent Prompt:\n\nBe helpful.\n\n## Transcript:\n\n\n## Language note:\nreply in English"
FLAG = {"language_switch_prompt_transcript": True}


def history():
    return [
        {"role": "system", "content": PROMPT},
        {"role": "assistant", "content": "வணக்கம், நான் நவ்யா."},
        {"role": "user", "content": "ஆமாங்க."},
        {"role": "assistant", "content": "இத பத்தி சொல்லட்டுமா?"},
        {"role": "user", "content": "Yeah tell me about it."},
    ]


# --- the transcript itself ---


def test_earlier_turns_move_into_the_prompt_above_the_language_note():
    messages = history()

    sent = with_prompt_transcript(messages)

    assert [m["role"] for m in sent] == ["system", "user"]
    assert sent[1] == messages[-1]
    prompt = sent[0]["content"]
    assert "## Transcript:\nAgent: வணக்கம், நான் நவ்யா.\nCaller: ஆமாங்க.\nAgent: இத பத்தி சொல்லட்டுமா?\n" in prompt
    assert prompt.index(CONTINUE_LINE) < prompt.index("## Language note:")
    assert prompt.endswith("## Language note:\nreply in English")
    assert messages == history()  # the input is not modified


def test_an_agent_turn_keeps_its_words_and_its_tool_call():
    messages = history()
    messages[3] = {
        "role": "assistant",
        "content": "ஒரு நிமிஷம், check பண்றேன்.",
        "tool_calls": [{"id": "t0", "function": {"name": "lookup_order", "arguments": '{"id": "7"}'}}],
    }
    messages[4:4] = [{"role": "tool", "tool_call_id": "t0", "content": '{"status": "shipped"}'}]

    prompt = with_prompt_transcript(messages)[0]["content"]

    assert (
        'Agent: ஒரு நிமிஷம், check பண்றேன்.\nAgent called: lookup_order({"id": "7"})\nTool result: {"status": "shipped"}\n'
    ) in prompt


def test_tool_turns_after_the_callers_turn_stay_messages():
    messages = history() + [
        {"role": "assistant", "content": None, "tool_calls": [{"id": "t1", "function": {"name": "transfer"}}]},
        {"role": "tool", "tool_call_id": "t1", "content": "{}"},
    ]
    assert with_prompt_transcript(messages)[1:] == messages[-3:]


def test_a_system_message_inside_the_history_is_not_put_in_the_callers_mouth():
    messages = history()
    messages[2:2] = [{"role": "system", "content": "Knowledge base: four levels."}]
    assert "Context: Knowledge base: four levels.\n" in with_prompt_transcript(messages)[0]["content"]


def test_without_a_transcript_section_it_goes_above_the_language_note():
    messages = history()
    messages[0] = {"role": "system", "content": "Be helpful.\n\n## Language note:\nreply in English"}

    prompt = with_prompt_transcript(messages)[0]["content"]

    assert prompt.startswith("Be helpful.\n\n## Transcript:\nAgent: வணக்கம்")
    assert prompt.endswith("## Language note:\nreply in English")


def test_without_any_section_it_is_appended():
    messages = history()
    messages[0] = {"role": "system", "content": "Be helpful."}
    assert with_prompt_transcript(messages)[0]["content"].startswith("Be helpful.\n\n## Transcript:\nAgent:")


def test_a_non_text_caller_turn_or_system_prompt_is_left_alone():
    messages = history()
    messages[-1] = {"role": "user", "content": [{"type": "text", "text": "hi"}]}
    assert with_prompt_transcript(messages) is messages

    messages = history()
    messages[0] = {"role": "system", "content": [{"type": "text", "text": PROMPT}]}
    assert with_prompt_transcript(messages) is messages


# --- when a reply gets it ---


def holder(flag=True, switcher=True):
    tm = MagicMock()
    tm.task_config = {"tools_config": FLAG if flag else {}}
    tm.language_switcher = MagicMock() if switcher else None
    tm.conversation_history = ConversationHistory(initial_history=history())
    tm.language_switch_history_start = None
    return tm


def switched(tm):
    tm.language_switch_history_start = len(tm.conversation_history.messages)
    return tm


def due(tm, speculative=False):
    return TaskManager.prompt_transcript_due(tm, speculative)


def test_off_without_the_flag_a_switcher_or_a_switch():
    assert not due(switched(holder(flag=False)))
    assert not due(switched(holder(switcher=False)))
    assert not due(holder())


def test_the_speculative_reply_gets_it_before_the_switch_lands():
    assert due(holder(), speculative=True)
    assert not due(holder(flag=False), speculative=True)


def test_it_lasts_until_the_switch_is_followed_by_enough_agent_replies():
    tm = switched(holder())
    h = tm.conversation_history
    # A tool call and its result are not replies the caller heard in the new language.
    h.append_assistant(None, tool_calls=[{"id": "t1", "function": {"name": "lookup"}}])
    h.append_tool_result("t1", "{}")
    assert due(tm)

    for n in range(LANGUAGE_SWITCH_TRANSCRIPT_REPLIES):
        assert due(tm)
        h.append_assistant(f"Reply {n} in English.")
        h.append_user("ok")

    assert not due(tm)


def test_it_stops_if_the_history_shrank_below_the_switch_point():
    tm = switched(holder())
    tm.language_switch_history_start += 5
    assert not due(tm)


# --- the request that reaches the LLM ---


def generation_tm(agent, agent_type):
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
    tm.task_config = {"task_type": "conversation", "tools_config": {**FLAG, "llm_agent": {"agent_type": agent_type}}}
    tm.language_switcher = MagicMock()
    tm.language_switch_history_start = 0
    tm._inject_language_instruction = lambda messages: messages
    tm._apply_node_interruption_threshold = lambda node: None
    tm._synthesize = AsyncMock()
    return tm


async def generate_once(tm, messages):
    meta_info = {"turn_id": 2, "sequence_id": 3, "response_uid": "r1"}
    with patch("bolna.agent_manager.task_manager.convert_to_request_log"):
        await TaskManager._TaskManager__do_llm_generation_impl(tm, messages, meta_info, "synthesizer")
    return meta_info


async def test_a_reply_after_a_switch_gets_the_transcript_and_history_keeps_every_turn():
    captured = {}

    async def generate(messages, **kwargs):
        captured["messages"] = messages
        yield {"static_message": "Sure, English it is.", "static_audio_hash": "hash"}

    agent = MagicMock()
    agent.generate = generate
    tm = generation_tm(agent, "simple_llm_agent")
    messages = history()

    await generate_once(tm, messages)

    assert [m["role"] for m in captured["messages"]] == ["system", "user"]
    assert "Caller: ஆமாங்க." in captured["messages"][0]["content"]
    assert messages[: len(history())] == history()  # every turn stays a message in history


def graph_agent():
    node = {"id": "pitch", "node_type": "llm", "prompt": "Explain the loyalty program.", "edges": []}
    cfg = {"agent_information": "You are Navya.", "model": "gpt-4o-mini", "current_node_id": "pitch", "nodes": [node]}
    with (
        patch("bolna.agent_types.graph_agent.SUPPORTED_LLM_PROVIDERS", {"openai": MagicMock(return_value=MagicMock())}),
        patch("bolna.agent_types.graph_agent.OpenAiLLM", return_value=MagicMock()),
    ):
        agent = GraphAgent(cfg)
    agent.context_data = {"detected_language": "en"}
    return agent


async def test_a_graph_agent_routes_on_the_full_history_and_its_node_prompt_gets_the_transcript():
    captured = {}
    agent = graph_agent()

    async def generate(messages, synthesize=True, meta_info=None):
        captured["routed_on"] = messages
        captured["sent"] = await agent._build_messages(messages, meta_info=meta_info)
        yield {"static_message": "Sure.", "static_audio_hash": "hash"}

    agent.generate = generate
    tm = generation_tm(agent, "graph_agent")
    messages = history()

    meta_info = await generate_once(tm, messages)

    assert meta_info["prompt_transcript"] is True
    assert captured["routed_on"][: len(history())] == history()
    sent = captured["sent"]
    assert [m["role"] for m in sent] == ["system", "user"]
    prompt = sent[0]["content"]
    assert "## Conversation History" not in prompt
    assert "Agent: வணக்கம், நான் நவ்யா.\nCaller: ஆமாங்க.\n" in prompt
    assert prompt.index("## Transcript:") < prompt.index("LANGUAGE GUIDELINES")


async def test_a_graph_agent_without_the_window_sends_the_history_as_messages():
    agent = graph_agent()

    sent = await agent._build_messages(history(), meta_info={"prompt_transcript": False})

    assert sent[0]["content"].endswith("## Conversation History")
    assert [m["role"] for m in sent] == ["system", "assistant", "user", "assistant", "user"]


async def test_the_speculative_reply_is_asked_with_the_transcript_and_history_is_untouched():
    captured = {}

    async def generate(messages, synthesize=False, meta_info=None):
        captured["messages"] = messages
        yield types.SimpleNamespace(data="Sure.", end_of_stream=True, is_function_call=False)

    h = ConversationHistory(initial_history=[{"role": "system", "content": "base prompt"}])
    h.append_assistant("வணக்கம்")
    h.append_user("कैन यू स्पीक इन इंग्लिश")
    tm = holder()
    tm.conversation_history = h
    tm.multilingual_prompts = {"en": "English prompt\n\n## Transcript:\n", "hi": "Hindi prompt"}
    tm._TaskManager__language_directive = TaskManager._TaskManager__language_directive.__get__(tm, TaskManager)
    tm.prompt_transcript_due = TaskManager.prompt_transcript_due.__get__(tm, TaskManager)
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
