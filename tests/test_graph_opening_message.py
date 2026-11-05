"""The line a call opens with, resolved by bolna and by the call-setup pre-render.

On a call whose graph config sets static_start_opening, a graph agent whose start node is static opens
with that node's own message and keeps that node's settings from the first second; every other call
opens with its welcome message.
"""

from unittest.mock import AsyncMock, MagicMock, patch

from bolna.agent_manager.assistant_manager import AssistantManager
from bolna.agent_manager.task_manager import TaskManager
from bolna.agent_types.graph_agent import GraphAgent
from bolna.helpers.utils import graph_opening_language, graph_opening_message, graph_opening_node


WELCOME = "Hello from the welcome message."
BILINGUAL = {"en": "Hello there.", "hi": "नमस्ते।"}


def _task(nodes, current_node_id, agent_type="graph_agent", static_start_opening=True, transcriber=None, **llm):
    llm_config = {"current_node_id": current_node_id, "nodes": nodes, **llm}
    if static_start_opening:
        llm_config["static_start_opening"] = True
    return {
        "tools_config": {
            "transcriber": transcriber or {"language": "hi"},
            "llm_agent": {"agent_type": agent_type, "llm_config": llm_config},
        }
    }


def _agent_config(nodes, current_node_id, welcome=WELCOME, **task_kwargs):
    return {"agent_welcome_message": welcome, "tasks": [_task(nodes, current_node_id, **task_kwargs)]}


def _static(node_id, message, **fields):
    return {"id": node_id, "node_type": "static", "static_message": message, "edges": [], **fields}


def _llm(node_id):
    return {"id": node_id, "prompt": "Greet the caller.", "edges": []}


class TestStaticStartNode:
    def test_static_start_node_message_opens_the_call(self):
        cfg = _agent_config([_static("greeting", "Am I speaking with Priya?")], "greeting")
        assert graph_opening_message(cfg) == "Am I speaking with Priya?"

    def test_without_static_start_opening_the_welcome_opens_the_call(self):
        cfg = _agent_config([_static("greeting", "Am I speaking with Priya?")], "greeting", static_start_opening=False)
        assert graph_opening_message(cfg) == WELCOME

    def test_language_variant_matches_the_call_language(self):
        for language, expected in (("hi", "नमस्ते।"), ("en", "Hello there."), ("ta", "Hello there.")):
            cfg = _agent_config([_static("greeting", BILINGUAL)], "greeting", transcriber={"language": language})
            assert graph_opening_message(cfg) == expected

    def test_regional_and_multilingual_codes_read_the_base_language(self):
        for language in ("hi-IN", "multi-hi"):
            cfg = _agent_config([_static("greeting", BILINGUAL)], "greeting", transcriber={"language": language})
            assert graph_opening_message(cfg) == "नमस्ते।"

    def test_a_multilingual_call_opens_in_its_starting_language(self):
        transcriber = {"language": "en", "active": "hi"}
        cfg = _agent_config([_static("greeting", BILINGUAL)], "greeting", transcriber=transcriber)
        assert graph_opening_message(cfg) == "नमस्ते।"

    def test_the_language_call_setup_pinned_wins(self):
        task = _task([_static("greeting", BILINGUAL)], "greeting", opening_language="en")
        assert graph_opening_language(task) == "en"
        assert graph_opening_message({"tasks": [task]}) == "Hello there."

    def test_empty_static_message_falls_back_to_the_welcome(self):
        cfg = _agent_config([_static("greeting", "")], "greeting")
        assert graph_opening_message(cfg) == WELCOME

    def test_static_start_node_opens_the_call_without_a_welcome(self):
        cfg = _agent_config([_static("greeting", "Am I speaking with Priya?")], "greeting", welcome="")
        assert graph_opening_message(cfg) == "Am I speaking with Priya?"


class TestOtherAgents:
    def test_llm_start_node_opens_with_the_welcome(self):
        cfg = _agent_config([_llm("greeting")], "greeting")
        assert graph_opening_message(cfg) == WELCOME

    def test_router_start_node_opens_with_the_welcome(self):
        router = {"id": "route", "node_type": "router", "edges": [{"to_node_id": "greeting"}]}
        cfg = _agent_config([router, _static("greeting", "Static line.")], "route")
        assert graph_opening_message(cfg) == WELCOME

    def test_non_graph_agent_opens_with_the_welcome(self):
        cfg = _agent_config([_static("greeting", "Static line.")], "greeting", agent_type="simple_llm_agent")
        assert graph_opening_message(cfg) == WELCOME

    def test_missing_start_node_falls_back_to_the_welcome(self):
        cfg = _agent_config([_static("greeting", "Static line.")], "gone")
        assert graph_opening_message(cfg) == WELCOME

    def test_config_without_tasks_falls_back_to_the_welcome(self):
        assert graph_opening_message({"agent_welcome_message": WELCOME}) == WELCOME

    def test_config_without_a_welcome_opens_with_nothing(self):
        assert graph_opening_message({}) == ""


class TestCallOpening:
    """The resolved line is what the call actually opens with, variables and all."""

    def test_call_opens_with_the_static_start_node_message(self):
        cfg = _agent_config([_static("greeting", {"hi": "नमस्ते {name} जी"})], "greeting")
        manager = AssistantManager(cfg, context_data={"recipient_data": {"name": "Priya"}})
        assert manager.kwargs["agent_welcome_message"] == "नमस्ते Priya जी"

    def test_call_opens_with_the_welcome_without_static_start_opening(self):
        cfg = _agent_config([_static("greeting", "Static line.")], "greeting", static_start_opening=False)
        manager = AssistantManager(cfg, context_data={"recipient_data": {}})
        assert manager.kwargs["agent_welcome_message"] == WELCOME

    def test_call_opens_with_the_welcome_for_an_llm_start_node(self):
        cfg = _agent_config([_llm("greeting")], "greeting")
        manager = AssistantManager(cfg, context_data={"recipient_data": {}})
        assert manager.kwargs["agent_welcome_message"] == WELCOME


def _task_manager(task, welcome_sent=True):
    tm = TaskManager.__new__(TaskManager)
    tm.task_config = task
    tm.repeat_after_silence_seconds = None
    tm.default_number_of_words_for_interruption = 3
    tm.number_of_words_for_interruption = 3
    tm.interruption_manager = MagicMock()
    tm.hangup_triggered = False
    tm._end_call_in_progress = False
    tm.conversation_ended = False
    tm.tools = {
        "input": MagicMock(welcome_message_played=MagicMock(return_value=True)),
        "output": MagicMock(get_welcome_message_sent_ts=MagicMock(return_value=1.0 if welcome_sent else None)),
    }
    return tm


class TestOpeningNodeSettings:
    """The node that opened the call keeps its own settings from the opening."""

    def test_opening_node_arms_its_silence_repeat_and_barge_in_threshold(self):
        node = _static(
            "greeting", "Am I speaking with Priya?", repeat_after_silence_seconds=6, number_of_words_for_interruption=0
        )
        tm = _task_manager(_task([node], "greeting"))
        tm._arm_opening_node()
        assert tm.opening_node is node
        assert tm.repeat_after_silence_seconds == 6
        assert tm.number_of_words_for_interruption == 0
        assert tm.interruption_manager.number_of_words_for_interruption == 0

    def test_without_static_start_opening_nothing_is_armed(self):
        node = _static("greeting", "Am I speaking with Priya?", repeat_after_silence_seconds=6)
        tm = _task_manager(_task([node], "greeting", static_start_opening=False))
        tm._arm_opening_node()
        assert tm.opening_node is None
        assert tm.repeat_after_silence_seconds is None
        assert tm.number_of_words_for_interruption == 3

    def test_a_start_node_with_nothing_to_say_does_not_open_the_call(self):
        assert graph_opening_node(_task([_static("greeting", "")], "greeting")) is None


async def _welcome_played(tm):
    tm.transcriber_message = "hello"
    tm.handle_accumulated_message_task = None
    with (
        patch.object(TaskManager, "_TaskManager__hang_up_after_reply", new_callable=AsyncMock) as hang_up,
        patch.object(TaskManager, "_TaskManager__send_first_message", new_callable=AsyncMock) as forward,
    ):
        await tm._TaskManager__handle_accumulated_message()
    return hang_up, forward


def _end_call_opening(**kwargs):
    node = _static("bye", "This is a reminder, goodbye.", function_call="end_call")
    tm = _task_manager(_task([node], "bye"), **kwargs)
    tm._arm_opening_node()
    return tm


class TestOpeningNodeEndsCall:
    async def test_an_end_call_opening_node_hangs_up_once_it_has_played(self):
        hang_up, forward = await _welcome_played(_end_call_opening())
        hang_up.assert_awaited_once()
        forward.assert_not_awaited()

    async def test_a_hangup_already_under_way_keeps_its_own_reason(self):
        tm = _end_call_opening()
        tm.hangup_triggered = True
        hang_up, _ = await _welcome_played(tm)
        hang_up.assert_not_awaited()

    async def test_an_opening_that_sent_no_audio_leaves_the_node_to_speak(self):
        hang_up, forward = await _welcome_played(_end_call_opening(welcome_sent=False))
        hang_up.assert_not_awaited()
        forward.assert_awaited_once_with("hello")

    async def test_an_opening_node_with_transitions_hands_the_caller_on(self):
        node = _static("greeting", "Am I speaking with Priya?", function_call="end_call")
        node["edges"] = [{"to_node_id": "next", "condition": "confirmed"}]
        tm = _task_manager(_task([node], "greeting"))
        tm._arm_opening_node()
        hang_up, forward = await _welcome_played(tm)
        hang_up.assert_not_awaited()
        forward.assert_awaited_once_with("hello")


def _graph_agent(**config):
    cfg = {"agent_information": "t", "model": "gpt-4o-mini", "current_node_id": "greeting", "nodes": [], **config}
    with (
        patch("bolna.agent_types.graph_agent.SUPPORTED_LLM_PROVIDERS", {"openai": MagicMock(return_value=MagicMock())}),
        patch("bolna.agent_types.graph_agent.OpenAiLLM", return_value=MagicMock()),
    ):
        return GraphAgent(cfg)


class TestRepromptLanguage:
    def test_a_static_reprompt_keeps_the_opening_language_until_the_caller_is_heard(self):
        chunk = _graph_agent(opening_language="hi")._static_message_chunk(_static("greeting", BILINGUAL))
        assert chunk["static_message"] == "नमस्ते।"

    def test_without_an_opening_language_a_static_node_reads_english(self):
        chunk = _graph_agent()._static_message_chunk(_static("greeting", BILINGUAL))
        assert chunk["static_message"] == "Hello there."
