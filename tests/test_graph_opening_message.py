"""The line a call opens with, resolved by bolna and by the call-setup pre-render.

A graph agent whose start node is static opens with that node's own message, so the node is
heard instead of being routed past; every other agent opens with its welcome message.
"""

from bolna.agent_manager.assistant_manager import AssistantManager
from bolna.helpers.utils import graph_opening_message


WELCOME = "Hello from the welcome message."


def _agent_config(nodes, current_node_id, agent_type="graph_agent", welcome=WELCOME):
    return {
        "agent_welcome_message": welcome,
        "tasks": [
            {
                "tools_config": {
                    "transcriber": {"language": "hi"},
                    "llm_agent": {
                        "agent_type": agent_type,
                        "llm_config": {"current_node_id": current_node_id, "nodes": nodes},
                    },
                }
            }
        ],
    }


def _static(node_id, message):
    return {"id": node_id, "node_type": "static", "static_message": message, "edges": []}


def _llm(node_id):
    return {"id": node_id, "prompt": "Greet the caller.", "edges": []}


class TestStaticStartNode:
    def test_static_start_node_message_opens_the_call(self):
        cfg = _agent_config([_static("greeting", "Am I speaking with Priya?")], "greeting")
        assert graph_opening_message(cfg) == "Am I speaking with Priya?"

    def test_language_variant_matches_the_call_language(self):
        cfg = _agent_config([_static("greeting", {"en": "Hello there.", "hi": "नमस्ते।"})], "greeting")
        assert graph_opening_message(cfg, "hi") == "नमस्ते।"
        assert graph_opening_message(cfg, "en") == "Hello there."

    def test_unconfigured_language_falls_back_to_english(self):
        cfg = _agent_config([_static("greeting", {"en": "Hello there.", "hi": "नमस्ते।"})], "greeting")
        assert graph_opening_message(cfg, "ta") == "Hello there."

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

    def test_call_opens_with_the_welcome_for_an_llm_start_node(self):
        cfg = _agent_config([_llm("greeting")], "greeting")
        manager = AssistantManager(cfg, context_data={"recipient_data": {}})
        assert manager.kwargs["agent_welcome_message"] == WELCOME
