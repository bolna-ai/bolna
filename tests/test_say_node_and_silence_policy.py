"""Static say nodes and the graph agent's silence-repeat policy, including the voice cache key."""

import hashlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bolna.models import GraphNode
from bolna.enums import NodeType
from bolna.agent_types.graph_agent import GraphAgent
from bolna.helpers.utils import get_md5_hash


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


class TestNodeType:
    def test_llm_equals_string(self):
        assert NodeType.LLM == "llm"

    def test_static_equals_string(self):
        assert NodeType.STATIC == "static"

    def test_default_is_llm(self):
        node = GraphNode(id="test")
        assert node.node_type == NodeType.LLM

    def test_static_from_string(self):
        node = GraphNode(id="test", node_type="static", static_message="Hello")
        assert node.node_type == NodeType.STATIC


class TestGraphNodeModel:
    def test_backward_compat_no_new_fields(self):
        node = GraphNode(id="test", prompt="hello")
        assert node.node_type == NodeType.LLM
        assert node.static_message is None
        assert node.repeat_after_silence_seconds is None

    def test_static_node_with_repeat(self):
        node = GraphNode(
            id="greeting",
            node_type=NodeType.STATIC,
            static_message="Hello! How can I help?",
            repeat_after_silence_seconds=8.0,
        )
        assert node.node_type == NodeType.STATIC
        assert node.static_message == "Hello! How can I help?"
        assert node.repeat_after_silence_seconds == 8.0

    def test_prompt_default_empty(self):
        node = GraphNode(id="test", node_type=NodeType.STATIC, static_message="Hi")
        assert node.prompt == ""

    def test_json_roundtrip(self):
        node = GraphNode(
            id="test",
            node_type=NodeType.STATIC,
            static_message="Test message",
            repeat_after_silence_seconds=5.0,
        )
        data = node.model_dump()
        assert data["node_type"] == "static"
        assert data["static_message"] == "Test message"
        assert data["repeat_after_silence_seconds"] == 5.0

        restored = GraphNode(**data)
        assert restored.node_type == NodeType.STATIC
        assert restored.static_message == "Test message"
        assert restored.repeat_after_silence_seconds == 5.0


# ---------------------------------------------------------------------------
# Graph Agent - Static Node Dispatch
# ---------------------------------------------------------------------------


def _async_iter(items):
    async def _gen():
        for item in items:
            yield item

    return _gen()


def _make_config(**overrides):
    defaults = {
        "agent_information": "Test agent",
        "model": "gpt-4o-mini",
        "provider": "openai",
        "temperature": 0.7,
        "max_tokens": 150,
        "current_node_id": "greeting",
        "nodes": [
            {
                "id": "greeting",
                "node_type": "static",
                "static_message": "Hello! Welcome to Acme Corp.",
                "repeat_after_silence_seconds": 8,
                "edges": [
                    {
                        "to_node_id": "sales",
                        "condition": "wants to buy",
                        "function_name": "go_to_sales",
                    },
                    {
                        "to_node_id": "goodbye",
                        "condition_type": "expression",
                        "expression": {
                            "conditions": [
                                {"variable": "_silence_repeats", "operator": "gte", "value": 3},
                            ],
                        },
                    },
                ],
            },
            {
                "id": "sales",
                "prompt": "Help the user with purchasing.",
                "edges": [],
            },
            {
                "id": "goodbye",
                "node_type": "static",
                "static_message": "Goodbye!",
                "edges": [],
            },
        ],
    }
    defaults.update(overrides)
    return defaults


def _make_agent(config_overrides=None):
    cfg = _make_config(**(config_overrides or {}))
    mock_llm = MagicMock()
    mock_llm.generate_stream = AsyncMock(return_value=_async_iter([]))
    mock_llm.trigger_function_call = False
    mock_openai_client = MagicMock()
    mock_openai_llm_cls = MagicMock(return_value=mock_llm)

    with (
        patch("bolna.agent_types.graph_agent.SUPPORTED_LLM_PROVIDERS", {"openai": mock_openai_llm_cls}),
        patch("bolna.agent_types.graph_agent.OpenAiLLM", return_value=MagicMock()),
    ):
        agent = GraphAgent(cfg)

    agent._mock_llm = mock_llm
    return agent


def _no_transition_routing():
    """A routing response with no tool call: the judge chose to stay on the node."""
    message = MagicMock()
    message.tool_calls = None
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    response.usage = None
    return response, False


@pytest.fixture(autouse=True)
def _routing_stays_offline():
    """Intent edges consult the routing LLM; expression edges are still decided locally."""
    with patch("asyncio.to_thread", new_callable=AsyncMock, return_value=_no_transition_routing()):
        yield


class TestStaticNodeDispatch:
    async def test_static_node_yields_static_message_and_hash(self):
        agent = _make_agent()
        assert agent.current_node_id == "greeting"

        history = [{"role": "user", "content": "hello"}]
        chunks = []
        async for chunk in agent.generate(history):
            chunks.append(chunk)

        routing_chunk = chunks[0]
        assert "routing_info" in routing_chunk
        assert routing_chunk["routing_info"]["node_type"] == NodeType.STATIC

        static_chunk = chunks[1]
        assert "static_message" in static_chunk
        assert static_chunk["static_message"] == "Hello! Welcome to Acme Corp."
        expected_hash = get_md5_hash("Hello! Welcome to Acme Corp.")
        assert static_chunk["static_audio_hash"] == expected_hash

        assert len(chunks) == 2

    async def test_static_node_routing_info_includes_node_type(self):
        agent = _make_agent()
        history = [{"role": "user", "content": "hello"}]

        async for chunk in agent.generate(history):
            if "routing_info" in chunk:
                assert chunk["routing_info"]["node_type"] == NodeType.STATIC
                break

    async def test_llm_node_routing_info_has_llm_type(self):
        agent = _make_agent({"current_node_id": "sales"})
        history = [{"role": "user", "content": "I want to buy"}]

        async for chunk in agent.generate(history):
            if "routing_info" in chunk:
                assert chunk["routing_info"]["node_type"] == NodeType.LLM
                break

    async def test_static_node_empty_message_yields_nothing(self):
        agent = _make_agent(
            {
                "current_node_id": "empty_static",
                "nodes": [
                    {
                        "id": "empty_static",
                        "node_type": "static",
                        "static_message": "",
                        "edges": [],
                    },
                ],
            }
        )
        history = [{"role": "user", "content": "hello"}]
        chunks = []
        async for chunk in agent.generate(history):
            chunks.append(chunk)

        assert len(chunks) == 1
        assert "routing_info" in chunks[0]

    async def test_static_node_with_transition(self):
        agent = _make_agent()
        history = [{"role": "user", "content": "hello"}]
        chunks = []
        async for chunk in agent.generate(history):
            chunks.append(chunk)

        routing = chunks[0]["routing_info"]
        assert routing["current_node"] == "greeting"


class TestSilenceRepeats:
    async def test_silence_trigger_increments_counter(self):
        agent = _make_agent()
        assert agent._silence_repeats == 0

        history = [{"role": "user", "content": "[silence] User was silent for 8 seconds"}]
        async for _ in agent.generate(history):
            pass

        assert agent._silence_repeats == 1

    async def test_silence_counter_resets_on_transition(self):
        """The new node gets a fresh silence budget, not the one the caller exhausted."""
        agent = _make_agent()
        agent._silence_repeats = 3  # trips greeting's `_silence_repeats gte 3` expression edge

        async for _ in agent.generate([{"role": "user", "content": "hello"}]):
            pass

        assert agent.current_node_id == "goodbye"
        assert agent._silence_repeats == 0

    async def test_silence_counter_survives_a_turn_that_stays_put(self):
        agent = _make_agent()
        agent._silence_repeats = 1

        async for _ in agent.generate([{"role": "user", "content": "hello"}]):
            pass

        assert agent.current_node_id == "greeting"
        assert agent._silence_repeats == 1

    async def test_silence_repeats_in_context_data(self):
        agent = _make_agent()
        agent._silence_repeats = 3

        history = [{"role": "user", "content": "hello"}]
        async for _ in agent.generate(history):
            pass

        assert agent.context_data.get("_silence_repeats") == 3

    async def test_silence_trigger_on_static_node_replays_message(self):
        agent = _make_agent()
        history = [{"role": "user", "content": "[silence] User was silent for 8 seconds"}]

        chunks = []
        async for chunk in agent.generate(history):
            chunks.append(chunk)

        assert any("static_message" in c for c in chunks)
        static_chunk = next(c for c in chunks if "static_message" in c)
        assert static_chunk["static_message"] == "Hello! Welcome to Acme Corp."

    async def test_routing_info_has_is_silence_trigger(self):
        agent = _make_agent()

        history = [{"role": "user", "content": "[silence] User was silent"}]
        async for chunk in agent.generate(history):
            if "routing_info" in chunk:
                assert chunk["routing_info"]["is_silence_trigger"] is True
                break

    async def test_normal_message_not_silence_trigger(self):
        agent = _make_agent()

        history = [{"role": "user", "content": "hello"}]
        async for chunk in agent.generate(history):
            if "routing_info" in chunk:
                assert chunk["routing_info"]["is_silence_trigger"] is False
                break

    async def test_expression_edge_transitions_on_silence_repeats(self):
        agent = _make_agent()
        agent._silence_repeats = 2

        history = [{"role": "user", "content": "[silence] User was silent for 8 seconds"}]
        chunks = []
        async for chunk in agent.generate(history):
            chunks.append(chunk)

        routing = chunks[0]["routing_info"]
        assert routing["transitioned"] is True
        assert routing["current_node"] == "goodbye"
        assert agent._silence_repeats == 0

    async def test_expression_edge_no_transition_below_threshold(self):
        agent = _make_agent()
        agent._silence_repeats = 1

        history = [{"role": "user", "content": "[silence] User was silent for 8 seconds"}]
        chunks = []
        async for chunk in agent.generate(history):
            chunks.append(chunk)

        routing = chunks[0]["routing_info"]
        assert routing["transitioned"] is False
        assert routing["current_node"] == "greeting"
        assert agent._silence_repeats == 2


class TestSilenceExitThreshold:
    """The re-prompts a node needs before its _silence_repeats edge fires; the task manager's cap never drops below it."""

    def test_reads_the_current_nodes_silence_edge(self):
        assert _make_agent().silence_exit_threshold() == 3  # greeting: `_silence_repeats gte 3`

    @pytest.mark.parametrize(
        "operator,value,needed",
        [("gte", 3, 3), ("eq", 2, 2), ("gt", 3, 4), ("gte", "4", 4), ("lt", 3, 0), ("neq", 1, 0)],
    )
    def test_operator_maps_to_reprompts_needed(self, operator, value, needed):
        agent = _make_agent()
        agent.get_node_by_id("greeting")["edges"][1]["expression"]["conditions"] = [
            {"variable": "_silence_repeats", "operator": operator, "value": value}
        ]
        assert agent.silence_exit_threshold() == needed

    def test_takes_the_highest_silence_edge_and_ignores_other_variables(self):
        agent = _make_agent()
        greeting = agent.get_node_by_id("greeting")
        greeting["edges"][1]["expression"]["conditions"].append(
            {"variable": "_node_turns", "operator": "gte", "value": 9}
        )
        greeting["edges"].append(
            {
                "to_node_id": "sales",
                "condition_type": "expression",
                "expression": {"conditions": [{"variable": "_silence_repeats", "operator": "gte", "value": 5}]},
            }
        )
        assert agent.silence_exit_threshold() == 5

    def test_a_node_without_a_silence_edge_needs_none(self):
        agent = _make_agent()
        agent.current_node_id = "sales"
        assert agent.silence_exit_threshold() == 0


class TestMd5Hash:
    def test_get_md5_hash_consistency(self):
        text = "Hello! Welcome to Acme Corp."
        h1 = get_md5_hash(text)
        h2 = get_md5_hash(text)
        assert h1 == h2
        assert h1 == hashlib.md5(text.encode()).hexdigest()

    def test_different_text_different_hash(self):
        assert get_md5_hash("hello") != get_md5_hash("goodbye")
