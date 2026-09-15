"""A graph node may name its own conversation LLM — the model that actually speaks.

The graph editor's "LLM overrides" accordion writes node.llm_config
{model, provider, temperature, max_tokens, reasoning_effort}. Before this existed the field was an
unknown key that Pydantic dropped on save, and the engine only ever read one agent-level model.
Unset fields inherit; a node that overrides nothing reuses the agent's own LLM instance.
"""

import os
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

from bolna.agent_types.graph_agent import GraphAgent
from bolna.llms import LiteLLM
from bolna.models import GraphNode

ENV = {"OPENAI_API_KEY": "platform-openai-key"}
PROVIDERS = ["openai", "azure", "google", "groq"]


def _registry(real=None):
    def for_provider(name):
        def make(**kwargs):
            client = MagicMock()
            client.captured_kwargs = kwargs
            client.generate_stream = MagicMock()
            return client

        return make

    reg = {p: for_provider(p) for p in PROVIDERS}
    reg.update(real or {})
    return reg


@contextmanager
def _agent(real_registry=None, **config_overrides):
    config = {
        "agent_information": "Test agent",
        "model": "gpt-4.1-mini",
        "provider": "openai",
        "llm_key": "conv-key",
        "base_url": "https://conv.example",
        "temperature": 0.7,
        "max_tokens": 150,
        "current_node_id": "start",
        "nodes": [{"id": "start", "prompt": "hi", "edges": []}],
    }
    config.update(config_overrides)
    with (
        patch.dict(os.environ, ENV, clear=True),
        patch("bolna.agent_types.graph_agent.SUPPORTED_LLM_PROVIDERS", _registry(real_registry)),
        patch("bolna.agent_types.graph_agent.OpenAiLLM", side_effect=lambda **kw: MagicMock()),
    ):
        yield GraphAgent(config)


# ------------------------------------------------------------------ schema


def test_the_node_schema_accepts_and_keeps_the_override():
    node = GraphNode(
        id="n1",
        prompt="hi",
        llm_config={"model": "gpt-5", "provider": "openai", "temperature": 0.2, "reasoning_effort": "high"},
    )
    assert node.llm_config.model == "gpt-5"
    assert node.llm_config.reasoning_effort.value == "high"
    # The whole point of the bug: it must survive a round trip, not be dropped as an extra key.
    assert node.model_dump()["llm_config"]["model"] == "gpt-5"


def test_a_node_without_an_override_dumps_none():
    assert GraphNode(id="n1", prompt="hi").model_dump()["llm_config"] is None


# ------------------------------------------------------------------ resolution


def test_a_node_without_an_override_reuses_the_agent_llm():
    with _agent() as agent:
        assert agent._conversation_llm_for({"id": "n1"}) is agent.llm
        assert agent._conversation_llm_for({"id": "n1", "llm_config": None}) is agent.llm
        # All-None override is the same as no override — the UI sends this after a reset.
        assert agent._conversation_llm_for({"id": "n1", "llm_config": {"model": None}}) is agent.llm
        assert agent._conversation_llm_cache == {}


def test_a_model_override_builds_a_separate_llm_inheriting_agent_credentials():
    with _agent() as agent:
        llm = agent._conversation_llm_for({"id": "n1", "llm_config": {"model": "gpt-5"}})
        assert llm is not agent.llm
        kwargs = llm.captured_kwargs
        assert kwargs["model"] == "gpt-5"
        assert kwargs["provider"] == "openai"
        assert kwargs["llm_key"] == "conv-key"
        assert kwargs["base_url"] == "https://conv.example"
        # Unset fields inherit the agent values.
        assert kwargs["temperature"] == 0.7
        assert kwargs["max_tokens"] == 150


def test_temperature_and_max_tokens_override_independently():
    with _agent() as agent:
        llm = agent._conversation_llm_for({"id": "n1", "llm_config": {"temperature": 0.0, "max_tokens": 900}})
        assert llm.captured_kwargs["temperature"] == 0.0
        assert llm.captured_kwargs["max_tokens"] == 900
        # Model still inherited.
        assert llm.captured_kwargs["model"] == "gpt-4.1-mini"


def test_a_zero_temperature_override_is_not_treated_as_unset():
    with _agent() as agent:
        llm = agent._conversation_llm_for({"id": "n1", "llm_config": {"temperature": 0}})
        assert llm.captured_kwargs["temperature"] == 0


# ------------------------------------------------------------------ effort


def test_an_effort_override_reaches_a_reasoning_model():
    with _agent() as agent:
        llm = agent._conversation_llm_for({"id": "n1", "llm_config": {"model": "gpt-5", "reasoning_effort": "high"}})
        assert llm.captured_kwargs["reasoning_effort"] == "high"


def test_an_inherited_effort_is_dropped_for_a_non_reasoning_override_model():
    # The agent runs a reasoning model with an effort; the node drops to a plain model, which
    # would reject the inherited effort.
    with _agent(model="gpt-5", reasoning_effort="high") as agent:
        llm = agent._conversation_llm_for({"id": "n1", "llm_config": {"model": "gpt-4.1-mini"}})
        assert "reasoning_effort" not in llm.captured_kwargs


def test_an_effort_enum_is_serialized_to_its_value():
    from bolna.enums import ReasoningEffort

    with _agent() as agent:
        llm = agent._conversation_llm_for(
            {"id": "n1", "llm_config": {"model": "gpt-5", "reasoning_effort": ReasoningEffort.MEDIUM}}
        )
        assert llm.captured_kwargs["reasoning_effort"] == "medium"


# ------------------------------------------------------------------ provider


def test_switching_provider_does_not_carry_the_agents_credentials():
    with _agent() as agent:
        llm = agent._conversation_llm_for({"id": "n1", "llm_config": {"provider": "groq", "model": "llama-3.3"}})
        assert llm.captured_kwargs["provider"] == "groq"
        # Another provider cannot authenticate with openai's key/base_url.
        assert "llm_key" not in llm.captured_kwargs
        assert "base_url" not in llm.captured_kwargs


def test_an_unknown_provider_falls_back_to_the_agents():
    with _agent() as agent:
        llm = agent._conversation_llm_for({"id": "n1", "llm_config": {"provider": "nope", "model": "x"}})
        assert llm.captured_kwargs["provider"] == "openai"


def test_a_litellm_provider_qualifies_the_override_model():
    captured = {}

    def make_litellm(**kwargs):
        captured.update(kwargs)
        return MagicMock()

    litellm_cls = MagicMock(side_effect=make_litellm)
    with patch("bolna.agent_types.graph_agent.LiteLLM", litellm_cls):
        with _agent(real_registry={"groq": litellm_cls}) as agent:
            agent._conversation_llm_for({"id": "n1", "llm_config": {"provider": "groq", "model": "llama-3.3"}})
    assert captured["model"] == "groq/llama-3.3"


# ------------------------------------------------------------------ caching


def test_the_override_is_cached_and_shared_across_identical_nodes():
    with _agent() as agent:
        a = agent._conversation_llm_for({"id": "n1", "llm_config": {"model": "gpt-5"}})
        b = agent._conversation_llm_for({"id": "n2", "llm_config": {"model": "gpt-5"}})
        c = agent._conversation_llm_for({"id": "n3", "llm_config": {"model": "gpt-4o"}})
        assert a is b
        assert c is not a
        assert len(agent._conversation_llm_cache) == 2


def test_the_same_model_at_two_efforts_is_two_clients():
    with _agent() as agent:
        low = agent._conversation_llm_for({"id": "n1", "llm_config": {"model": "gpt-5", "reasoning_effort": "low"}})
        high = agent._conversation_llm_for({"id": "n2", "llm_config": {"model": "gpt-5", "reasoning_effort": "high"}})
        assert low is not high
        assert low.captured_kwargs["reasoning_effort"] == "low"
        assert high.captured_kwargs["reasoning_effort"] == "high"


def test_the_cache_is_bounded():
    with _agent() as agent:
        agent._conversation_llm_cache_max_size = 2
        for m in ("m1", "m2", "m3"):
            agent._conversation_llm_for({"id": m, "llm_config": {"model": m}})
        assert len(agent._conversation_llm_cache) == 2


def test_an_unconverted_pydantic_override_is_tolerated():
    # Nodes reach the agent as dicts, but a shallow dict() of a validated config leaves this one
    # nested model intact. Raising here would kill the turn, not just skip the override.
    from bolna.models import GraphNodeLlmOverride

    with _agent() as agent:
        llm = agent._conversation_llm_for({"id": "n1", "llm_config": GraphNodeLlmOverride(model="gpt-5")})
        assert llm.captured_kwargs["model"] == "gpt-5"


# ------------------------------------------------------------------ end to end through generate()


def _streaming_agent(node):
    """Real GraphAgent whose provider registry yields async-streaming clients."""

    def make(**kwargs):
        client = MagicMock()
        client.captured_kwargs = kwargs
        client.stream_calls = []

        async def generate_stream(messages, **kw):
            client.stream_calls.append(messages)
            yield {"data": "spoken", "end_of_llm_stream": True}

        client.generate_stream = generate_stream
        return client

    return {p: make for p in PROVIDERS}


def _prepare(agent, node):
    agent.config["nodes"] = [node]
    agent.current_node_id = node["id"]
    agent._base_url_validated = True
    agent._should_hold_for_first_delivery = MagicMock(return_value=False)
    agent._build_messages = AsyncMock(return_value=[{"role": "user", "content": "hi"}])
    agent._get_tool_choice_for_node = MagicMock(return_value=None)
    agent._tools_for_node = MagicMock(return_value=[])
    agent.decide_next_node_with_functions = AsyncMock(return_value=(None, None, 0.0, None, None, None, None, None))
    agent.llm.generate_stream = MagicMock(side_effect=AssertionError("agent LLM must not be used"))


async def _drain(agen):
    return [chunk async for chunk in agen]


async def test_generate_streams_from_the_node_override_not_the_agent_llm():
    node = {"id": "n1", "node_type": "llm", "prompt": "hi", "edges": [], "llm_config": {"model": "gpt-5"}}
    with _agent(real_registry=_streaming_agent(node)) as agent:
        _prepare(agent, node)
        chunks = await _drain(agent.generate([{"role": "user", "content": "hi"}], meta_info={}))

    override = agent._conversation_llm_for(node)
    assert override.captured_kwargs["model"] == "gpt-5"
    assert len(override.stream_calls) == 1  # the override actually served the turn
    assert any(c.get("data") == "spoken" for c in chunks)


async def test_the_event_triggered_path_also_uses_the_override():
    # Separate call site in generate(); a fix applied to only one of the two would pass the test above.
    node = {"id": "n1", "node_type": "llm", "prompt": "hi", "edges": [], "llm_config": {"model": "gpt-5"}}
    with _agent(real_registry=_streaming_agent(node)) as agent:
        _prepare(agent, node)
        agent._event_triggered_generation = True
        agent.context_data["_last_event"] = "webhook"
        await _drain(agent.generate([{"role": "user", "content": "hi"}], meta_info={}))

    assert len(agent._conversation_llm_for(node).stream_calls) == 1


async def test_a_node_without_an_override_still_streams_from_the_agent_llm():
    node = {"id": "n1", "node_type": "llm", "prompt": "hi", "edges": []}
    with _agent(real_registry=_streaming_agent(node)) as agent:
        _prepare(agent, node)
        streamed = []

        async def agent_stream(messages, **kw):
            streamed.append(messages)
            yield {"data": "spoken", "end_of_llm_stream": True}

        agent.llm.generate_stream = agent_stream
        await _drain(agent.generate([{"role": "user", "content": "hi"}], meta_info={}))

        assert len(streamed) == 1
        assert agent._conversation_llm_cache == {}
