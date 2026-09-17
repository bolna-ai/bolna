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


def test_a_provider_switch_is_refused_rather_than_silently_rekeyed():
    # A node carries no credentials, so the only options are the agent's key against the wrong
    # provider, or the platform env default — which would move a BYOK customer onto our account.
    with _agent() as agent:
        llm = agent._conversation_llm_for({"id": "n1", "llm_config": {"provider": "groq", "model": "llama-3.3"}})
        assert llm is agent.llm
        assert agent._conversation_llm_cache == {}


def test_a_model_override_on_the_agents_own_provider_is_served():
    with _agent() as agent:
        llm = agent._conversation_llm_for({"id": "n1", "llm_config": {"provider": "openai", "model": "gpt-5"}})
        assert llm is not agent.llm
        assert llm.captured_kwargs["llm_key"] == "conv-key"


def test_an_unknown_provider_falls_back_to_the_agents():
    with _agent() as agent:
        llm = agent._conversation_llm_for({"id": "n1", "llm_config": {"provider": "nope", "model": "x"}})
        assert llm.captured_kwargs["provider"] == "openai"


def test_a_litellm_model_override_is_qualified_from_the_bare_name():
    # Same provider as the agent, so it is served; the inherited model already carries a prefix and
    # re-qualifying from it would keep dispatching to the old backend.
    captured = {}

    def make_litellm(**kwargs):
        captured.update(kwargs)
        return MagicMock()

    litellm_cls = MagicMock(side_effect=make_litellm)
    with patch("bolna.agent_types.graph_agent.LiteLLM", litellm_cls):
        with _agent(real_registry={"groq": litellm_cls}, provider="groq", model="groq/llama-3.1-8b") as agent:
            agent._conversation_llm_for({"id": "n1", "llm_config": {"model": "llama-3.3-70b"}})
    assert captured["model"] == "groq/llama-3.3-70b"


def test_an_inherited_prefixed_model_is_not_double_prefixed():
    captured = {}

    def make_litellm(**kwargs):
        captured.update(kwargs)
        return MagicMock()

    litellm_cls = MagicMock(side_effect=make_litellm)
    with patch("bolna.agent_types.graph_agent.LiteLLM", litellm_cls):
        with _agent(real_registry={"groq": litellm_cls}, provider="groq", model="groq/llama-3.1-8b") as agent:
            agent._conversation_llm_for({"id": "n1", "llm_config": {"temperature": 0.1}})
    assert captured["model"] == "groq/llama-3.1-8b"


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


# ------------------------------------------------------------------ transport follows the model


def test_a_responses_only_override_model_gets_the_responses_transport():
    # Agent on a chat-completions model, node on a Responses-API one. Inheriting the agent's flag
    # put the override on chat completions with an effort and transition tools.
    with _agent(model="gpt-4.1-mini") as agent:
        llm = agent._conversation_llm_for({"id": "n1", "llm_config": {"model": "gpt-5.4-mini"}})
        assert llm.captured_kwargs.get("use_responses_api") is True


def test_a_chat_only_override_model_does_not_inherit_the_responses_transport():
    # The reverse: a gpt-4.1 node under a Responses agent would otherwise join a
    # previous_response_id chain it cannot serve.
    with _agent(model="gpt-5.4", use_responses_api=True) as agent:
        llm = agent._conversation_llm_for({"id": "n1", "llm_config": {"model": "gpt-4.1-mini"}})
        assert "use_responses_api" not in llm.captured_kwargs


def test_an_override_that_keeps_the_model_keeps_the_agents_transport():
    with _agent(model="gpt-5.4", use_responses_api=True) as agent:
        llm = agent._conversation_llm_for({"id": "n1", "llm_config": {"temperature": 0.1}})
        assert llm.captured_kwargs.get("use_responses_api") is True


# ------------------------------------------------------------------ construction safety


def test_a_failed_override_construction_falls_back_to_the_agent_llm():
    # _initialize_llm has a fallback; this path runs inline in generate(), which re-raises and
    # would end the call.
    def explode_on_the_override(**kwargs):
        # Only the override model fails; the agent and routing clients must still build.
        if kwargs.get("model") == "gpt-5.4-mini":
            raise RuntimeError("azure_endpoint is None")
        client = MagicMock()
        client.captured_kwargs = kwargs
        return client

    with _agent(real_registry={"openai": explode_on_the_override}) as agent:
        llm = agent._conversation_llm_for({"id": "n1", "llm_config": {"model": "gpt-5.4-mini"}})
        assert llm is agent.llm
        assert agent._conversation_llm_cache == {}


# ------------------------------------------------------------------ lifecycle


async def test_overrides_are_closed_on_teardown():
    closed = []

    def make(**kwargs):
        client = MagicMock()
        client.captured_kwargs = kwargs
        client.close = AsyncMock(side_effect=lambda: closed.append(kwargs.get("model")))
        return client

    with _agent(real_registry={"openai": make}) as agent:
        agent._conversation_llm_for({"id": "n1", "llm_config": {"model": "gpt-5"}})
        agent._conversation_llm_for({"id": "n2", "llm_config": {"model": "gpt-4o"}})
        assert len(agent._conversation_llm_cache) == 2
        await agent.close_conversation_llm_overrides()

    assert sorted(closed) == ["gpt-4o", "gpt-5"]
    assert agent._conversation_llm_cache == {}


def test_overrides_are_prewarmed_at_graph_load():
    # Constructing opens a socket in Responses mode; doing it lazily puts that in a first-token path.
    nodes = [
        {"id": "a", "prompt": "hi", "edges": [], "llm_config": {"model": "gpt-5"}},
        {"id": "b", "prompt": "hi", "edges": [], "llm_config": {"model": "gpt-5"}},  # same -> shared
        {"id": "c", "prompt": "hi", "edges": [], "llm_config": {"model": "gpt-4o"}},
        {"id": "d", "prompt": "hi", "edges": []},  # no override
    ]
    with _agent(nodes=nodes, current_node_id="a") as agent:
        assert len(agent._conversation_llm_cache) == 2
        # Prewarm must not leave a half-resolved node as the active client.
        assert agent.current_conversation_llm() is agent.llm


def test_the_active_client_follows_the_node():
    with _agent() as agent:
        override = agent._conversation_llm_for({"id": "n1", "llm_config": {"model": "gpt-5"}})
        assert agent.current_conversation_llm() is override
        agent._conversation_llm_for({"id": "n2"})
        assert agent.current_conversation_llm() is agent.llm


# ------------------------------------------------------------------ real client, real transport


def _real_agent(**overrides):
    """No mock registry — SUPPORTED_LLM_PROVIDERS builds an actual OpenAiLLM per node."""
    config = {
        "agent_information": "Test agent",
        "model": "gpt-4.1-mini",
        "provider": "openai",
        "current_node_id": "start",
        "nodes": [{"id": "start", "prompt": "hi", "edges": []}],
    }
    config.update(overrides)
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True):
        yield GraphAgent(config)


async def test_a_real_override_client_picks_the_transport_from_its_own_model():
    # The mocked registry can't see this: every factory there is a MagicMock, so the transport
    # decision that use_responses_api drives is invisible. Async because a Responses-mode client
    # opens its socket at construction and needs a running loop.
    for agent in _real_agent():
        chat = agent._conversation_llm_for({"id": "n1", "llm_config": {"model": "gpt-4.1-mini", "temperature": 0.1}})
        responses = agent._conversation_llm_for({"id": "n2", "llm_config": {"model": "gpt-5.4-mini"}})

        assert chat is not agent.llm and responses is not agent.llm
        assert getattr(chat, "use_responses_api", False) is False
        assert getattr(responses, "use_responses_api", False) is True
        # And the agent's own client is untouched by either.
        assert getattr(agent.llm, "use_responses_api", False) is False
