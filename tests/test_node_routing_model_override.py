"""A graph node may name its own routing model; it runs on the agent's routing provider and
credentials. Nodes without one keep using the single agent-level routing LLM, so the default
path is unchanged.
"""

import os
import time
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

from bolna.agent_types.graph_agent import GraphAgent
from bolna.constants import default_reasoning_effort
from bolna.llms import LiteLLM

ENV = {"OPENAI_API_KEY": "platform-openai-key"}
PROVIDERS = ["openai", "azure", "google", "groq"]


def _registry(real=None):
    def for_provider(name):
        def make(**kwargs):
            client = MagicMock()
            client.captured_kwargs = kwargs
            client.route = AsyncMock(return_value=None)
            return client

        return make

    reg = {p: for_provider(p) for p in PROVIDERS}
    reg.update(real or {})
    return reg


@contextmanager
def _agent(real_registry=None, **config_overrides):
    """Registry stays patched for the whole block: overrides are built lazily, after __init__."""
    config = {
        "agent_information": "Test agent",
        "model": "gpt-4.1-mini",
        "provider": "openai",
        "llm_key": "conv-key",
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


# ------------------------------------------------------------------ resolution


def test_a_node_without_an_override_uses_the_agent_routing_llm_itself():
    with _agent() as agent:
        llm, model, effort = agent._routing_llm_for({"id": "n"})
        assert llm is agent.routing_llm
        assert (model, effort) == (agent.routing_model, agent._routing_reasoning_effort_used)


def test_an_override_builds_a_separate_llm_on_the_agents_provider_and_credentials():
    with _agent() as agent:
        llm, model, _ = agent._routing_llm_for({"id": "verify", "routing_model": "gpt-4.1"})
        assert llm is not agent.routing_llm
        assert model == "gpt-4.1"
        kw = llm.captured_kwargs
        base = agent.routing_llm.captured_kwargs
        assert kw["model"] == "gpt-4.1"
        assert kw["provider"] == base["provider"] == "openai"
        assert kw["llm_key"] == base["llm_key"] == "conv-key"  # creds ride along, never per node
        assert kw["temperature"] == 0 and kw["max_tokens"] == base["max_tokens"]


def test_the_override_is_cached_per_model_and_shared_across_nodes():
    with _agent() as agent:
        a, _, _ = agent._routing_llm_for({"id": "n1", "routing_model": "gpt-4.1"})
        b, _, _ = agent._routing_llm_for({"id": "n2", "routing_model": "gpt-4.1"})
        c, _, _ = agent._routing_llm_for({"id": "n3", "routing_model": "gpt-4o"})
        assert a is b
        assert c is not a
        assert set(agent._routing_llm_cache) == {"gpt-4.1", "gpt-4o"}


def test_the_cache_is_bounded():
    with _agent() as agent:
        agent._routing_llm_cache_max_size = 2
        for m in ("m1", "m2", "m3"):
            agent._routing_llm_for({"id": m, "routing_model": m})
        assert len(agent._routing_llm_cache) == 2
        assert "m1" not in agent._routing_llm_cache  # oldest evicted


def test_a_gpt5_override_gets_a_reasoning_effort_and_a_non_gpt5_one_does_not():
    with _agent() as agent:  # agent routes on gpt-4.1-mini: no effort at agent level
        assert agent._routing_reasoning_effort_used is None
        five, _, effort5 = agent._routing_llm_for({"id": "a", "routing_model": "gpt-5-mini"})
        four, _, effort4 = agent._routing_llm_for({"id": "b", "routing_model": "gpt-4o"})
        assert five.captured_kwargs["reasoning_effort"] == effort5 == default_reasoning_effort("gpt-5-mini")
        assert "reasoning_effort" not in four.captured_kwargs and effort4 is None


def test_an_agent_level_effort_is_inherited_by_a_gpt5_override():
    with _agent(routing_model="gpt-5-mini", routing_reasoning_effort="low") as agent:
        llm, _, effort = agent._routing_llm_for({"id": "a", "routing_model": "gpt-5"})
        assert llm.captured_kwargs["reasoning_effort"] == effort == "low"


def test_a_litellm_provider_qualifies_the_override_model():
    with _agent(real_registry={"groq": LiteLLM}, routing_provider="groq", routing_model="llama-3.1-8b") as agent:
        assert agent.routing_model == "groq/llama-3.1-8b"
        _, model, _ = agent._routing_llm_for({"id": "a", "routing_model": "llama-3.3-70b"})
        assert model == "groq/llama-3.3-70b"


# ------------------------------------------------------------------ the routing call and telemetry


def _intent_node(node_id, **extra):
    return {
        "id": node_id,
        "prompt": "p",
        "edges": [{"to_node_id": "next", "condition": "user wants next", "function_name": "go_next"}],
        **extra,
    }


async def test_routing_uses_the_nodes_override_and_reports_it():
    with _agent() as agent:
        node = _intent_node("verify", routing_model="gpt-4.1")
        await agent._decide_next_node_llm(
            node, node["edges"], [{"role": "user", "content": "next"}], time.perf_counter()
        )

        override_llm = agent._routing_llm_cache["gpt-4.1"]
        assert override_llm.route.await_count == 1
        assert agent.routing_llm.route.await_count == 0
        assert agent._last_routing_model == "gpt-4.1"

        hop = agent._router_hop_info(
            "verify",
            routing_type="llm",
            latency_ms=1.0,
            started_at=0.0,
            reasoning=None,
            confidence=None,
            routing_messages=[{"role": "system", "content": "x"}],
        )
        assert hop["routing_model"] == "gpt-4.1"
        assert hop["routing_provider"] == "openai"


async def test_a_node_without_an_override_still_routes_on_the_agent_llm():
    with _agent() as agent:
        node = _intent_node("plain")
        await agent._decide_next_node_llm(
            node, node["edges"], [{"role": "user", "content": "next"}], time.perf_counter()
        )
        assert agent.routing_llm.route.await_count == 1
        assert agent._routing_llm_cache == {}
        assert agent._last_routing_model == agent.routing_model


def test_a_turn_that_makes_no_routing_call_reports_the_agent_model():
    with _agent() as agent:
        agent._last_routing_model, agent._last_routing_effort = "gpt-4.1", "low"  # left over from an override
        agent._reset_routing_identity()
        assert (agent._last_routing_model, agent._last_routing_effort) == (agent.routing_model, None)
