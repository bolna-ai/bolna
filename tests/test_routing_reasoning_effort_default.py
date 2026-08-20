"""Routing reasoning_effort must be one the routing model actually accepts.

The default used to be a hardcoded "minimal", which only gpt-5, gpt-5-mini and gpt-5-nano
accept, so routing was rejected by the provider on gpt-5.1 and later. The gpt-5 family check
also read the raw model name, missing Azure deployment names entirely.
"""

import pytest
from unittest.mock import MagicMock, patch

from bolna.agent_types.graph_agent import GraphAgent
from bolna.constants import GPT5_MODEL_PREFIX, MODEL_REASONING_EFFORT_MAP

# Routing only sends reasoning_effort for the gpt-5 family. The map also carries the
# realtime speech-to-speech models, which accept an effort but are never a routing model.
GPT5_MODELS = sorted(m for m in MODEL_REASONING_EFFORT_MAP if m.startswith(GPT5_MODEL_PREFIX))
AZURE_DEPLOYMENT_FORMS = ["azure/gpt-5.4-mini", "ptu-gpt-5.4-mini"]


def _make_agent(**config_overrides):
    config = {
        "agent_information": "Test agent",
        "model": "gpt-4o-mini",
        "provider": "openai",
        "temperature": 0.7,
        "max_tokens": 150,
        "current_node_id": "start",
        "nodes": [{"id": "start", "prompt": "hi", "edges": []}],
    }
    config.update(config_overrides)

    with (
        patch("bolna.agent_types.graph_agent.OpenAI", return_value=MagicMock()),
        patch("bolna.agent_types.graph_agent.SUPPORTED_LLM_PROVIDERS", {"openai": MagicMock()}),
        patch("bolna.agent_types.graph_agent.OpenAiLLM", return_value=MagicMock()),
    ):
        return GraphAgent(config)


async def _captured_routing_kwargs(agent):
    """Run one routing decision and return the kwargs sent to the routing client."""
    captured = {}

    def _create(**kwargs):
        captured.update(kwargs)
        response = MagicMock()
        response.usage = None
        response.choices[0].message.tool_calls = None
        return response

    agent.routing_client = MagicMock()
    agent.routing_client.chat.completions.create = _create

    node = {"id": "start", "prompt": "hi"}
    edges = [{"to_node_id": "next", "condition_type": "intent", "intent": "wants next"}]
    await agent._decide_next_node_llm(node, edges, [{"role": "user", "content": "hello"}], 0.0)
    return captured


@pytest.mark.parametrize("model", GPT5_MODELS)
async def test_default_effort_is_accepted_by_the_model(model):
    agent = _make_agent(routing_model=model, routing_provider="openai")
    effort = (await _captured_routing_kwargs(agent))["reasoning_effort"]
    assert effort in [e.value for e in MODEL_REASONING_EFFORT_MAP[model]]


@pytest.mark.parametrize("deployment", AZURE_DEPLOYMENT_FORMS)
async def test_azure_deployment_names_take_the_gpt5_branch(deployment):
    agent = _make_agent(routing_model=deployment, routing_provider="openai")
    kwargs = await _captured_routing_kwargs(agent)

    assert kwargs["reasoning_effort"] in [e.value for e in MODEL_REASONING_EFFORT_MAP["gpt-5.4-mini"]]
    assert "max_completion_tokens" in kwargs
    assert "temperature" not in kwargs
    assert kwargs["model"] == deployment


async def test_explicit_effort_wins_over_the_default():
    agent = _make_agent(routing_model="gpt-5.4-mini", routing_provider="openai", routing_reasoning_effort="high")
    assert (await _captured_routing_kwargs(agent))["reasoning_effort"] == "high"


async def test_env_override_applies_when_unset():
    agent = _make_agent(routing_model="gpt-5.4-mini", routing_provider="openai")
    with patch.dict("os.environ", {"GPT5_ROUTING_REASONING_EFFORT": "medium"}):
        assert (await _captured_routing_kwargs(agent))["reasoning_effort"] == "medium"


async def test_non_gpt5_routing_is_unchanged():
    agent = _make_agent(routing_model="gpt-4.1-mini", routing_provider="openai")
    kwargs = await _captured_routing_kwargs(agent)

    assert "reasoning_effort" not in kwargs
    assert kwargs["max_tokens"] == 250
    assert kwargs["temperature"] == 0.0


async def test_routing_max_tokens_override_is_respected():
    agent = _make_agent(routing_model="gpt-5.4-mini", routing_provider="openai", routing_max_tokens=99)
    assert (await _captured_routing_kwargs(agent))["max_completion_tokens"] == 99
