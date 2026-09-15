"""Routing sends a reasoning_effort only for the gpt-5 family, and one the model accepts.

"minimal" is accepted only by gpt-5, gpt-5-mini and gpt-5-nano, so it cannot be the blanket
default. The family check reads the model a deployment resolves to, or Azure deployment names
such as "ptu-gpt-5.4-mini" miss the branch entirely. Non-gpt-5 routers never receive an effort,
even an explicit one, since chat completions reject it there.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from bolna.agent_types.graph_agent import GraphAgent
from bolna.constants import GPT5_MODEL_PREFIX, MODEL_REASONING_EFFORT_MAP

# The map also carries realtime speech-to-speech models, which are never a routing model.
GPT5_MODELS = sorted(m for m in MODEL_REASONING_EFFORT_MAP if m.startswith(GPT5_MODEL_PREFIX))
AZURE_DEPLOYMENT_FORMS = ["azure/gpt-5.4-mini", "ptu-gpt-5.4-mini"]
PLATFORM_ENV = {"OPENAI_API_KEY": "platform-openai-key", "AZURE_OPENAI_API_KEY": "platform-azure-key"}


def _registry():
    def make(**kwargs):
        client = MagicMock()
        client.captured_kwargs = kwargs
        return client

    return {p: make for p in ["openai", "azure", "custom"]}


def _build(env=None, **config_overrides):
    config = {
        "agent_information": "Test agent",
        "model": "gpt-4o-mini",
        "provider": "openai",
        "current_node_id": "start",
        "nodes": [{"id": "start", "prompt": "hi", "edges": []}],
    }
    config.update(config_overrides)
    with (
        patch.dict(os.environ, env if env is not None else PLATFORM_ENV, clear=True),
        patch("bolna.agent_types.graph_agent.SUPPORTED_LLM_PROVIDERS", _registry()),
        patch("bolna.agent_types.graph_agent.OpenAiLLM", return_value=MagicMock()),
    ):
        return GraphAgent(config)


def _kwargs(agent):
    return agent.routing_llm.captured_kwargs


def _accepted(model):
    return [e.value for e in MODEL_REASONING_EFFORT_MAP[model]]


@pytest.mark.parametrize("model", GPT5_MODELS)
def test_default_effort_is_accepted_by_the_model(model):
    agent = _build(routing_model=model, routing_provider="openai")
    assert _kwargs(agent)["reasoning_effort"] in _accepted(model)


@pytest.mark.parametrize("deployment", AZURE_DEPLOYMENT_FORMS)
def test_azure_deployment_names_take_the_gpt5_branch(deployment):
    agent = _build(routing_model=deployment, routing_provider="azure", provider="azure")
    kw = _kwargs(agent)
    assert kw["reasoning_effort"] in _accepted("gpt-5.4-mini")
    assert kw["model"] == deployment  # the deployment name itself is what Azure is asked for


def test_explicit_effort_wins_over_the_default():
    agent = _build(routing_model="gpt-5.4-mini", routing_provider="openai", routing_reasoning_effort="high")
    assert _kwargs(agent)["reasoning_effort"] == "high"


def test_env_override_applies_when_unset():
    env = {**PLATFORM_ENV, "GPT5_ROUTING_REASONING_EFFORT": "medium"}
    agent = _build(env=env, routing_model="gpt-5.4-mini", routing_provider="openai")
    assert _kwargs(agent)["reasoning_effort"] == "medium"


def test_non_gpt5_routing_gets_no_effort():
    agent = _build(routing_model="gpt-4.1-mini", routing_provider="openai")
    kw = _kwargs(agent)
    assert "reasoning_effort" not in kw
    assert kw["max_tokens"] == 250
    assert kw["temperature"] == 0


def test_explicit_effort_on_non_gpt5_is_dropped():
    agent = _build(routing_model="gpt-4.1-mini", routing_provider="openai", routing_reasoning_effort="minimal")
    assert "reasoning_effort" not in _kwargs(agent)


def test_recorded_effort_mirrors_what_was_sent():
    assert (
        _build(routing_model="gpt-5.4-mini", routing_provider="openai")._routing_reasoning_effort_used
        == (_kwargs(_build(routing_model="gpt-5.4-mini", routing_provider="openai"))["reasoning_effort"])
    )
    assert _build(routing_model="gpt-4.1-mini", routing_provider="openai")._routing_reasoning_effort_used is None


def test_routing_max_tokens_override_is_respected():
    agent = _build(routing_model="gpt-5.4-mini", routing_provider="openai", routing_max_tokens=99)
    assert _kwargs(agent)["max_tokens"] == 99
