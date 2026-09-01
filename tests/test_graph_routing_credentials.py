"""Routing runs on an LLM built from the same registry as the conversation model.

Routing reuses the conversation credentials only when it shares the conversation's provider;
otherwise it names a different provider and each LLM class resolves its own platform key. This
is what stops a Gemini conversation from lending its key to an OpenAI router (BOLNA-2536).
"""

import os
from unittest.mock import MagicMock, patch

from bolna.agent_types.graph_agent import GraphAgent

PLATFORM_ENV = {
    "OPENAI_API_KEY": "platform-openai-key",
    "AZURE_OPENAI_API_KEY": "platform-azure-key",
    "AZURE_OPENAI_ENDPOINT": "https://platform.azure.example",
    "GOOGLE_API_KEY": "platform-google-key",
}

REGISTRY_PROVIDERS = ["openai", "azure", "azure-openai", "google", "custom", "ola", "groq", "cohere", "anthropic"]


def _registry(store):
    def for_provider(name):
        def make(**kwargs):
            client = MagicMock()
            client.captured_kwargs = kwargs
            store.append((name, kwargs))
            return client

        return make

    return {p: for_provider(p) for p in REGISTRY_PROVIDERS}


def _build(env=None, **config_overrides):
    config = {
        "agent_information": "Test agent",
        "model": "gpt-4o-mini",
        "provider": "openai",
        "llm_key": "conv-key",
        "current_node_id": "start",
        "nodes": [{"id": "start", "prompt": "hi", "edges": []}],
    }
    config.update(config_overrides)

    aux_calls = []

    def aux_factory(**kwargs):
        aux_calls.append(kwargs)
        return MagicMock()

    with (
        patch.dict(os.environ, env if env is not None else PLATFORM_ENV, clear=True),
        patch("bolna.agent_types.graph_agent.SUPPORTED_LLM_PROVIDERS", _registry([])),
        patch("bolna.agent_types.graph_agent.OpenAiLLM", side_effect=aux_factory),
    ):
        agent = GraphAgent(config)
    return agent, aux_calls


def _routing_kwargs(agent):
    return agent.routing_llm.captured_kwargs


def test_openai_conversation_reuses_its_own_routing_key():
    agent, _ = _build(provider="openai")
    kw = _routing_kwargs(agent)
    assert kw["provider"] == "openai"
    assert kw["llm_key"] == "conv-key"
    assert kw["temperature"] == 0
    assert kw["max_tokens"] == 250


def test_gemini_conversation_routes_on_platform_openai_key():
    agent, _ = _build(provider="google", model="gemini-3.5-flash-lite")
    kw = _routing_kwargs(agent)
    assert kw["provider"] == "openai"
    assert "llm_key" not in kw  # the OpenAI router resolves its own OPENAI_API_KEY, not the Gemini key


def test_route_routing_to_conversation_runs_routing_on_gemini():
    agent, _ = _build(provider="google", model="gemini-3.5-flash-lite", route_routing_to_conversation=True)
    kw = _routing_kwargs(agent)
    assert kw["provider"] == "google"
    assert kw["model"] == "gemini-3.5-flash-lite"
    assert kw["llm_key"] == "conv-key"


def test_route_routing_to_conversation_reuses_azure_creds():
    agent, _ = _build(
        provider="azure",
        llm_key="conv-key",
        base_url="https://conv.azure.example",
        api_version="2024-12-01-preview",
        route_routing_to_conversation=True,
    )
    kw = _routing_kwargs(agent)
    assert kw["provider"] == "azure"
    assert kw["llm_key"] == "conv-key"
    assert kw["base_url"] == "https://conv.azure.example"


def test_explicit_azure_routing_with_openai_conversation_uses_platform_creds():
    agent, _ = _build(provider="openai", routing_provider="azure")
    kw = _routing_kwargs(agent)
    assert kw["provider"] == "azure"
    assert "llm_key" not in kw
    assert "base_url" not in kw


def test_routing_forwards_service_tier_and_reasoning_effort():
    agent, _ = _build(provider="openai", service_tier="priority", routing_reasoning_effort="low")
    kw = _routing_kwargs(agent)
    assert kw["service_tier"] == "priority"
    assert kw["reasoning_effort"] == "low"


def test_gemini_conversation_aux_llm_uses_platform_openai_key():
    _, aux_calls = _build(provider="google", model="gemini-3.5-flash-lite")
    assert aux_calls, "aux OpenAiLLM was never constructed"
    assert all(call["llm_key"] == "platform-openai-key" for call in aux_calls)


def test_openai_conversation_aux_llm_reuses_its_own_key():
    _, aux_calls = _build(provider="openai")
    assert all(call["llm_key"] == "conv-key" for call in aux_calls)
