"""Routing and hangup/voicemail hops must authenticate with the routing provider's own key.

The router (default gpt-4.1-mini) and the aux OpenAiLLM hops run on OpenAI/Azure regardless of
the conversation model. A conversation on a non-OpenAI provider carries a key that only its own
provider accepts, so these hops use the platform key instead of borrowing the conversation key.
"""

import os
from unittest.mock import MagicMock, patch

from bolna.agent_types.graph_agent import GraphAgent

PLATFORM_ENV = {
    "OPENAI_API_KEY": "platform-openai-key",
    "AZURE_OPENAI_API_KEY": "platform-azure-key",
    "AZURE_OPENAI_ENDPOINT": "https://platform.azure.example",
}


def _record(store):
    def make(**kwargs):
        store.append(kwargs)
        client = MagicMock()
        client.captured_kwargs = kwargs
        return client

    return make


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

    openai_calls, azure_calls, aux_calls = [], [], []
    with (
        patch.dict(os.environ, env if env is not None else PLATFORM_ENV, clear=True),
        patch("bolna.agent_types.graph_agent.OpenAI", side_effect=_record(openai_calls)),
        patch("bolna.agent_types.graph_agent.AzureOpenAI", side_effect=_record(azure_calls)),
        patch("bolna.agent_types.graph_agent.OpenAiLLM", side_effect=_record(aux_calls)),
        patch.object(GraphAgent, "_initialize_llm", return_value=MagicMock()),
    ):
        agent = GraphAgent(config)
    return agent, aux_calls


def test_openai_conversation_reuses_its_own_routing_key():
    agent, _ = _build(provider="openai", routing_provider="openai")
    assert agent.routing_client.captured_kwargs["api_key"] == "conv-key"


def test_gemini_conversation_routes_on_platform_openai_key():
    agent, _ = _build(provider="google", model="gemini-3.5-flash-lite", routing_provider="openai")
    assert agent.routing_client.captured_kwargs["api_key"] == "platform-openai-key"


def test_custom_conversation_reuses_its_own_routing_key():
    agent, _ = _build(provider="custom", base_url="https://proxy.example/v1", routing_provider="openai")
    assert agent.routing_client.captured_kwargs["api_key"] == "conv-key"


def test_azure_router_with_openai_conversation_uses_platform_azure_creds():
    agent, _ = _build(provider="openai", routing_provider="azure")
    kwargs = agent.routing_client.captured_kwargs
    assert kwargs["api_key"] == "platform-azure-key"
    assert kwargs["azure_endpoint"] == "https://platform.azure.example"


def test_azure_conversation_reuses_its_own_azure_creds():
    agent, _ = _build(
        provider="azure", llm_key="conv-key", base_url="https://conv.azure.example", routing_provider="azure"
    )
    kwargs = agent.routing_client.captured_kwargs
    assert kwargs["api_key"] == "conv-key"
    assert kwargs["azure_endpoint"] == "https://conv.azure.example"


def test_route_routing_to_conversation_reuses_conversation_client():
    agent, _ = _build(provider="google", model="gemini-3.5-flash-lite", route_routing_to_conversation=True)
    assert agent.routing_client.captured_kwargs["api_key"] == "conv-key"


def test_azure_routing_without_credentials_falls_back_to_openai():
    env = {"OPENAI_API_KEY": "platform-openai-key"}
    agent, _ = _build(env=env, provider="google", model="gemini-3.5-flash-lite", routing_provider="azure")
    assert agent.routing_provider == "openai"
    assert agent.routing_client.captured_kwargs["api_key"] == "platform-openai-key"
    assert "azure_endpoint" not in agent.routing_client.captured_kwargs


def test_azure_openai_conversation_routes_on_its_own_azure_creds():
    agent, _ = _build(
        provider="azure-openai",
        llm_key="conv-key",
        base_url="https://conv.azure.example",
        route_routing_to_conversation=True,
    )
    kwargs = agent.routing_client.captured_kwargs
    assert kwargs["api_key"] == "conv-key"
    assert kwargs["azure_endpoint"] == "https://conv.azure.example"


def test_gemini_conversation_aux_llm_uses_platform_openai_key():
    _, aux_calls = _build(provider="google", model="gemini-3.5-flash-lite")
    assert aux_calls, "aux OpenAiLLM was never constructed"
    assert all(call["llm_key"] == "platform-openai-key" for call in aux_calls)


def test_openai_conversation_aux_llm_reuses_its_own_key():
    _, aux_calls = _build(provider="openai")
    assert all(call["llm_key"] == "conv-key" for call in aux_calls)
