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


def test_injected_routing_credentials_reach_the_routing_client():
    agent, _ = _build(
        provider="google",
        model="gemini-3.7-flash",
        routing_provider="azure",
        routing_model="azure/gpt-4.1-mini",
        routing_llm_key="routing-azure-key",
        routing_base_url="https://routing.azure.example",
        routing_api_version="2024-12-01-preview",
    )
    kw = _routing_kwargs(agent)
    assert kw["provider"] == "azure"
    assert kw["llm_key"] == "routing-azure-key"
    assert kw["base_url"] == "https://routing.azure.example"
    assert kw["api_version"] == "2024-12-01-preview"


def test_route_routing_to_conversation_overrides_injected_routing_credentials():
    # A PTU swap resolves routing creds for the original provider, then flips the conversation onto its
    # own deployment and sets route_routing_to_conversation. Routing must follow the conversation's
    # (PTU) credentials, not the standalone ones resolved before the swap.
    agent, _ = _build(
        provider="azure",
        llm_key="ptu-conv-key",
        base_url="https://ptu.azure.example",
        route_routing_to_conversation=True,
        routing_provider="azure",
        routing_llm_key="standalone-azure-key",
        routing_base_url="https://standalone.azure.example",
    )
    kw = _routing_kwargs(agent)
    assert kw["llm_key"] == "ptu-conv-key"
    assert kw["base_url"] == "https://ptu.azure.example"


def test_injected_routing_credentials_win_over_conversation_inheritance():
    # Same provider on both hops, but the caller supplied a distinct routing endpoint: use it, not the
    # conversation's, so a routing hop on a different deployment is not silently sent to the conv one.
    agent, _ = _build(
        provider="azure",
        llm_key="conv-key",
        base_url="https://conv.azure.example",
        routing_provider="azure",
        routing_model="azure/gpt-4.1-mini",
        routing_llm_key="routing-azure-key",
        routing_base_url="https://routing.azure.example",
    )
    kw = _routing_kwargs(agent)
    assert kw["llm_key"] == "routing-azure-key"
    assert kw["base_url"] == "https://routing.azure.example"


def test_routing_forwards_service_tier_and_reasoning_effort():
    agent, _ = _build(
        provider="openai", routing_model="gpt-5.4-mini", service_tier="priority", routing_reasoning_effort="low"
    )
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


# Real construction (no registry mock) for the paths whose behavior depends on the concrete LLM class.
def _real_agent(env=None, **overrides):
    config = {
        "agent_information": "Test agent",
        "model": "gpt-4o-mini",
        "provider": "openai",
        "llm_key": "conv-key",
        "current_node_id": "start",
        "nodes": [{"id": "start", "prompt": "hi", "edges": []}],
    }
    config.update(overrides)
    with patch.dict(
        os.environ, env if env is not None else {"OPENAI_API_KEY": "plat", "GOOGLE_API_KEY": "gg"}, clear=True
    ):
        return GraphAgent(config)


def test_custom_conversation_routes_on_its_own_endpoint():
    from bolna.llms import OpenAiLLM

    agent = _real_agent(provider="custom", base_url="https://box/v1", llm_key="ck", model="my-llama")
    assert agent.routing_provider == "custom"
    assert agent.routing_model == "my-llama"  # its endpoint has no gpt-4.1-mini
    assert isinstance(agent.routing_llm, OpenAiLLM)


def test_explicit_litellm_routing_prefixes_bare_model():
    from bolna.llms import LiteLLM

    agent = _real_agent(routing_provider="groq", routing_model="llama-3.3-70b-versatile")
    assert agent.routing_model == "groq/llama-3.3-70b-versatile"
    assert isinstance(agent.routing_llm, LiteLLM)


def test_already_prefixed_litellm_model_not_doubled():
    agent = _real_agent(routing_provider="anthropic", routing_model="anthropic/claude-3-5-sonnet")
    assert agent.routing_model == "anthropic/claude-3-5-sonnet"


# The runtime holds no Azure endpoint of its own, so an Azure routing hop under a non-Azure
# conversation model has no credentials to resolve and its client cannot be built.
AZURE_ROUTING = {"routing_provider": "azure", "routing_model": "azure/gpt-4.1-mini"}
AZURE_PLATFORM_ENV = {
    "OPENAI_API_KEY": "plat",
    "GOOGLE_API_KEY": "gg",
    "AZURE_OPENAI_ENDPOINT": "https://platform.azure.example",
    "AZURE_OPENAI_API_KEY": "platform-azure-key",
}
# What the runtime actually carries: an Azure key, and no endpoint to spend it against.
AZURE_KEY_ONLY_ENV = {k: v for k, v in AZURE_PLATFORM_ENV.items() if k != "AZURE_OPENAI_ENDPOINT"}


def test_azure_routing_falls_back_to_openai_without_a_platform_azure_endpoint():
    """A Gemini conversation routing on Azure: the agent is built, and the call is not lost."""
    from bolna.llms import OpenAiLLM

    agent = _real_agent(env=AZURE_KEY_ONLY_ENV, provider="google", model="gemini-3.7-flash", **AZURE_ROUTING)
    assert isinstance(agent.routing_llm, OpenAiLLM)
    assert agent.routing_provider == "openai"
    assert agent.routing_model == "gpt-4.1-mini"
    assert agent.routing_llm.max_tokens == 250


def test_azure_routing_fallback_honours_routing_max_tokens():
    agent = _real_agent(
        env=AZURE_KEY_ONLY_ENV, provider="google", model="gemini-3.7-flash", routing_max_tokens=64, **AZURE_ROUTING
    )
    assert agent.routing_llm.max_tokens == 64


def test_azure_routing_falls_back_with_no_azure_credentials_at_all():
    from bolna.llms import OpenAiLLM

    agent = _real_agent(provider="google", model="gemini-3.7-flash", **AZURE_ROUTING)
    assert isinstance(agent.routing_llm, OpenAiLLM)
    assert agent.routing_provider == "openai"


def test_azure_routing_stays_on_azure_when_the_platform_endpoint_is_set():
    from bolna.llms import AzureLLM

    agent = _real_agent(env=AZURE_PLATFORM_ENV, provider="google", model="gemini-3.7-flash", **AZURE_ROUTING)
    assert isinstance(agent.routing_llm, AzureLLM)
    assert agent.routing_provider == "azure"


def test_azure_routing_fallback_drops_the_unbuilt_models_reasoning_effort():
    gpt5_routing = {"routing_provider": "azure", "routing_model": "azure/gpt-5.4-mini"}

    on_azure = _real_agent(env=AZURE_PLATFORM_ENV, provider="google", model="gemini-3.7-flash", **gpt5_routing)
    assert on_azure._routing_reasoning_effort_used is not None

    fell_back = _real_agent(env=AZURE_KEY_ONLY_ENV, provider="google", model="gemini-3.7-flash", **gpt5_routing)
    assert fell_back.routing_provider == "openai"
    assert fell_back._routing_reasoning_effort_used is None


def test_injected_azure_routing_creds_build_the_real_client_without_a_platform_endpoint():
    """The proper fix: a Gemini conversation with caller-resolved Azure routing creds routes ON Azure,
    even though the runtime env carries no AZURE_OPENAI_ENDPOINT (the production shape)."""
    from bolna.llms import AzureLLM

    agent = _real_agent(
        env=AZURE_KEY_ONLY_ENV,
        provider="google",
        model="gemini-3.7-flash",
        routing_llm_key="routing-azure-key",
        routing_base_url="https://routing.azure.example",
        routing_api_version="2024-12-01-preview",
        **AZURE_ROUTING,
    )
    assert isinstance(agent.routing_llm, AzureLLM)
    assert agent.routing_provider == "azure"
    assert agent.routing_llm.llm_host == "routing.azure.example"
