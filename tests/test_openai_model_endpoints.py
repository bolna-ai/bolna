"""A model listed in openai_model_endpoints runs on its own OpenAI endpoint; every other model keeps the caller's."""

import os
from unittest.mock import MagicMock, patch

import pytest

from bolna.agent_types.graph_agent import GraphAgent
from bolna.llms.openai_llm import OpenAiLLM, responses_ws_url

EU = {"llm_key": "eu-key", "base_url": "https://eu.api.openai.com/v1"}
INDIA = {"llm_key": "india-key", "base_url": "https://in.api.openai.com/v1"}
ENDPOINTS = {"gpt-6-luna": INDIA}
PLATFORM_ENV = {
    "OPENAI_API_KEY": "platform-openai-key",
    "AZURE_OPENAI_API_KEY": "platform-azure-key",
    "AZURE_OPENAI_ENDPOINT": "https://platform.azure.example",
}


@pytest.mark.parametrize(
    "base_url, expected",
    [
        (None, "wss://api.openai.com/v1/responses"),
        ("https://in.api.openai.com/v1", "wss://in.api.openai.com/v1/responses"),
        ("https://eu.api.openai.com/v1", "wss://eu.api.openai.com/v1/responses"),
        ("https://IN.api.openai.com:443/v1", "wss://in.api.openai.com/v1/responses"),
        ("https://gemma.inference.example.com/v1", None),
    ],
)
def test_responses_ws_url_follows_verified_hosts(base_url, expected):
    assert responses_ws_url(base_url) == expected


def _client(llm):
    return str(llm.async_client.base_url).rstrip("/"), llm.async_client.api_key


def test_listed_model_runs_on_its_endpoint():
    llm = OpenAiLLM(model="gpt-6-luna", provider="openai", openai_model_endpoints=ENDPOINTS, **EU)
    assert _client(llm) == (INDIA["base_url"], INDIA["llm_key"])
    assert llm.llm_host == "in.api.openai.com"


def test_unlisted_model_keeps_the_callers_endpoint():
    llm = OpenAiLLM(model="gpt-4.1-mini", provider="openai", openai_model_endpoints=ENDPOINTS, **EU)
    assert _client(llm) == (EU["base_url"], EU["llm_key"])


def test_non_openai_provider_ignores_the_map():
    llm = OpenAiLLM(
        model="gpt-6-luna",
        provider="custom",
        base_url="https://gemma.inference.example.com/v1",
        llm_key="customer-key",
        openai_model_endpoints=ENDPOINTS,
    )
    assert _client(llm) == ("https://gemma.inference.example.com/v1", "customer-key")


@pytest.mark.parametrize(
    "model, ws_url",
    [("gpt-6-luna", "wss://in.api.openai.com/v1/responses"), ("gpt-4.1-mini", "wss://eu.api.openai.com/v1/responses")],
)
def test_responses_websocket_follows_the_resolved_endpoint(model, ws_url):
    with patch("bolna.llms.openai_llm.OpenAIWSConnection") as transport:
        OpenAiLLM(model=model, provider="openai", use_responses_api=True, openai_model_endpoints=ENDPOINTS, **EU)
    key = INDIA["llm_key"] if model in ENDPOINTS else EU["llm_key"]
    transport.assert_called_once_with(api_key=key, url=ws_url)


def test_graph_agent_hops_riding_the_conversation_carry_the_map():
    conversation_kwargs, aux_kwargs = [], []

    def conversation_factory(**kwargs):
        conversation_kwargs.append(kwargs)
        return MagicMock()

    def aux_factory(**kwargs):
        aux_kwargs.append(kwargs)
        return MagicMock()

    config = {
        "agent_information": "Test agent",
        "model": "gpt-6-luna",
        "provider": "openai",
        "openai_model_endpoints": ENDPOINTS,
        "current_node_id": "start",
        "nodes": [{"id": "start", "prompt": "hi", "edges": []}],
        **EU,
    }
    with (
        patch.dict(os.environ, PLATFORM_ENV, clear=True),
        patch("bolna.agent_types.graph_agent.SUPPORTED_LLM_PROVIDERS", {"openai": conversation_factory}),
        patch("bolna.agent_types.graph_agent.OpenAiLLM", side_effect=aux_factory),
    ):
        GraphAgent(config)

    routing, conversation = conversation_kwargs
    hangup = aux_kwargs[0]
    for hop in (routing, conversation, hangup):
        assert hop["openai_model_endpoints"] == ENDPOINTS
        assert hop["base_url"] == EU["base_url"]
