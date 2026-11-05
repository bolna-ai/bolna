"""Sarvam LLM: Sarvam's endpoint only, its own key, and graph hops kept off it."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bolna.agent_types.graph_agent import GraphAgent
from bolna.llms.azure_llm import AzureLLM
from bolna.llms.openai_llm import OpenAiLLM
from bolna.providers import SUPPORTED_LLM_PROVIDERS

MODEL = "sarvam-105b-conversations"
SARVAM_URL = "https://api.sarvam.ai/v1"
PLATFORM_SARVAM_KEY = "platform-sarvam-key"
CUSTOMER_SARVAM_KEY = "customer-sarvam-key"
PLATFORM_OPENAI_KEY = "platform-openai-key"
MESSAGES = [{"role": "user", "content": "hi"}]


@pytest.fixture(autouse=True)
def platform_keys(monkeypatch):
    monkeypatch.setenv("SARVAM_API_KEY", PLATFORM_SARVAM_KEY)
    monkeypatch.setenv("OPENAI_API_KEY", PLATFORM_OPENAI_KEY)


def _sarvam_llm(**kwargs):
    return OpenAiLLM(model=MODEL, provider="sarvam", **kwargs)


async def _sent_request(llm):
    llm.async_client = MagicMock()
    llm.async_client.chat.completions.create = AsyncMock(side_effect=RuntimeError("stop"))
    with pytest.raises(RuntimeError):
        async for _ in llm._generate_stream_chat(MESSAGES):
            pass
    return llm.async_client.chat.completions.create.call_args.kwargs


def test_provider_runs_on_openai_compatible_client():
    assert SUPPORTED_LLM_PROVIDERS["sarvam"] is OpenAiLLM


class TestClient:
    def test_defaults_to_sarvam_endpoint_and_platform_key(self):
        llm = _sarvam_llm()
        assert str(llm.async_client.base_url).rstrip("/") == SARVAM_URL
        assert llm.async_client.api_key == PLATFORM_SARVAM_KEY
        assert llm.llm_host == "api.sarvam.ai"

    def test_customer_key_wins_over_platform_key(self):
        assert _sarvam_llm(llm_key=CUSTOMER_SARVAM_KEY).async_client.api_key == CUSTOMER_SARVAM_KEY

    def test_agent_base_url_never_receives_the_key(self):
        llm = _sarvam_llm(base_url="https://elsewhere.example.com/v1")
        assert str(llm.async_client.base_url).rstrip("/") == SARVAM_URL
        assert llm.async_client.api_key == PLATFORM_SARVAM_KEY

    def test_no_key_fails_instead_of_sending_the_openai_key(self, monkeypatch):
        monkeypatch.delenv("SARVAM_API_KEY")
        with pytest.raises(ValueError, match="SARVAM_API_KEY"):
            _sarvam_llm()

    def test_responses_api_stays_off(self):
        assert _sarvam_llm(use_responses_api=True).use_responses_api is False

    async def test_request_carries_no_openai_only_fields(self):
        sent = await _sent_request(_sarvam_llm(service_tier="priority", reasoning_effort="low"))
        assert sent["model"] == MODEL
        assert "service_tier" not in sent
        assert "reasoning_effort" not in sent
        assert sent["max_tokens"]


def test_graph_agent_keeps_hops_off_sarvam(monkeypatch):
    conversation = MagicMock()
    openai_llm = MagicMock()
    config = {
        "agent_information": "Test agent",
        "model": MODEL,
        "provider": "sarvam",
        "llm_key": CUSTOMER_SARVAM_KEY,
        "current_node_id": "start",
        "nodes": [{"id": "start", "prompt": "hi", "edges": []}],
    }
    with (
        patch("bolna.agent_types.graph_agent.OpenAiLLM", openai_llm),
        patch("bolna.agent_types.graph_agent.SUPPORTED_LLM_PROVIDERS", {"sarvam": conversation}),
    ):
        agent = GraphAgent(config)

    assert conversation.call_args.kwargs["provider"] == "sarvam"
    assert conversation.call_args.kwargs["llm_key"] == CUSTOMER_SARVAM_KEY
    assert isinstance(agent.voicemail_llm, AzureLLM)
    hops = [call.kwargs for call in openai_llm.call_args_list]
    assert len(hops) == 2
    for kwargs in hops:
        assert "base_url" not in kwargs
        assert kwargs.get("llm_key") != CUSTOMER_SARVAM_KEY
        assert kwargs.get("provider") != "sarvam"


def _graph_with_sarvam_node():
    node = {"id": "pitch", "prompt": "hi", "edges": [], "llm_config": {"provider": "sarvam", "model": MODEL}}
    config = {
        "agent_information": "Test agent",
        "model": "gpt-4.1-mini",
        "provider": "openai",
        "current_node_id": "pitch",
        "nodes": [node],
        "node_llm_credentials": {"pitch": {}},
    }
    return GraphAgent(config), node


def test_node_switched_to_sarvam_runs_on_sarvam():
    agent, node = _graph_with_sarvam_node()
    assert agent._conversation_llm_for(node).llm_host == "api.sarvam.ai"


def test_node_switched_to_sarvam_without_a_key_stays_on_the_agent_llm(monkeypatch):
    monkeypatch.delenv("SARVAM_API_KEY")
    agent, node = _graph_with_sarvam_node()
    assert agent._conversation_llm_for(node) is agent.llm
