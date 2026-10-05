"""A Bolna-hosted model runs on OpenAiLLM against the endpoint and key the platform injects, and
only the conversation uses that endpoint: routing and the hangup/voicemail hops stay on OpenAI."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bolna.agent_types.graph_agent import GraphAgent
from bolna.llms.openai_llm import OpenAiLLM
from bolna.providers import SUPPORTED_LLM_PROVIDERS

MODEL = "google/gemma-4-26B-A4B-it"
BASE_URL = "https://gemma.inference.example.com/v1"
HOSTED_KEY = "hosted-key"
PLATFORM_KEY = "platform-openai-key"
MESSAGES = [{"role": "user", "content": "hi"}]


def _hosted_llm(**kwargs):
    return OpenAiLLM(model=MODEL, provider="bolna", base_url=BASE_URL, llm_key=HOSTED_KEY, **kwargs)


def test_provider_runs_on_openai_compatible_client():
    assert SUPPORTED_LLM_PROVIDERS["bolna"] is OpenAiLLM


class TestClient:
    def test_uses_the_injected_endpoint_and_key(self):
        llm = _hosted_llm()
        assert str(llm.async_client.base_url).rstrip("/") == BASE_URL
        assert llm.async_client.api_key == HOSTED_KEY

    def test_is_not_ssrf_guarded(self):
        assert _hosted_llm()._base_url is None

    def test_responses_api_stays_off(self):
        assert _hosted_llm(use_responses_api=True).use_responses_api is False

    async def test_request_carries_no_openai_only_fields(self):
        llm = _hosted_llm()
        llm.async_client = MagicMock()
        llm.async_client.chat.completions.create = AsyncMock(side_effect=RuntimeError("stop"))
        with pytest.raises(RuntimeError):
            async for _ in llm._generate_stream_chat(MESSAGES):
                pass
        sent = llm.async_client.chat.completions.create.call_args.kwargs
        assert sent["model"] == MODEL
        assert "service_tier" not in sent


def test_graph_agent_keeps_routing_and_hops_off_the_hosted_endpoint(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", PLATFORM_KEY)
    conversation = MagicMock()
    openai_llm = MagicMock()
    config = {
        "agent_information": "Test agent",
        "model": MODEL,
        "provider": "bolna",
        "base_url": BASE_URL,
        "llm_key": HOSTED_KEY,
        "current_node_id": "start",
        "nodes": [{"id": "start", "prompt": "hi", "edges": []}],
    }
    with (
        patch("bolna.agent_types.graph_agent.OpenAiLLM", openai_llm),
        patch("bolna.agent_types.graph_agent.SUPPORTED_LLM_PROVIDERS", {"bolna": conversation}),
    ):
        GraphAgent(config)

    assert conversation.call_args.kwargs["base_url"] == BASE_URL
    assert conversation.call_args.kwargs["llm_key"] == HOSTED_KEY
    # routing, completion check, voicemail
    others = [call.kwargs for call in openai_llm.call_args_list]
    assert len(others) == 3
    for kwargs in others:
        assert "base_url" not in kwargs
        assert kwargs.get("llm_key") != HOSTED_KEY
