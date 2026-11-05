"""bedrock/<model> runs on Bedrock's OpenAI-compatible API under its inference profile id."""

from unittest.mock import AsyncMock, patch

import pytest

from bolna.agent_types.graph_agent import OPENAI_KEYED_PROVIDERS
from bolna.constants import BEDROCK_DEFAULT_BASE_URL
from bolna.llms import BedrockLLM
from bolna.providers import SUPPORTED_LLM_PROVIDERS

INDIA_URL = "https://bedrock-runtime.ap-south-1.amazonaws.com/openai/v1"


def _client(llm):
    return str(llm.async_client.base_url).rstrip("/"), llm.async_client.api_key


def test_bedrock_provider_is_registered():
    assert SUPPORTED_LLM_PROVIDERS["bedrock"] is BedrockLLM


def test_wire_id_is_the_india_profile_and_the_family_stays():
    llm = BedrockLLM(model="bedrock/gpt-5.6-luna", provider="bedrock", llm_key="bedrock-key")
    assert llm.model == llm.model_args["model"] == "in.openai.gpt-5.6-luna"
    assert llm.model_family == "gpt-5.6-luna"
    assert llm.request_log_model == "bedrock/gpt-5.6-luna"
    assert "max_completion_tokens" in llm.model_args and "reasoning_effort" in llm.model_args


def test_platform_key_and_india_endpoint_by_default():
    with patch.dict("os.environ", {"BEDROCK_API_KEY": "platform-bedrock-key"}, clear=False):
        llm = BedrockLLM(model="bedrock/gpt-5.6-luna", provider="bedrock")
    assert _client(llm) == (BEDROCK_DEFAULT_BASE_URL, "platform-bedrock-key")
    assert BEDROCK_DEFAULT_BASE_URL == INDIA_URL


def test_explicit_key_and_base_url_win():
    llm = BedrockLLM(
        model="bedrock/gpt-5.6-luna", provider="bedrock", llm_key="k", base_url="https://bedrock.example/openai/v1"
    )
    assert _client(llm) == ("https://bedrock.example/openai/v1", "k")


def test_responses_api_streams_over_http():
    llm = BedrockLLM(model="bedrock/gpt-5.6-luna", provider="bedrock", llm_key="k", use_responses_api=True)
    assert llm.use_responses_api and llm._ws_transport is None


def test_side_hops_do_not_reuse_the_bedrock_key():
    assert "bedrock" not in OPENAI_KEYED_PROVIDERS


@pytest.mark.asyncio
async def test_streaming_chat_sends_the_profile_id_without_stop():
    llm = BedrockLLM(model="bedrock/gpt-5.6-luna", provider="bedrock", llm_key="k")
    llm.async_client.chat.completions.create = AsyncMock(side_effect=RuntimeError("stop here"))
    with pytest.raises(RuntimeError):
        async for _ in llm.generate_stream([{"role": "user", "content": "hi"}], meta_info={}):
            pass
    sent = llm.async_client.chat.completions.create.call_args.kwargs
    assert sent["model"] == "in.openai.gpt-5.6-luna"
    assert "stop" not in sent


def test_missing_bedrock_key_raises_instead_of_sending_the_openai_key(monkeypatch):
    monkeypatch.delenv("BEDROCK_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "platform-openai-key")
    with pytest.raises(ValueError, match="BEDROCK_API_KEY"):
        BedrockLLM(model="bedrock/gpt-5.6-luna", provider="bedrock")
