from unittest.mock import patch

import pytest

from bolna.constants import LLM_VENDOR_ENDPOINTS
from bolna.enums import LLMProvider
from bolna.llms.atlascloud_llm import AtlasCloudLLM
from bolna.llms.openai_llm import OpenAiLLM
from bolna.providers import SUPPORTED_LLM_PROVIDERS


def test_atlascloud_provider_is_registered():
    assert LLMProvider.ATLASCLOUD.value == "atlascloud"
    assert SUPPORTED_LLM_PROVIDERS["atlascloud"] is AtlasCloudLLM
    assert LLM_VENDOR_ENDPOINTS["atlascloud"] == ("https://api.atlascloud.ai/v1", "ATLASCLOUD_API_KEY")


def test_atlascloud_requires_api_key(monkeypatch):
    monkeypatch.delenv("ATLASCLOUD_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ATLASCLOUD_API_KEY"):
        AtlasCloudLLM()


def test_atlascloud_uses_vendor_endpoint_and_env_key(monkeypatch):
    monkeypatch.setenv("ATLASCLOUD_API_KEY", "test-key")

    with (
        patch("bolna.llms.openai_llm.get_shared_http_client") as get_http_client,
        patch("bolna.llms.openai_llm.AsyncOpenAI") as async_openai,
    ):
        llm = AtlasCloudLLM()

    assert llm.model == "qwen/qwen3.5-397b-a17b"
    async_openai.assert_called_once()
    kwargs = async_openai.call_args.kwargs
    assert kwargs["base_url"] == "https://api.atlascloud.ai/v1"
    assert kwargs["api_key"] == "test-key"
    # A platform vendor is not a customer endpoint, so the SSRF-guarded client is not used.
    assert get_http_client.call_args.kwargs["guarded"] is False
    # Atlas Cloud's chat-completions endpoint does not use OpenAI service tiers.
    assert "service_tier" not in llm.model_args


def test_atlascloud_allows_explicit_model_and_key(monkeypatch):
    monkeypatch.delenv("ATLASCLOUD_API_KEY", raising=False)

    with (
        patch("bolna.llms.openai_llm.get_shared_http_client"),
        patch("bolna.llms.openai_llm.AsyncOpenAI") as async_openai,
    ):
        llm = AtlasCloudLLM(model="deepseek-ai/deepseek-v4-flash", llm_key="direct-key")

    assert llm.model == "deepseek-ai/deepseek-v4-flash"
    assert async_openai.call_args.kwargs["api_key"] == "direct-key"


def test_atlascloud_does_not_send_stop_sequences():
    assert AtlasCloudLLM.STOP_SEQUENCES is None
    assert OpenAiLLM.STOP_SEQUENCES == ("User:",)
