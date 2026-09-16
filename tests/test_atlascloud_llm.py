from unittest.mock import patch

import pytest

from bolna.enums import LLMProvider
from bolna.llms.atlascloud_llm import AtlasCloudLLM
from bolna.providers import SUPPORTED_LLM_PROVIDERS


def test_atlascloud_provider_is_registered():
    assert LLMProvider.ATLASCLOUD.value == "atlascloud"
    assert SUPPORTED_LLM_PROVIDERS["atlascloud"] is AtlasCloudLLM


def test_atlascloud_requires_api_key(monkeypatch):
    monkeypatch.delenv("ATLASCLOUD_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ATLASCLOUD_API_KEY"):
        AtlasCloudLLM()


def test_atlascloud_applies_defaults_without_sdk_retries(monkeypatch):
    monkeypatch.setenv("ATLASCLOUD_API_KEY", "env-key")
    captured = {}

    def fake_openai_init(self, *args, **kwargs):
        captured.update(kwargs)
        self.model_args = {"service_tier": "default"}

    with patch("bolna.llms.atlascloud_llm.OpenAiLLM.__init__", fake_openai_init):
        llm = AtlasCloudLLM()

    assert captured["model"] == "qwen/qwen3.5-397b-a17b"
    assert captured["provider"] == "custom"
    assert captured["llm_key"] == "env-key"
    assert captured["base_url"] == "https://api.atlascloud.ai/v1"
    assert captured["max_retries"] == 0
    assert "service_tier" not in llm.model_args


def test_atlascloud_allows_explicit_credentials_and_endpoint(monkeypatch):
    monkeypatch.delenv("ATLASCLOUD_API_KEY", raising=False)
    captured = {}

    def fake_openai_init(self, *args, **kwargs):
        captured.update(kwargs)
        self.model_args = {}

    with patch("bolna.llms.atlascloud_llm.OpenAiLLM.__init__", fake_openai_init):
        AtlasCloudLLM(
            model="custom/model",
            llm_key="direct-key",
            base_url="https://example.com/v1",
        )

    assert captured["model"] == "custom/model"
    assert captured["llm_key"] == "direct-key"
    assert captured["base_url"] == "https://example.com/v1"


def test_atlascloud_configures_the_sdk_without_retries(monkeypatch):
    monkeypatch.setenv("ATLASCLOUD_API_KEY", "test-key")

    with (
        patch("bolna.llms.openai_llm.get_shared_http_client") as get_http_client,
        patch("bolna.llms.openai_llm.AsyncOpenAI") as async_openai,
    ):
        AtlasCloudLLM()

    async_openai.assert_called_once_with(
        base_url="https://api.atlascloud.ai/v1",
        api_key="test-key",
        http_client=get_http_client.return_value,
        max_retries=0,
    )
