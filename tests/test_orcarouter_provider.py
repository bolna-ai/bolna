from unittest.mock import patch

from bolna.enums import LLMProvider
from bolna.providers import OrcaRouterLLM, SUPPORTED_LLM_PROVIDERS


def test_orcarouter_is_registered_as_an_openai_compatible_provider():
    assert SUPPORTED_LLM_PROVIDERS[LLMProvider.ORCAROUTER.value] is OrcaRouterLLM


def test_orcarouter_defaults_to_its_endpoint_and_api_key(monkeypatch):
    monkeypatch.setenv("ORCAROUTER_API_KEY", "orca-key")

    with patch("bolna.providers.OpenAiLLM.__init__", return_value=None) as init:
        OrcaRouterLLM(model="orcarouter/auto")

    init.assert_called_once_with(
        model="orcarouter/auto",
        provider="custom",
        base_url="https://api.orcarouter.ai/v1",
        llm_key="orca-key",
    )