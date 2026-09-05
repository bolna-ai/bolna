"""Reasoning models take a different request shape, and the family check decides who gets it.

`max_tokens` becomes `max_completion_tokens`, `stop` is dropped, temperature is pinned to 1 and
an effort is always sent. A model that misses the check falls through to the legacy shape and
the provider rejects every one of those four, so the agent answers nothing for the whole call.

Each case runs across two generations. The check used to be `startswith("gpt-5")`, so gpt-6
fell through; parameterising is what stops the next generation repeating it.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bolna.constants import canonical_model, default_reasoning_effort, is_reasoning_model
from bolna.llms.azure_llm import AzureLLM
from bolna.llms.openai_llm import OpenAiLLM

# (model, the lowest-latency effort it supports)
REASONING_MODELS = [("gpt-5.4-mini", "none"), ("gpt-6-astra", "low")]
REASONING_MODEL_IDS = [model for model, _ in REASONING_MODELS]


def _openai_llm(model, **kwargs):
    return OpenAiLLM(model=model, max_tokens=150, temperature=0.2, llm_key="test-key", **kwargs)


def _capture_chat_kwargs(llm):
    """Drive the chat-completions stream far enough to read the kwargs it would send."""
    llm.async_client = MagicMock()
    llm.async_client.chat.completions.create = AsyncMock(side_effect=RuntimeError("stop stream"))
    return llm.async_client.chat.completions.create


def _responses_kwargs(llm):
    kwargs, _ = llm._build_responses_create_kwargs([{"role": "user", "content": "hi"}], None, False, None, stream=True)
    return kwargs


class TestFamilyDetection:
    @pytest.mark.parametrize("model", REASONING_MODEL_IDS + ["gpt-5", "gpt-5.6-luna", "gpt-6"])
    def test_reasoning_families_are_recognised(self, model):
        assert is_reasoning_model(model)

    @pytest.mark.parametrize("model", ["gpt-4.1-mini", "gpt-4o", "claude-sonnet-5", "", None])
    def test_everything_else_is_not(self, model):
        assert not is_reasoning_model(model)

    @pytest.mark.parametrize("model", REASONING_MODEL_IDS)
    def test_an_azure_deployment_name_resolves_to_the_family(self, model):
        """Deployment names are chosen freely, so the raw name cannot be read directly."""
        assert canonical_model(f"azure/ptu-{model}") == model
        assert is_reasoning_model(canonical_model(f"azure/ptu-{model}"))


class TestChatCompletions:
    @pytest.mark.parametrize("model", REASONING_MODEL_IDS)
    def test_the_output_cap_uses_the_reasoning_key(self, model):
        llm = _openai_llm(model)
        assert llm.model_args["max_completion_tokens"] == 150
        assert "max_tokens" not in llm.model_args

    @pytest.mark.parametrize("model, expected", REASONING_MODELS)
    def test_the_lowest_latency_effort_is_the_default(self, model, expected):
        assert _openai_llm(model).model_args["reasoning_effort"] == expected
        assert default_reasoning_effort(model) == expected

    @pytest.mark.parametrize("model", REASONING_MODEL_IDS)
    def test_a_configured_effort_wins_over_the_default(self, model):
        assert _openai_llm(model, reasoning_effort="high").model_args["reasoning_effort"] == "high"

    @pytest.mark.parametrize("model", REASONING_MODEL_IDS)
    async def test_stop_is_omitted(self, model):
        """The sequence only ever guarded gpt-4-era rambling, and reasoning models reject it."""
        llm = _openai_llm(model)
        create = _capture_chat_kwargs(llm)

        with pytest.raises(RuntimeError):
            async for _ in llm._generate_stream_chat([{"role": "user", "content": "hi"}]):
                pass

        assert "stop" not in create.call_args.kwargs

    async def test_a_non_reasoning_model_still_gets_stop(self):
        llm = _openai_llm("gpt-4.1-mini")
        create = _capture_chat_kwargs(llm)

        with pytest.raises(RuntimeError):
            async for _ in llm._generate_stream_chat([{"role": "user", "content": "hi"}]):
                pass

        assert create.call_args.kwargs["stop"] == ["User:"]

    def test_a_non_reasoning_model_keeps_the_legacy_cap_and_no_effort(self):
        llm = _openai_llm("gpt-4.1-mini")
        assert llm.model_args["max_tokens"] == 150
        assert "reasoning_effort" not in llm.model_args


class TestResponsesApi:
    @pytest.mark.parametrize("model", REASONING_MODEL_IDS)
    def test_temperature_is_pinned_to_one(self, model):
        """These models accept only the default; the configured 0.2 would be a 400."""
        assert _responses_kwargs(_openai_llm(model))["temperature"] == 1

    @pytest.mark.parametrize("model", REASONING_MODEL_IDS)
    def test_the_effort_reaches_the_payload(self, model):
        kwargs = _responses_kwargs(_openai_llm(model, reasoning_effort="high"))
        assert kwargs["reasoning"]["effort"] == "high"


class TestAzureDeployments:
    @pytest.mark.parametrize("model, expected", REASONING_MODELS)
    def test_a_deployment_gets_the_shape_of_the_model_it_serves(self, model, expected):
        with patch("bolna.llms.azure_llm.AsyncAzureOpenAI"):
            llm = AzureLLM(
                model=f"azure/ptu-{model}",
                max_tokens=150,
                temperature=0.2,
                llm_key="test-key",
                api_version="2026-01-01",
                base_url="https://example.openai.azure.com",
            )
        assert llm.model_family == model
        assert llm.model_args["max_completion_tokens"] == 150
        assert llm.model_args["reasoning_effort"] == expected
