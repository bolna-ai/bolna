"""gpt-6 takes the same request shape as gpt-5, so the family check has to see it.

Every reasoning branch used to test `model.startswith("gpt-5")`. A gpt-6 model fell through to
the legacy path and OpenAI rejected the request four separate ways: `max_tokens` instead of
`max_completion_tokens`, a `stop` list, a temperature other than 1, and function tools on chat
completions. The last one has no escape on gpt-6-astra, which unlike gpt-5.4 and up has no
"none" effort, so tool-using agents must reach the Responses API.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bolna.constants import (
    MODEL_REASONING_EFFORT_MAP,
    RESPONSES_API_MODEL_PREFIXES,
    canonical_model,
    default_reasoning_effort,
    is_reasoning_model,
)
from bolna.llms.azure_llm import AzureLLM
from bolna.llms.openai_llm import OpenAiLLM
from bolna.models import validate_reasoning_effort_for_model

ASTRA = "gpt-6-astra"


def _openai_llm(model=ASTRA, **kwargs):
    return OpenAiLLM(model=model, max_tokens=150, temperature=0.2, llm_key="test-key", **kwargs)


class TestFamilyDetection:
    @pytest.mark.parametrize("model", [ASTRA, "gpt-6", "gpt-5", "gpt-5.4-mini", "gpt-5.6-luna"])
    def test_reasoning_families_are_recognised(self, model):
        assert is_reasoning_model(model)

    @pytest.mark.parametrize("model", ["gpt-4.1-mini", "gpt-4o", "claude-sonnet-5", "", None])
    def test_everything_else_is_not(self, model):
        assert not is_reasoning_model(model)

    def test_an_azure_deployment_name_resolves_to_the_family(self):
        """Deployment names are chosen freely, so the raw name cannot be read directly."""
        assert canonical_model("azure/ptu-gpt-6-astra") == ASTRA
        assert is_reasoning_model(canonical_model("azure/ptu-gpt-6-astra"))


class TestReasoningEffort:
    def test_the_floor_is_low(self):
        """gpt-6 dropped the "none" and "minimal" floors, so the lowest-latency effort is "low"."""
        assert default_reasoning_effort(ASTRA) == "low"

    @pytest.mark.parametrize("effort", ["none", "minimal"])
    def test_efforts_gpt6_dropped_are_rejected(self, effort):
        with pytest.raises(ValueError, match=effort):
            validate_reasoning_effort_for_model(ASTRA, effort)

    def test_max_is_accepted(self):
        """ "max" is new in gpt-6 and had no ReasoningEffort member."""
        validate_reasoning_effort_for_model(ASTRA, "max")

    def test_the_catalog_advertises_what_the_api_accepts(self):
        assert [e.value for e in MODEL_REASONING_EFFORT_MAP[ASTRA]] == ["low", "medium", "high", "xhigh", "max"]


class TestChatCompletionsRequestShape:
    def test_the_output_cap_uses_the_reasoning_key(self):
        llm = _openai_llm()
        assert llm.model_args["max_completion_tokens"] == 150
        assert "max_tokens" not in llm.model_args

    def test_an_effort_is_always_sent(self):
        assert _openai_llm().model_args["reasoning_effort"] == "low"

    def test_a_configured_effort_wins_over_the_default(self):
        assert _openai_llm(reasoning_effort="high").model_args["reasoning_effort"] == "high"

    async def test_stop_is_omitted(self):
        """gpt-6 rejects `stop` outright, and the sequence only ever guarded gpt-4-era rambling."""
        llm = _openai_llm()
        llm.async_client = MagicMock()
        llm.async_client.chat.completions.create = AsyncMock(side_effect=RuntimeError("stop stream"))

        with pytest.raises(RuntimeError):
            async for _ in llm._generate_stream_chat([{"role": "user", "content": "hi"}]):
                pass

        assert "stop" not in llm.async_client.chat.completions.create.call_args.kwargs

    async def test_a_non_reasoning_model_still_gets_stop(self):
        llm = _openai_llm(model="gpt-4.1-mini")
        llm.async_client = MagicMock()
        llm.async_client.chat.completions.create = AsyncMock(side_effect=RuntimeError("stop stream"))

        with pytest.raises(RuntimeError):
            async for _ in llm._generate_stream_chat([{"role": "user", "content": "hi"}]):
                pass

        assert llm.async_client.chat.completions.create.call_args.kwargs["stop"] == ["User:"]


class TestResponsesRequestShape:
    def test_tool_using_agents_are_routed_to_the_responses_api(self):
        """Function tools on chat completions are a 400 for gpt-6-astra at every effort."""
        assert any(prefix in ASTRA for prefix in RESPONSES_API_MODEL_PREFIXES)

    def test_temperature_is_pinned_to_one(self):
        """The model accepts only the default; the configured 0.2 would be a 400."""
        llm = _openai_llm()
        create_kwargs, _ = llm._build_responses_create_kwargs(
            [{"role": "user", "content": "hi"}], None, False, None, stream=True
        )
        assert create_kwargs["temperature"] == 1

    def test_the_effort_reaches_the_payload(self):
        llm = _openai_llm(reasoning_effort="xhigh")
        create_kwargs, _ = llm._build_responses_create_kwargs(
            [{"role": "user", "content": "hi"}], None, False, None, stream=True
        )
        assert create_kwargs["reasoning"]["effort"] == "xhigh"


class TestAzureDeployment:
    def test_a_deployment_serving_astra_gets_the_reasoning_shape(self):
        with patch("bolna.llms.azure_llm.AsyncAzureOpenAI"):
            llm = AzureLLM(
                model="azure/ptu-gpt-6-astra",
                max_tokens=150,
                temperature=0.2,
                llm_key="test-key",
                api_version="2026-01-01",
                base_url="https://example.openai.azure.com",
            )
        assert llm.model_family == ASTRA
        assert llm.model_args["max_completion_tokens"] == 150
        assert llm.model_args["reasoning_effort"] == "low"
