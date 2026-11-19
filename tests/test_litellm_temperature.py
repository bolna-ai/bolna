"""Claude models after 4.6 400 on any temperature, so LiteLLM must leave it out of every request for them."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from bolna.llms import LiteLLM

REJECTING = [
    "anthropic/claude-opus-4-8",
    "anthropic/claude-opus-4-7",
    "claude-opus-5-5",
    "anthropic/claude-sonnet-5-5",
    "anthropic/claude-sonnet-5",
    "claude-haiku-5-5",
    "anthropic/claude-fable-5-2",
    "bedrock/global.anthropic.claude-opus-4-8-v1:0",
]
ACCEPTING = [
    "anthropic/claude-sonnet-4-6",
    "claude-haiku-4-5-20251001",
    "bedrock/in.anthropic.claude-haiku-4-5-20251001-v1:0",
    "groq/llama-3.3-70b-versatile",
]

TOOLS = [{"type": "function", "function": {"name": "go_to_x", "parameters": {"type": "object", "properties": {}}}}]


async def _empty_stream():
    for chunk in [SimpleNamespace(choices=[], usage=None, service_tier=None)]:
        yield chunk


@pytest.mark.parametrize("model", REJECTING)
def test_temperature_is_omitted_for_models_that_reject_it(model):
    assert "temperature" not in LiteLLM(model=model, temperature=0.2, llm_key="k").model_args


@pytest.mark.parametrize("model", ACCEPTING)
def test_temperature_is_kept_for_models_that_accept_it(model):
    assert LiteLLM(model=model, temperature=0.2, llm_key="k").model_args["temperature"] == 0.2


@pytest.mark.parametrize("model", REJECTING)
async def test_route_sends_no_temperature_for_models_that_reject_it(model):
    llm = LiteLLM(model=model, llm_key="k")
    mock = AsyncMock(return_value=_empty_stream())
    with patch("bolna.llms.litellm.acompletion", mock), patch("bolna.llms.litellm.read_routing_stream", AsyncMock()):
        await llm.route([{"role": "user", "content": "hi"}], TOOLS)
    assert "temperature" not in mock.call_args.kwargs


async def test_route_keeps_zero_temperature_for_models_that_accept_it():
    llm = LiteLLM(model="anthropic/claude-sonnet-4-6", llm_key="k")
    mock = AsyncMock(return_value=_empty_stream())
    with patch("bolna.llms.litellm.acompletion", mock), patch("bolna.llms.litellm.read_routing_stream", AsyncMock()):
        await llm.route([{"role": "user", "content": "hi"}], TOOLS)
    assert mock.call_args.kwargs["temperature"] == 0.0
