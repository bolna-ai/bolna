"""Every provider's route() issues one forced tool-call and returns a normalized decision.

Graph routing is provider-agnostic: it calls llm.route(messages, tools) and reads back
{function_name, arguments, usage, service_tier, overflowed} regardless of the backend.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from bolna.llms import OpenAiLLM, AzureLLM, LiteLLM, GeminiLLM

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "go_to_x",
            "description": "move to node x",
            "parameters": {
                "type": "object",
                "properties": {"reasoning": {"type": "string"}, "confidence": {"type": "number"}},
            },
        },
    }
]
MSGS = [{"role": "user", "content": "yes", "turn_id": 7}]  # turn_id is bolna-internal, must be stripped


def _openai_completion(name="go_to_x", args=None, service_tier="default"):
    args = json.dumps(args if args is not None else {"reasoning": "clear", "confidence": 0.9})
    func = SimpleNamespace(name=name, arguments=args)
    msg = SimpleNamespace(tool_calls=[SimpleNamespace(function=func)])
    usage = SimpleNamespace(
        prompt_tokens=100, completion_tokens=10, completion_tokens_details=None, prompt_tokens_details=None
    )
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=usage, service_tier=service_tier)


async def test_openai_route_normalizes_and_strips_internal_keys():
    llm = OpenAiLLM(model="gpt-4.1-mini", llm_key="k")
    llm.async_client = MagicMock()
    llm.async_client.chat.completions.create = AsyncMock(return_value=_openai_completion())
    res = await llm.route(MSGS, TOOLS)

    assert res["function_name"] == "go_to_x"
    assert res["arguments"] == {"reasoning": "clear", "confidence": 0.9}
    assert res["usage"]["input_tokens"] == 100
    assert res["overflowed"] is False

    sent = llm.async_client.chat.completions.create.call_args.kwargs
    assert "turn_id" not in sent["messages"][0]
    assert sent["tool_choice"] == "required"
    assert sent["parallel_tool_calls"] is False
    assert sent["stream"] is False
    assert sent["temperature"] == 0.0


async def test_openai_route_gpt5_omits_temperature_keeps_reasoning_effort():
    llm = OpenAiLLM(model="gpt-5.4-mini", llm_key="k")
    llm.async_client = MagicMock()
    llm.async_client.chat.completions.create = AsyncMock(return_value=_openai_completion())
    await llm.route(MSGS, TOOLS)

    sent = llm.async_client.chat.completions.create.call_args.kwargs
    assert "temperature" not in sent
    assert "reasoning_effort" in sent
    assert "max_completion_tokens" in sent


async def test_azure_route_reports_overflow():
    llm = AzureLLM(model="gpt-4.1-mini", llm_key="k", base_url="https://x.openai.azure.com")
    llm._create_completion = AsyncMock(return_value=(_openai_completion(), True))
    res = await llm.route(MSGS, TOOLS)
    assert res["overflowed"] is True
    assert res["function_name"] == "go_to_x"


async def test_litellm_route_normalizes():
    llm = LiteLLM(model="groq/llama-3.3-70b-versatile", llm_key="k")
    with patch("bolna.llms.litellm.acompletion", AsyncMock(return_value=_openai_completion())):
        res = await llm.route(MSGS, TOOLS)
    assert res["function_name"] == "go_to_x"
    assert res["arguments"]["confidence"] == 0.9


async def test_gemini_route_normalizes():
    llm = GeminiLLM(model="gemini-2.0-flash", llm_key="k")
    fc = SimpleNamespace(name="go_to_x", args={"reasoning": "clear", "confidence": 1.0})
    part = SimpleNamespace(function_call=fc)
    candidate = SimpleNamespace(content=SimpleNamespace(parts=[part]))
    usage = SimpleNamespace(
        prompt_token_count=50, candidates_token_count=5, thoughts_token_count=0, cached_content_token_count=0
    )
    resp = SimpleNamespace(candidates=[candidate], usage_metadata=usage)
    llm.client = MagicMock()
    llm.client.aio.models.generate_content = AsyncMock(return_value=resp)

    res = await llm.route([{"role": "user", "content": "yes"}], TOOLS)
    assert res["function_name"] == "go_to_x"
    assert res["arguments"] == {"reasoning": "clear", "confidence": 1.0}
    assert res["usage"]["input_tokens"] == 50


async def test_route_returns_none_without_tool_call():
    llm = OpenAiLLM(model="gpt-4.1-mini", llm_key="k")
    empty = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=None))], usage=None, service_tier=None
    )
    llm.async_client = MagicMock()
    llm.async_client.chat.completions.create = AsyncMock(return_value=empty)
    assert await llm.route(MSGS, TOOLS) is None
