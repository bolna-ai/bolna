"""Every provider's route() issues one forced tool-call and returns a normalized decision.

Graph routing is provider-agnostic: it calls llm.route(messages, tools) and reads back
{function_name, arguments, usage, service_tier, overflowed} regardless of the backend.
The OpenAI-shaped providers stream it so the decision lands before the rationale.
"""

import json
from types import SimpleNamespace

import httpx
from openai import APIStatusError
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


def _chunk(name=None, arguments=None, usage=None, service_tier=None):
    tool_calls = None
    if name is not None or arguments is not None:
        tool_calls = [SimpleNamespace(index=0, id="c1", function=SimpleNamespace(name=name, arguments=arguments))]
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(tool_calls=tool_calls))],
        usage=usage,
        service_tier=service_tier,
    )


async def _aiter(chunks):
    for chunk in chunks:
        yield chunk


def _openai_stream(name="go_to_x", args=None, service_tier="default"):
    """A forced tool call streamed the way the providers emit it: name first, then arguments."""
    args = json.dumps(args if args is not None else {"reasoning": "clear", "confidence": 0.9})
    usage = SimpleNamespace(
        prompt_tokens=100, completion_tokens=10, completion_tokens_details=None, prompt_tokens_details=None
    )
    return _aiter(
        [
            _chunk(name=name, arguments="", service_tier=service_tier),
            _chunk(arguments=args),
            SimpleNamespace(choices=[], usage=usage, service_tier=service_tier),
        ]
    )


async def test_openai_route_normalizes_and_strips_internal_keys():
    llm = OpenAiLLM(model="gpt-4.1-mini", llm_key="k")
    llm.async_client = MagicMock()
    llm.async_client.chat.completions.create = AsyncMock(return_value=_openai_stream())
    res = await llm.route(MSGS, TOOLS)

    assert res["function_name"] == "go_to_x"
    assert res["arguments"] == {}  # decided on the function name, before reasoning/confidence
    tail = await res["routing_tail"]
    assert tail["reasoning"] == "clear"
    assert tail["confidence"] == 0.9
    assert res["overflowed"] is False

    sent = llm.async_client.chat.completions.create.call_args.kwargs
    assert "turn_id" not in sent["messages"][0]
    assert sent["tool_choice"] == "required"
    assert sent["parallel_tool_calls"] is False
    assert sent["stream"] is True
    assert sent["stream_options"] == {"include_usage": True}
    assert sent["temperature"] == 0.0


async def test_openai_route_gpt5_omits_temperature_keeps_reasoning_effort():
    llm = OpenAiLLM(model="gpt-5.4-mini", llm_key="k")
    llm.async_client = MagicMock()
    llm.async_client.chat.completions.create = AsyncMock(return_value=_openai_stream())
    await llm.route(MSGS, TOOLS)

    sent = llm.async_client.chat.completions.create.call_args.kwargs
    assert "temperature" not in sent
    assert "reasoning_effort" in sent
    assert "max_completion_tokens" in sent


async def test_azure_route_reports_overflow():
    llm = AzureLLM(model="gpt-4.1-mini", llm_key="k", base_url="https://x.openai.azure.com")
    llm._create_completion = AsyncMock(return_value=(_openai_stream(), True))
    res = await llm.route(MSGS, TOOLS)
    assert res["overflowed"] is True
    assert res["function_name"] == "go_to_x"


async def test_azure_route_overflows_a_saturated_pool_while_streaming():
    # The SDK raises on the response status before it hands back a stream, so a saturated
    # pool still falls through to the overflow backend on the routing call.
    llm = AzureLLM(model="gpt-4.1-mini", llm_key="k", base_url="https://x.openai.azure.com")
    saturated = APIStatusError(
        "saturated", response=httpx.Response(429, request=httpx.Request("POST", "https://x")), body=None
    )
    llm.async_client = MagicMock()
    llm.async_client.chat.completions.create = AsyncMock(side_effect=saturated)
    llm._overflow_client = MagicMock()
    llm._overflow_client.chat.completions.create = AsyncMock(return_value=_openai_stream())
    llm._overflow_model, llm._overflow_service_tier = "gpt-4.1-mini-payg", "priority"

    res = await llm.route(MSGS, TOOLS)

    assert res["overflowed"] is True
    assert res["function_name"] == "go_to_x"
    assert llm._overflow_client.chat.completions.create.call_args.kwargs["stream"] is True


async def test_litellm_route_normalizes():
    llm = LiteLLM(model="groq/llama-3.3-70b-versatile", llm_key="k")
    with patch("bolna.llms.litellm.acompletion", AsyncMock(return_value=_openai_stream())):
        res = await llm.route(MSGS, TOOLS)
    assert res["function_name"] == "go_to_x"
    assert (await res["routing_tail"])["confidence"] == 0.9


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


async def test_openai_route_runs_ssrf_guard():
    llm = OpenAiLLM(model="gpt-4.1-mini", llm_key="k")
    llm._ensure_base_url_allowed = AsyncMock()
    llm.async_client = MagicMock()
    llm.async_client.chat.completions.create = AsyncMock(return_value=_openai_stream())
    await llm.route(MSGS, TOOLS)
    llm._ensure_base_url_allowed.assert_awaited_once()


def test_gemini_flattens_tool_history_for_routing():
    msgs = [
        {
            "role": "assistant",
            "content": "ok",
            "tool_calls": [{"function": {"name": "verify", "arguments": '{"id":1}'}}],
        },
        {"role": "tool", "tool_call_id": "x", "content": "verified"},
        {"role": "user", "content": "yes"},
    ]
    flat = GeminiLLM._flatten_tool_history_for_routing(msgs)
    assert all("tool_calls" not in m for m in flat)  # no function-call Part -> no thought_signature needed
    assert any("called verify" in m.get("content", "") for m in flat)
    assert any("tool result: verified" in m.get("content", "") for m in flat)


async def test_route_returns_none_without_tool_call():
    llm = OpenAiLLM(model="gpt-4.1-mini", llm_key="k")
    llm.async_client = MagicMock()
    llm.async_client.chat.completions.create = AsyncMock(return_value=_aiter([_chunk()]))
    assert await llm.route(MSGS, TOOLS) is None
