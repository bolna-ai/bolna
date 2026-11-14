"""LiteLLM conversation streams never record token usage.

Groq, Anthropic, OpenRouter and Deepseek all go through LiteLLM. OpenAI and Azure
ask the provider for a trailing usage record and copy it onto the chunk TaskManager
reads; generate_stream does neither. A usage-only chunk (choices=[]) also indexes
choices[0] and aborts the turn before any text already yielded can be closed out.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from bolna.llms.litellm import LiteLLM

MESSAGES = [{"role": "user", "content": "hi", "turn_id": 3}]
LOOKUP = {
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Look up an account",
                "parameters": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]},
            },
        }
    ],
    "tools_params": {"lookup": {"url": "https://tool.test/lookup", "method": "POST"}},
}


def _usage(prompt=120, completion=18, reasoning=3, cached=40):
    return SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=reasoning),
        prompt_tokens_details=SimpleNamespace(cached_tokens=cached),
    )


def _text(text, finish_reason=None, usage=None):
    return {
        "id": "c1",
        "choices": [{"delta": SimpleNamespace(content=text, tool_calls=None), "finish_reason": finish_reason}],
        "usage": usage,
    }


def _usage_only(usage):
    # OpenAI-compatible providers (Groq, OpenRouter, Deepseek) send the tally on its own chunk.
    return {"id": "c1", "choices": [], "usage": usage}


def _tool(name=None, arguments=None, finish_reason=None):
    call = SimpleNamespace(index=0, id="call-1", function=SimpleNamespace(name=name, arguments=arguments))
    return {
        "id": "c1",
        "choices": [{"delta": SimpleNamespace(content=None, tool_calls=[call]), "finish_reason": finish_reason}],
        "usage": None,
    }


async def _aiter(chunks):
    for chunk in chunks:
        yield chunk


def _llm(**kwargs):
    return LiteLLM(model="groq/llama-3.3-70b-versatile", llm_key="k", **kwargs)


async def _chunks(llm, chunks, **kwargs):
    with patch("bolna.llms.litellm.acompletion", AsyncMock(return_value=_aiter(chunks))) as mocked:
        out = [c async for c in llm.generate_stream(MESSAGES, meta_info={"sequence_id": 4}, **kwargs)]
    return out, mocked.await_args.kwargs


async def test_spoken_turn_requests_usage_and_copies_the_trailing_record():
    usage = _usage()
    chunks, sent = await _chunks(
        _llm(),
        [_text("Hello there", finish_reason="stop"), _usage_only(usage)],
    )

    assert sent["stream_options"] == {"include_usage": True}
    assert "turn_id" not in sent["messages"][0]
    final = chunks[-1]
    assert final.end_of_stream is True
    assert final.data == "Hello there"
    assert final.input_tokens == 120
    assert final.output_tokens == 18
    assert final.reasoning_tokens == 3
    assert final.cached_tokens == 40


async def test_flushed_reply_still_reports_usage_on_a_closing_chunk():
    # A reply with no trailing word is yielded in the loop, leaving buffer empty.
    llm = _llm()
    llm.buffer_size = 4
    chunks, _sent = await _chunks(llm, [_text("Hello", finish_reason="stop"), _usage_only(_usage())])

    assert chunks[0].data == "Hello"
    assert chunks[0].end_of_stream is False
    assert chunks[0].input_tokens is None
    closing = chunks[-1]
    assert closing.end_of_stream is True
    assert closing.data == ""
    assert closing.input_tokens == 120
    assert closing.output_tokens == 18


async def test_usage_on_the_last_content_chunk_is_kept():
    # Anthropic's translated stream puts the tally on the chunk that also carries finish_reason.
    chunks, _sent = await _chunks(
        _llm(), [_text("Hello", usage=_usage(prompt=11, completion=2, reasoning=0, cached=0))]
    )

    final = chunks[-1]
    assert final.input_tokens == 11
    assert final.output_tokens == 2
    assert final.reasoning_tokens == 0
    assert final.cached_tokens == 0


async def test_tool_call_turn_puts_usage_on_the_function_call_chunk():
    llm = _llm(api_tools=LOOKUP)
    chunks, sent = await _chunks(
        llm,
        [
            _tool(name="lookup", arguments=""),
            _tool(arguments='{"id": "u1"}', finish_reason="tool_calls"),
            _usage_only(_usage(prompt=80, completion=9, reasoning=None, cached=None)),
        ],
    )

    assert sent["stream_options"] == {"include_usage": True}
    call = next(c for c in chunks if c.is_function_call)
    assert call.input_tokens == 80
    assert call.output_tokens == 9
    assert call.data.called_fun == "lookup"
