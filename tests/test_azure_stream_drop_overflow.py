"""A provisioned-pool stream that drops before any output is re-issued on the overflow target."""

import asyncio
import json

import httpx
import openai
import pytest
from openai import AsyncOpenAI

from bolna.llms import AzureLLM

DROP = "peer closed connection without sending complete message body (incomplete chunked read)"
MSGS = [{"role": "user", "content": "hi"}]
POOL_MODEL, OVERFLOW_MODEL = "gpt-4.1-mini", "gpt-4.1-mini-overflow"
DROP_AFTER_S = 0.2


def sse(delta=None, finish_reason=None, usage=None):
    chunk = {"id": "chatcmpl-x", "object": "chat.completion.chunk", "created": 0, "model": "m", "choices": []}
    if delta is not None or finish_reason is not None:
        chunk["choices"] = [{"index": 0, "delta": delta or {}, "finish_reason": finish_reason}]
    if usage is not None:
        chunk["usage"] = usage
    return f"data: {json.dumps(chunk)}\n\n".encode()


ROLE = sse({"role": "assistant", "content": ""})
REPLY = "Sure, I can help you book that appointment for tomorrow morning."
HEALTHY = [
    ROLE,
    sse({"content": REPLY}),
    sse(finish_reason="stop"),
    sse(usage={"prompt_tokens": 10, "completion_tokens": 12, "total_tokens": 22}),
    b"data: [DONE]\n\n",
]


class Body(httpx.AsyncByteStream):
    def __init__(self, events, then=None):
        self.events, self.then = events, then

    async def __aiter__(self):
        for event in self.events:
            yield event
        if self.then is not None:
            await self.then()


async def drop():
    await asyncio.sleep(DROP_AFTER_S)
    raise httpx.RemoteProtocolError(DROP)


class Backend:
    """A chat-completions endpoint that records each request and answers with the next scripted body."""

    def __init__(self, *bodies):
        self.bodies, self.requests = list(bodies), []

    def client(self):
        transport = httpx.MockTransport(self.handle)
        return AsyncOpenAI(
            api_key="k",
            base_url="https://backend.test/v1",
            max_retries=0,
            http_client=httpx.AsyncClient(transport=transport),
        )

    async def handle(self, request):
        self.requests.append(json.loads(request.content))
        body = self.bodies.pop(0)
        if isinstance(body, int):
            return httpx.Response(body, json={"error": {"message": "saturated"}})
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=body)


def azure_llm(pool, overflow=None, **kwargs):
    llm = AzureLLM(model=POOL_MODEL, llm_key="k", base_url="https://x.openai.azure.com", **kwargs)
    llm.async_client = pool.client()
    llm._overflow_client = overflow.client() if overflow else None
    llm._overflow_model, llm._overflow_service_tier = OVERFLOW_MODEL, "priority"
    return llm


async def run_turn(llm, meta_info=None):
    meta_info = {"sequence_id": 1} if meta_info is None else meta_info
    return [chunk async for chunk in llm.generate_stream(MSGS, meta_info=meta_info)]


def words(chunks):
    """The task manager joins streamed pieces with a space, so compare word by word."""
    return " ".join(c.data for c in chunks if c.data).split()


async def test_a_stream_that_drops_before_any_output_is_reissued_on_the_overflow():
    pool = Backend(Body([ROLE], then=drop))
    overflow = Backend(Body(HEALTHY))
    meta_info = {"sequence_id": 1}

    chunks = await run_turn(azure_llm(pool, overflow), meta_info)

    assert words(chunks) == REPLY.split()
    assert chunks[-1].overflowed is True
    assert chunks[-1].latency.first_token_latency_ms >= DROP_AFTER_S * 1000, "measured from the original request"
    assert [r["model"] for r in pool.requests] == [POOL_MODEL]
    assert [(r["model"], r["service_tier"]) for r in overflow.requests] == [(OVERFLOW_MODEL, "priority")]
    assert [e["error_type"] for e in meta_info["_non_fatal_errors"]] == ["stream_dropped_before_output"]


def stream_error(error_type):
    return f"data: {json.dumps({'error': {'message': 'The server had an error.', 'type': error_type}})}\n\n".encode()


async def test_a_server_error_inside_the_stream_before_any_output_is_reissued():
    pool = Backend(Body([ROLE, stream_error("server_error")]))
    overflow = Backend(Body(HEALTHY))

    chunks = await run_turn(azure_llm(pool, overflow))

    assert words(chunks) == REPLY.split()
    assert chunks[-1].overflowed is True


async def test_any_other_error_inside_the_stream_is_not_reissued():
    pool = Backend(Body([ROLE, stream_error("invalid_request_error")]))
    overflow = Backend(Body(HEALTHY))

    with pytest.raises(openai.APIError):
        await run_turn(azure_llm(pool, overflow))
    assert overflow.requests == []


async def test_a_stream_that_drops_after_output_is_not_reissued():
    """Part of the reply already went out to be spoken, so a second answer would repeat it."""
    pool = Backend(Body([ROLE, sse({"content": "Sure, I can help you book that appointment for"})], then=drop))
    overflow = Backend(Body(HEALTHY))

    with pytest.raises(httpx.RemoteProtocolError):
        await run_turn(azure_llm(pool, overflow))
    assert overflow.requests == []


async def test_a_stream_that_drops_with_its_text_still_buffered_is_reissued():
    """Text shorter than the buffer never left the LLM, so the caller heard nothing of it."""
    pool = Backend(Body([ROLE, sse({"content": "Sure, I can help"})], then=drop))
    overflow = Backend(Body(HEALTHY))
    meta_info = {"sequence_id": 1}

    chunks = await run_turn(azure_llm(pool, overflow), meta_info)

    assert words(chunks) == REPLY.split(), "the dropped fragment is discarded, not prepended"
    assert chunks[-1].overflowed is True
    assert [e["error_type"] for e in meta_info["_non_fatal_errors"]] == ["stream_dropped_before_output"]


END_CALL_TOOLS = {
    "tools": [{"type": "function", "function": {"name": "end_call", "parameters": {"type": "object"}}}],
    "tools_params": {},
}


async def test_a_stream_that_drops_mid_tool_call_is_reissued_without_the_half_call():
    """end_call speaks no filler, so a drop inside its arguments has sent nothing to the caller."""
    tool_call = {"index": 0, "id": "call_1", "type": "function", "function": {"name": "end_call", "arguments": '{"rea'}}
    pool = Backend(Body([ROLE, sse({"tool_calls": [tool_call]})], then=drop))
    overflow = Backend(Body(HEALTHY))

    chunks = await run_turn(azure_llm(pool, overflow, api_tools=END_CALL_TOOLS))

    assert words(chunks) == REPLY.split()
    assert not any(c.is_function_call for c in chunks)
    assert chunks[-1].overflowed is True


async def test_an_error_that_is_not_a_dropped_connection_is_not_reissued():
    async def boom():
        raise RuntimeError("boom")

    pool = Backend(Body([ROLE], then=boom))
    overflow = Backend(Body(HEALTHY))

    with pytest.raises(RuntimeError):
        await run_turn(azure_llm(pool, overflow))
    assert overflow.requests == []


async def test_a_turn_already_on_the_overflow_is_not_reissued_again():
    pool = Backend(429)
    overflow = Backend(Body([ROLE], then=drop), Body(HEALTHY))

    with pytest.raises(httpx.RemoteProtocolError):
        await run_turn(azure_llm(pool, overflow))
    assert len(overflow.requests) == 1


async def test_an_overflow_that_also_fails_ends_the_turn():
    pool = Backend(Body([ROLE], then=drop))
    overflow = Backend(429)

    with pytest.raises(openai.RateLimitError):
        await run_turn(azure_llm(pool, overflow))


async def test_without_an_overflow_target_a_dropped_stream_still_raises():
    pool = Backend(Body([ROLE], then=drop))

    with pytest.raises(httpx.RemoteProtocolError):
        await run_turn(azure_llm(pool))


async def test_a_healthy_stream_stays_on_the_pool():
    pool = Backend(Body(HEALTHY))
    overflow = Backend(Body(HEALTHY))
    meta_info = {"sequence_id": 1}

    chunks = await run_turn(azure_llm(pool, overflow), meta_info)

    assert words(chunks) == REPLY.split()
    assert chunks[-1].overflowed is False
    assert overflow.requests == []
    assert "_non_fatal_errors" not in meta_info


async def test_a_cancelled_turn_is_not_reissued():
    """Barge-in cancels the turn; that must never look like a dropped stream."""
    streaming = asyncio.Event()

    async def hold_open():
        streaming.set()
        await asyncio.Event().wait()

    pool = Backend(Body([ROLE], then=hold_open))
    overflow = Backend(Body(HEALTHY))

    turn = asyncio.create_task(run_turn(azure_llm(pool, overflow)))
    await asyncio.wait_for(streaming.wait(), timeout=5)
    turn.cancel()
    with pytest.raises(asyncio.CancelledError):
        await turn
    assert overflow.requests == []
