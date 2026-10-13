"""A provisioned-pool stream that drops before any output is re-issued on the overflow target.

Driven through the real OpenAI SDK over a mock transport, so the exception is the one httpx raises
when the server closes a chunked response early.
"""

import asyncio
import json

import httpx
import pytest
from openai import AsyncOpenAI

from bolna.llms import AzureLLM

DROP = "peer closed connection without sending complete message body (incomplete chunked read)"
MSGS = [{"role": "user", "content": "hi"}]


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


def azure_llm(pool, overflow=None):
    llm = AzureLLM(model="gpt-4.1-mini", llm_key="k", base_url="https://x.openai.azure.com")
    llm.async_client = pool.client()
    llm._overflow_client = overflow.client() if overflow else None
    llm._overflow_model, llm._overflow_service_tier = "gpt-4.1-mini", "priority"
    return llm


async def run_turn(llm):
    return [chunk async for chunk in llm.generate_stream(MSGS, meta_info={"sequence_id": 1})]


def words(chunks):
    """The task manager joins streamed pieces with a space, so compare word by word."""
    return " ".join(c.data for c in chunks if c.data).split()


async def test_a_stream_that_drops_before_any_output_is_reissued_on_the_overflow():
    pool = Backend(Body([ROLE], then=drop))
    overflow = Backend(Body(HEALTHY))

    chunks = await run_turn(azure_llm(pool, overflow))

    assert words(chunks) == REPLY.split()
    assert chunks[-1].overflowed is True
    assert chunks[-1].latency.first_token_latency_ms is not None
    assert len(pool.requests) == 1
    assert overflow.requests[0]["service_tier"] == "priority"


async def test_a_stream_that_drops_after_output_is_not_reissued():
    """Part of the reply may already be spoken, so a second answer would repeat it."""
    pool = Backend(Body([ROLE, sse({"content": "Sure, I can help"})], then=drop))
    overflow = Backend(Body(HEALTHY))

    with pytest.raises(httpx.RemoteProtocolError):
        await run_turn(azure_llm(pool, overflow))
    assert overflow.requests == []


async def test_a_turn_already_on_the_overflow_is_not_reissued_again():
    pool = Backend(429)
    overflow = Backend(Body([ROLE], then=drop), Body(HEALTHY))

    with pytest.raises(httpx.RemoteProtocolError):
        await run_turn(azure_llm(pool, overflow))
    assert len(overflow.requests) == 1


async def test_without_an_overflow_target_a_dropped_stream_still_raises():
    pool = Backend(Body([ROLE], then=drop))

    with pytest.raises(httpx.RemoteProtocolError):
        await run_turn(azure_llm(pool))


async def test_a_healthy_stream_stays_on_the_pool():
    pool = Backend(Body(HEALTHY))
    overflow = Backend(Body(HEALTHY))

    chunks = await run_turn(azure_llm(pool, overflow))

    assert words(chunks) == REPLY.split()
    assert chunks[-1].overflowed is False
    assert overflow.requests == []


async def test_a_cancelled_turn_is_not_reissued():
    """Barge-in cancels the turn; that must never look like a dropped stream."""
    pool = Backend(Body([ROLE], then=asyncio.Event().wait))
    overflow = Backend(Body(HEALTHY))

    turn = asyncio.create_task(run_turn(azure_llm(pool, overflow)))
    await asyncio.sleep(0.05)
    turn.cancel()
    with pytest.raises(asyncio.CancelledError):
        await turn
    assert overflow.requests == []
