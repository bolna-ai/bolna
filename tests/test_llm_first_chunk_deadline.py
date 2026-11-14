"""A conversation turn whose stream sends nothing within the first-chunk deadline is re-issued once."""

import asyncio
import json

import httpx
import pytest

from bolna.llms import AzureLLM, OpenAiLLM
from tests.test_azure_stream_drop_overflow import (
    HEALTHY,
    MSGS,
    OVERFLOW_MODEL,
    POOL_MODEL,
    REPLY,
    Backend,
    Body,
    azure_llm,
    run_turn,
    words,
)

DEADLINE_S = 0.2
NO_HEADERS = "no headers"


@pytest.fixture(autouse=True)
def short_deadline(monkeypatch):
    monkeypatch.setenv("LLM_FIRST_CHUNK_TIMEOUT_S", str(DEADLINE_S))


async def wait_forever():
    await asyncio.Event().wait()


def within_test_budget(turn):
    """Fails a turn that hangs instead of letting the test hang."""
    return asyncio.wait_for(turn, timeout=5)


class SilentBody(Body):
    """Response headers arrive, then nothing."""

    def __init__(self):
        super().__init__([], then=wait_forever)
        self.closed = False

    async def aclose(self):
        self.closed = True


class SlowBody(Body):
    """A healthy reply that starts later than the deadline."""

    async def __aiter__(self):
        await asyncio.sleep(DEADLINE_S * 2)
        async for event in super().__aiter__():
            yield event


class Endpoint(Backend):
    """A Backend that can also hold a request without ever sending response headers."""

    async def handle(self, request):
        if self.bodies[0] == NO_HEADERS:
            self.bodies.pop(0)
            self.requests.append(json.loads(request.content))
            await wait_forever()
        return await super().handle(request)


async def test_a_turn_that_gets_no_response_is_reissued_on_the_overflow():
    pool = Endpoint(NO_HEADERS)
    overflow = Endpoint(Body(HEALTHY))
    meta_info = {"sequence_id": 1}

    chunks = await within_test_budget(run_turn(azure_llm(pool, overflow), meta_info))

    assert words(chunks) == REPLY.split()
    assert chunks[-1].overflowed is True
    assert chunks[-1].latency.first_token_latency_ms >= DEADLINE_S * 1000, "measured from the original request"
    assert [r["model"] for r in overflow.requests] == [OVERFLOW_MODEL]
    assert [e["error_type"] for e in meta_info["_non_fatal_errors"]] == ["first_chunk_timeout"]


async def test_a_stream_silent_after_its_headers_is_closed_and_reissued_on_the_same_deployment():
    silent = SilentBody()
    pool = Endpoint(silent, Body(HEALTHY))

    chunks = await within_test_budget(run_turn(azure_llm(pool)))

    assert words(chunks) == REPLY.split()
    assert chunks[-1].overflowed is False
    assert [r["model"] for r in pool.requests] == [POOL_MODEL, POOL_MODEL]
    assert silent.closed


async def test_the_reissued_turn_is_not_cut_off_by_the_deadline():
    pool = Endpoint(NO_HEADERS, SlowBody(HEALTHY))

    chunks = await within_test_budget(run_turn(azure_llm(pool)))

    assert words(chunks) == REPLY.split()
    assert len(pool.requests) == 2


async def test_a_turn_cancelled_while_waiting_is_not_reissued():
    """Barge-in during the wait cancels the turn; the deadline must not fire a second request after it."""
    pool = Endpoint(NO_HEADERS, Body(HEALTHY))
    overflow = Endpoint(Body(HEALTHY))

    turn = asyncio.create_task(run_turn(azure_llm(pool, overflow)))
    await asyncio.sleep(DEADLINE_S / 2)
    turn.cancel()
    with pytest.raises(asyncio.CancelledError):
        await turn
    await asyncio.sleep(DEADLINE_S * 2)
    assert len(pool.requests) == 1
    assert overflow.requests == []


async def test_an_openai_turn_that_gets_no_response_is_reissued():
    backend = Endpoint(NO_HEADERS, Body(HEALTHY))
    llm = OpenAiLLM(model="gpt-4.1-mini", llm_key="k")
    llm.async_client = backend.client()

    async def turn():
        return [chunk async for chunk in llm.generate_stream(MSGS, meta_info={"sequence_id": 1})]

    chunks = await within_test_budget(turn())

    assert words(chunks) == REPLY.split()
    assert len(backend.requests) == 2


def test_reasoning_models_get_no_deadline():
    assert (
        AzureLLM(model="gpt-5.4-mini", llm_key="k", base_url="https://x.openai.azure.com")._first_chunk_timeout_s
        is None
    )
    assert OpenAiLLM(model="gpt-5.4-mini", llm_key="k")._first_chunk_timeout_s is None
    assert OpenAiLLM(model="gpt-4.1-mini", llm_key="k")._first_chunk_timeout_s == DEADLINE_S


def test_openai_compatible_endpoints_get_no_deadline():
    llm = OpenAiLLM(model="my-model", llm_key="k", base_url="https://llm.example.com/v1", provider="custom")
    assert llm._first_chunk_timeout_s is None


def test_zero_turns_the_deadline_off(monkeypatch):
    monkeypatch.setenv("LLM_FIRST_CHUNK_TIMEOUT_S", "0")
    assert AzureLLM(model=POOL_MODEL, llm_key="k", base_url="https://x.openai.azure.com")._first_chunk_timeout_s is None
