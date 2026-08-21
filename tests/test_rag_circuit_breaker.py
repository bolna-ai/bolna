"""A failing knowledge-base service must not hold up the conversation.

Every turn of a KB agent queries RAG before the LLM runs, so a dead service would add its
timeout to each turn. After a run of failures the client stops calling and returns an empty
context, so the agent answers without the knowledge base rather than leaving dead air. Recovery is
probed by a single request, not by every concurrent turn at once.
"""

import asyncio
import time

from bolna.helpers.rag_service_client import RAGServiceClient

THRESHOLD = 3
COOLDOWN = 30.0


class _FakeResponse:
    def __init__(self, status=200, payload=None):
        self.status = status
        self._payload = payload or {}

    async def json(self):
        return self._payload

    async def text(self):
        return "error body"

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeSession:
    """Stands in for aiohttp.ClientSession: post() is used as an async context manager."""

    def __init__(self, response=None, raises=None):
        self._response = response
        self._raises = raises
        self.calls = 0

    def post(self, *args, **kwargs):
        self.calls += 1
        if self._raises:
            raise self._raises
        return self._response

    async def close(self):
        pass


def _client(session=None, failures=0, last_failure_at=None):
    client = RAGServiceClient("http://rag.invalid")
    client.session = session or _FakeSession(_FakeResponse())
    client._consecutive_failures = failures
    client._last_failure_time = time.time() if last_failure_at is None else last_failure_at
    return client


def test_a_fresh_client_is_available():
    assert _client(failures=0)._is_available() is True


def test_failures_below_the_threshold_do_not_open_the_breaker():
    assert _client(failures=THRESHOLD - 1)._is_available() is True


def test_the_breaker_opens_at_the_threshold():
    assert _client(failures=THRESHOLD)._is_available() is False


def test_the_breaker_admits_a_probe_after_the_cooldown():
    client = _client(failures=THRESHOLD, last_failure_at=time.time() - COOLDOWN - 1)
    assert client._is_available() is True


def test_only_one_probe_gets_through():
    """Otherwise every concurrent turn retries the dead service at once."""
    client = _client(failures=THRESHOLD, last_failure_at=time.time() - COOLDOWN - 1)
    assert client._is_available() is True
    assert client._is_available() is False


async def test_an_open_breaker_returns_empty_context_without_calling_out():
    session = _FakeSession(_FakeResponse())
    client = _client(session=session, failures=THRESHOLD)

    response = await client.query_for_conversation("what are your hours", ["col-1"])

    assert response.contexts == []
    assert response.total_results == 0
    assert session.calls == 0


async def test_a_successful_query_returns_contexts_and_resets_the_breaker():
    payload = {
        "documents": [{"text": "We open at nine.", "score": 0.91, "metadata": {"source": "faq"}}],
        "total_retrieved": 1,
        "query_time_ms": 120.0,
    }
    client = _client(session=_FakeSession(_FakeResponse(200, payload)), failures=THRESHOLD - 1)

    response = await client.query_for_conversation("hours", ["col-1"])

    assert [c.text for c in response.contexts] == ["We open at nine."]
    assert response.contexts[0].score == 0.91
    assert response.total_results == 1
    assert client._consecutive_failures == 0


async def test_a_non_200_counts_as_a_failure_and_degrades_quietly():
    client = _client(session=_FakeSession(_FakeResponse(503)), failures=0)

    response = await client.query_for_conversation("hours", ["col-1"])

    assert response.contexts == []
    assert client._consecutive_failures == 1


async def test_a_timeout_counts_as_a_failure_and_degrades_quietly():
    client = _client(session=_FakeSession(raises=asyncio.TimeoutError()), failures=0)

    response = await client.query_for_conversation("hours", ["col-1"])

    assert response.contexts == []
    assert client._consecutive_failures == 1


async def test_repeated_failures_eventually_open_the_breaker():
    client = _client(session=_FakeSession(_FakeResponse(500)), failures=0)

    for _ in range(THRESHOLD):
        await client.query_for_conversation("hours", ["col-1"])

    assert client._is_available() is False
