"""End-to-end tests for RAG service client circuit breaker.

Verifies that:
1. After 3 consecutive failures, queries are skipped instantly (no network call)
2. After cooldown, a retry is allowed
3. A successful response resets the failure counter
4. Concurrent calls from multiple turns all benefit from the skip
"""
import asyncio
import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from bolna.helpers.rag_service_client import RAGServiceClient, RAGResponse


@pytest.fixture
def client():
    c = RAGServiceClient("http://fake-rag:8000", timeout=5)
    # Pre-create a mock session so _ensure_session doesn't open a real connection
    c.session = MagicMock()
    return c


def _make_timeout_post(session_mock):
    """Make session.post(...) raise asyncio.TimeoutError."""
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(side_effect=asyncio.TimeoutError)
    ctx.__aexit__ = AsyncMock(return_value=False)
    session_mock.post.return_value = ctx


def _make_success_post(session_mock, documents=None):
    """Make session.post(...) return a 200 with documents."""
    if documents is None:
        documents = [{"text": "some context", "score": 0.9, "metadata": {"collection_id": "col1"}}]

    resp_mock = AsyncMock()
    resp_mock.status = 200
    resp_mock.json = AsyncMock(return_value={
        "documents": documents,
        "total_retrieved": len(documents),
        "query_time_ms": 10.0,
    })

    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=resp_mock)
    ctx.__aexit__ = AsyncMock(return_value=False)
    session_mock.post.return_value = ctx


def _make_500_post(session_mock):
    """Make session.post(...) return HTTP 500."""
    resp_mock = AsyncMock()
    resp_mock.status = 500
    resp_mock.text = AsyncMock(return_value="Internal Server Error")

    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=resp_mock)
    ctx.__aexit__ = AsyncMock(return_value=False)
    session_mock.post.return_value = ctx


QUERY_ARGS = dict(query="hello", collections=["col1"])


@pytest.mark.asyncio
async def test_skips_after_3_timeouts(client):
    """After 3 timeout failures, the 4th call should skip without hitting the network."""
    _make_timeout_post(client.session)

    # First 3 calls hit the network and timeout
    for i in range(3):
        resp = await client.query_for_conversation(**QUERY_ARGS)
        assert resp.contexts == []
        assert client._consecutive_failures == i + 1

    assert client._consecutive_failures == 3

    # 4th call should skip instantly - reset the mock to track it wasn't called again
    client.session.post.reset_mock()
    resp = await client.query_for_conversation(**QUERY_ARGS)

    assert resp.contexts == []
    assert resp.total_results == 0
    # post() should NOT have been called - we skipped the network call
    client.session.post.assert_not_called()


@pytest.mark.asyncio
async def test_skips_after_mixed_failures(client):
    """Mix of timeouts and HTTP 500s should all count toward the threshold."""
    # 1st: timeout
    _make_timeout_post(client.session)
    await client.query_for_conversation(**QUERY_ARGS)
    assert client._consecutive_failures == 1

    # 2nd: HTTP 500
    _make_500_post(client.session)
    await client.query_for_conversation(**QUERY_ARGS)
    assert client._consecutive_failures == 2

    # 3rd: timeout again
    _make_timeout_post(client.session)
    await client.query_for_conversation(**QUERY_ARGS)
    assert client._consecutive_failures == 3

    # Now should skip
    client.session.post.reset_mock()
    resp = await client.query_for_conversation(**QUERY_ARGS)
    assert resp.contexts == []
    client.session.post.assert_not_called()


@pytest.mark.asyncio
async def test_success_resets_failures(client):
    """A successful response after failures should reset the counter."""
    # 2 failures
    _make_timeout_post(client.session)
    await client.query_for_conversation(**QUERY_ARGS)
    await client.query_for_conversation(**QUERY_ARGS)
    assert client._consecutive_failures == 2

    # Now a success
    _make_success_post(client.session)
    resp = await client.query_for_conversation(**QUERY_ARGS)
    assert client._consecutive_failures == 0
    assert len(resp.contexts) == 1
    assert resp.contexts[0].text == "some context"

    # Should still work (not skipping)
    client.session.post.reset_mock()
    _make_success_post(client.session)
    resp = await client.query_for_conversation(**QUERY_ARGS)
    client.session.post.assert_called_once()


@pytest.mark.asyncio
async def test_cooldown_allows_retry(client):
    """After cooldown period, should allow one more attempt."""
    _make_timeout_post(client.session)

    # Trip the threshold
    for _ in range(3):
        await client.query_for_conversation(**QUERY_ARGS)
    assert client._consecutive_failures == 3

    # Fake the cooldown by backdating _last_failure_time
    client._last_failure_time = time.time() - 31  # 31s ago, cooldown is 30s

    # Should allow a retry now
    client.session.post.reset_mock()
    _make_timeout_post(client.session)
    resp = await client.query_for_conversation(**QUERY_ARGS)

    # post() WAS called (retry allowed)
    client.session.post.assert_called_once()
    assert client._consecutive_failures == 4  # still failing


@pytest.mark.asyncio
async def test_cooldown_retry_succeeds_resets(client):
    """If the retry after cooldown succeeds, the counter resets."""
    _make_timeout_post(client.session)

    for _ in range(3):
        await client.query_for_conversation(**QUERY_ARGS)

    # Backdate and make next call succeed
    client._last_failure_time = time.time() - 31
    _make_success_post(client.session)
    resp = await client.query_for_conversation(**QUERY_ARGS)

    assert client._consecutive_failures == 0
    assert len(resp.contexts) == 1


@pytest.mark.asyncio
async def test_concurrent_calls_skip_when_down(client):
    """Simulates multiple turns firing RAG queries concurrently - all should skip after threshold."""
    _make_timeout_post(client.session)

    # Trip the threshold
    for _ in range(3):
        await client.query_for_conversation(**QUERY_ARGS)

    # Now simulate 5 concurrent calls (like 5 user utterances piling up)
    client.session.post.reset_mock()
    results = await asyncio.gather(*[
        client.query_for_conversation(query=f"turn {i}", collections=["col1"])
        for i in range(5)
    ])

    # All 5 should have been skipped
    for resp in results:
        assert resp.contexts == []
    client.session.post.assert_not_called()


@pytest.mark.asyncio
async def test_skip_is_instant(client):
    """Skipped calls should return in <1ms, not wait for any timeout."""
    _make_timeout_post(client.session)

    for _ in range(3):
        await client.query_for_conversation(**QUERY_ARGS)

    start = time.time()
    for _ in range(100):
        await client.query_for_conversation(**QUERY_ARGS)
    elapsed = time.time() - start

    # 100 skipped calls should complete nearly instantly
    assert elapsed < 0.1, f"100 skipped calls took {elapsed:.3f}s, expected <0.1s"


# ── Edge case tests ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_concurrent_retry_after_cooldown_only_one_gets_through(client):
    """When cooldown expires and 5 calls race, only 1 should hit the network.

    Without the _last_failure_time stamp in _is_available(), all 5 would
    pass the cooldown check and all 5 would make a network call (5 * 5s timeout).
    """
    _make_timeout_post(client.session)

    # Trip the threshold
    for _ in range(3):
        await client.query_for_conversation(**QUERY_ARGS)
    assert client._consecutive_failures == 3

    # Expire the cooldown
    client._last_failure_time = time.time() - 31

    # Fire 5 concurrent calls - only 1 should actually hit the network
    client.session.post.reset_mock()
    _make_timeout_post(client.session)
    results = await asyncio.gather(*[
        client.query_for_conversation(query=f"turn {i}", collections=["col1"])
        for i in range(5)
    ])

    # All return empty (regardless of skip or timeout)
    for resp in results:
        assert resp.contexts == []

    # Only 1 call should have actually hit the network
    assert client.session.post.call_count == 1, (
        f"Expected 1 network call after cooldown, got {client.session.post.call_count}"
    )


@pytest.mark.asyncio
async def test_sustained_downtime_retries_every_cooldown(client):
    """During sustained downtime, we should retry once per cooldown window, not more."""
    _make_timeout_post(client.session)

    # Trip the threshold
    for _ in range(3):
        await client.query_for_conversation(**QUERY_ARGS)

    # Simulate 3 cooldown cycles (RAG stays down the whole time)
    for cycle in range(3):
        client._last_failure_time = time.time() - 31  # expire cooldown
        client.session.post.reset_mock()
        _make_timeout_post(client.session)

        # One retry goes through
        resp = await client.query_for_conversation(**QUERY_ARGS)
        assert resp.contexts == []
        client.session.post.assert_called_once()

        # Immediately after, should skip again
        client.session.post.reset_mock()
        resp = await client.query_for_conversation(**QUERY_ARGS)
        client.session.post.assert_not_called()


@pytest.mark.asyncio
async def test_recovery_after_long_downtime(client):
    """After being down for a long time, the very first call after recovery works."""
    _make_timeout_post(client.session)

    # Trip the threshold
    for _ in range(3):
        await client.query_for_conversation(**QUERY_ARGS)

    # Simulate 10 minutes of downtime (many cooldown cycles pass)
    client._last_failure_time = time.time() - 600

    # RAG is back up now
    _make_success_post(client.session)
    resp = await client.query_for_conversation(**QUERY_ARGS)

    assert client._consecutive_failures == 0
    assert len(resp.contexts) == 1
    assert resp.contexts[0].text == "some context"

    # Normal operation should resume for all subsequent calls
    client.session.post.reset_mock()
    _make_success_post(client.session)
    resp = await client.query_for_conversation(**QUERY_ARGS)
    client.session.post.assert_called_once()
    assert len(resp.contexts) == 1


@pytest.mark.asyncio
async def test_flaky_service_resets_on_each_success(client):
    """A flaky service (alternating success/failure) should never trip the circuit."""
    for _ in range(10):
        # Fail once
        _make_timeout_post(client.session)
        await client.query_for_conversation(**QUERY_ARGS)
        assert client._consecutive_failures == 1

        # Succeed once - resets counter
        _make_success_post(client.session)
        resp = await client.query_for_conversation(**QUERY_ARGS)
        assert client._consecutive_failures == 0
        assert len(resp.contexts) == 1

    # Circuit should never have opened
    assert client._consecutive_failures == 0


@pytest.mark.asyncio
async def test_recovery_mid_burst_unblocks_following_calls(client):
    """After recovery, the next user turns should get RAG context immediately."""
    _make_timeout_post(client.session)

    # Trip the threshold
    for _ in range(3):
        await client.query_for_conversation(**QUERY_ARGS)

    # Cooldown expires, retry succeeds
    client._last_failure_time = time.time() - 31
    _make_success_post(client.session)
    await client.query_for_conversation(**QUERY_ARGS)
    assert client._consecutive_failures == 0

    # Now simulate a burst of 5 concurrent calls - ALL should go through (service is healthy)
    client.session.post.reset_mock()
    _make_success_post(client.session)
    results = await asyncio.gather(*[
        client.query_for_conversation(query=f"turn {i}", collections=["col1"])
        for i in range(5)
    ])

    assert client.session.post.call_count == 5
    for resp in results:
        assert len(resp.contexts) == 1
