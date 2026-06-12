"""Unit tests for TranscriberPool.reconnect_active — the mid-call recovery path
for active transcriber sockets that die (sarvam saarika drops connections within
seconds; without reconnect the pool's closed-handler ends the call).
"""

import asyncio
from unittest.mock import AsyncMock

import pytest

from bolna.transcriber.transcriber_pool import TranscriberPool


class FakeTranscriber:
    def __init__(self):
        self.run = AsyncMock()
        self.input_queue = asyncio.Queue()
        self.transcription_task = None


def _pool():
    transcribers = {"hi": FakeTranscriber(), "ta": FakeTranscriber()}
    pool = TranscriberPool(
        transcribers=transcribers,
        shared_input_queue=asyncio.Queue(),
        output_queue=asyncio.Queue(),
        active_label="ta",
        multilingual_config={},
    )
    return pool, transcribers


@pytest.mark.asyncio
async def test_reconnect_active_success_restarts_and_counts():
    pool, transcribers = _pool()
    assert await pool.reconnect_active() is True
    transcribers["ta"].run.assert_awaited_once()
    # Only the ACTIVE transcriber is touched — standbys reconnect at switch time.
    transcribers["hi"].run.assert_not_awaited()
    assert pool.reconnect_count == 1


@pytest.mark.asyncio
async def test_reconnect_active_provider_failure_returns_false():
    pool, transcribers = _pool()
    transcribers["ta"].run.side_effect = RuntimeError("connect refused")
    assert await pool.reconnect_active() is False
    # A failed attempt must not consume reconnect budget.
    assert pool.reconnect_count == 0


@pytest.mark.asyncio
async def test_reconnect_active_respects_per_call_cap():
    pool, transcribers = _pool()
    pool.reconnect_count = TranscriberPool._MAX_RECONNECTS_PER_CALL
    assert await pool.reconnect_active() is False
    transcribers["ta"].run.assert_not_awaited()
