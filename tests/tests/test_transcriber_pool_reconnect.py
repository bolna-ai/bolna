"""Unit tests for TranscriberPool.reconnect_active — the mid-call recovery path
for active transcriber sockets that die (sarvam saarika drops connections within
seconds; without reconnect the pool's closed-handler ends the call).
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

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


@pytest.mark.asyncio
async def test_reconnect_active_refused_after_eos():
    # The active transcriber closing AFTER eos is the hangup teardown trigger —
    # reconnecting it resurrects a zombie call (QA f544513a: 33 min post-hangup).
    pool, transcribers = _pool()
    pool.call_ended = True
    assert await pool.reconnect_active() is False
    transcribers["ta"].run.assert_not_awaited()


@pytest.mark.asyncio
async def test_audio_router_sets_call_ended_on_eos_and_still_forwards():
    pool, transcribers = _pool()
    pool.shared_input_queue.put_nowait({"data": b"\x00\x00", "meta_info": {"io": "plivo"}})
    pool.shared_input_queue.put_nowait({"data": None, "meta_info": {"io": "default", "eos": True}})
    router = asyncio.create_task(pool._audio_router())
    await asyncio.sleep(0.01)  # let the router drain both packets
    router.cancel()
    assert pool.call_ended is True
    # The eos packet must still reach the active transcriber — its closure drives teardown.
    assert transcribers["ta"].input_queue.qsize() == 2


@pytest.mark.asyncio
async def test_switch_does_not_reconnect_dropped_target_after_eos():
    pool, transcribers = _pool()
    done_task = MagicMock()
    done_task.done.return_value = True
    transcribers["hi"].transcription_task = done_task
    pool.call_ended = True
    await pool.switch("hi")
    transcribers["hi"].run.assert_not_awaited()
    assert pool.active_label == "hi"
