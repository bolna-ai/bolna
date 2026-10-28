"""A language whose transcriber fails moves to its next fallback provider instead of retrying one that cannot
connect; only the provider now serving it speaks for the caller."""

import asyncio
from unittest.mock import AsyncMock

from bolna.helpers.fallbacks import FallbackEntry
from bolna.helpers.utils import create_ws_data_packet
from bolna.transcriber.transcriber_pool import TranscriberPool


class FakeTranscriber:
    def __init__(self, model, connected_before=False, billed_s=0.0):
        self.model = model
        self.run = AsyncMock()
        self.toggle_connection = AsyncMock()
        self.cleanup = AsyncMock()
        self.input_queue = asyncio.Queue()
        self.transcriber_output_queue = asyncio.Queue()
        self.transcription_task = None
        self.connection_time = 120 if connected_before else None
        self.connection_error = None
        self.connection_on = True
        self.turn_counter = 0
        self.turn_latencies = []
        self.context_source = None
        self.billed_s = billed_s

    def is_connected(self):
        return False

    def set_agent_context_source(self, source):
        self.context_source = source

    def billed_audio_s(self):
        return self.billed_s

    async def emit(self, data):
        await self.transcriber_output_queue.put(create_ws_data_packet(data, {}))


def entry(transcriber, builds=None):
    def build():
        if builds is not None:
            builds.append(transcriber.model)
        return transcriber

    return FallbackEntry("sarvam", transcriber.model, build)


async def started_pool(primary, *fallbacks):
    switches = []
    call_queue = asyncio.Queue()
    pool = TranscriberPool(
        transcribers={"hi": primary},
        shared_input_queue=asyncio.Queue(),
        output_queue=call_queue,
        active_label="hi",
        multilingual_config={},
        lazy_legs=True,
        fallbacks={"hi": list(fallbacks)},
        on_fallback=lambda label, failed, e, error: switches.append((failed.model, e.model, error)),
    )
    await pool.run()
    return pool, call_queue, switches


async def settle():
    for _ in range(5):
        await asyncio.sleep(0)


async def test_a_provider_that_never_connected_moves_to_the_next_fallback_without_a_retry():
    primary = FakeTranscriber("nova-3")
    backup = FakeTranscriber("saaras:v3")
    pool, _, switches = await started_pool(primary, entry(backup))
    primary.turn_counter = 4
    primary.run.reset_mock()

    assert await pool.recover_active("HTTP 429") is True
    await settle()

    primary.run.assert_not_awaited()
    backup.run.assert_awaited_once()
    assert pool.transcribers["hi"] is backup
    assert backup.turn_counter == 4
    assert switches == [("nova-3", "saaras:v3", "HTTP 429")]
    primary.toggle_connection.assert_awaited_once()
    primary.cleanup.assert_awaited_once()
    await pool.cleanup()


async def test_a_connection_that_had_been_working_gets_one_reconnect_before_moving_on():
    primary = FakeTranscriber("nova-3", connected_before=True)
    backup = FakeTranscriber("saaras:v3")
    pool, _, switches = await started_pool(primary, entry(backup))
    primary.run.reset_mock()

    assert await pool.recover_active("socket dropped") is True
    primary.run.assert_awaited_once()
    assert pool.transcribers["hi"] is primary

    assert await pool.recover_active("socket dropped again") is True
    await settle()
    assert pool.transcribers["hi"] is backup
    assert switches == [("nova-3", "saaras:v3", "socket dropped again")]
    await pool.cleanup()


async def test_only_the_provider_now_serving_reaches_the_call():
    primary = FakeTranscriber("nova-3")
    backup = FakeTranscriber("saaras:v3")
    pool, call_queue, _ = await started_pool(primary, entry(backup))
    await pool.recover_active("HTTP 429")
    await settle()

    await primary.emit("transcriber_connection_closed")
    await backup.emit({"type": "transcript", "content": "haan ji"})
    await settle()

    assert [call_queue.get_nowait()["data"] for _ in range(call_queue.qsize())] == [
        {"type": "transcript", "content": "haan ji"}
    ]
    await pool.cleanup()


async def test_a_finished_call_never_moves_to_a_fallback():
    primary = FakeTranscriber("nova-3")
    pool, _, switches = await started_pool(primary, entry(FakeTranscriber("saaras:v3")))
    pool.call_ended = True

    assert await pool.recover_active("closed after hangup") is False
    assert pool.transcribers["hi"] is primary
    assert switches == []
    await pool.cleanup()


async def test_once_the_fallbacks_are_spent_the_last_provider_is_reconnected_as_any_provider_is():
    backup = FakeTranscriber("saaras:v3")
    pool, _, _ = await started_pool(FakeTranscriber("nova-3"), entry(backup))
    await pool.recover_active("HTTP 429")
    await settle()
    backup.run.reset_mock()

    pool.active_reconnect_count = TranscriberPool._MAX_RECONNECTS_PER_CALL - 1
    assert await pool.recover_active("HTTP 429") is True
    backup.run.assert_awaited_once()
    assert await pool.recover_active("HTTP 429") is False
    await pool.cleanup()


async def test_fallbacks_are_built_only_when_needed():
    builds = []
    pool, _, _ = await started_pool(FakeTranscriber("nova-3"), entry(FakeTranscriber("saaras:v3"), builds))
    assert builds == []
    await pool.recover_active("HTTP 429")
    assert builds == ["saaras:v3"]
    await pool.cleanup()


async def test_the_failed_providers_audio_still_counts_toward_usage():
    pool, _, _ = await started_pool(FakeTranscriber("nova-3", billed_s=12.5), entry(FakeTranscriber("saaras:v3")))
    await pool.recover_active("HTTP 429")
    pool.transcribers["hi"].billed_s = 30.0

    assert pool.billed_audio_s() == 42.5
    await pool.cleanup()


async def test_without_fallbacks_recovery_reconnects_the_same_provider():
    primary = FakeTranscriber("nova-3")
    pool = TranscriberPool(
        transcribers={"hi": primary},
        shared_input_queue=asyncio.Queue(),
        output_queue=asyncio.Queue(),
        active_label="hi",
        multilingual_config={},
    )
    assert await pool.recover_active("socket dropped") is True
    primary.run.assert_awaited_once()
    assert pool.active_reconnect_count == 1
