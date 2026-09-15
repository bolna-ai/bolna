"""Azure reports its own connection liveness, and a dead single transcriber reconnects.

Azure is SDK-driven and never sets transcription_task, so TranscriberPool's old task probe read it
as permanently dead: every standby close was misread as the active transcriber dying and burned the
reconnect budget. During the 2026-09 centralindia outage that ended calls with
hangup_reason=transcriber_connection_error in ~5s. Single-transcriber agents had no recovery at all.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from bolna.agent_manager.task_manager import TaskManager
from bolna.transcriber.azure_transcriber import AzureTranscriber
from bolna.transcriber.transcriber_pool import TranscriberPool


def _azure():
    # telephony_provider="twilio" takes the 8k mulaw branch; no network until run().
    return AzureTranscriber(
        telephony_provider="twilio",
        input_queue=asyncio.Queue(),
        output_queue=asyncio.Queue(),
    )


def _cancel_event(error_details="WS_OPEN_ERROR_UNDERLYING_IO_OPEN_FAILED"):
    return SimpleNamespace(cancellation_details=SimpleNamespace(reason="Error", error_details=error_details))


async def test_azure_reports_dead_until_the_session_starts():
    t = _azure()
    assert t.is_connected() is False
    await t.session_started_handler(SimpleNamespace())
    assert t.is_connected() is True


async def test_a_refused_connection_reports_dead_and_keeps_the_error():
    t = _azure()
    await t.session_started_handler(SimpleNamespace())
    await t.canceled_handler(_cancel_event())
    assert t.is_connected() is False
    assert t.connection_error == "WS_OPEN_ERROR_UNDERLYING_IO_OPEN_FAILED"


async def test_session_stop_reports_dead():
    t = _azure()
    await t.session_started_handler(SimpleNamespace())
    await t.session_stopped_handler(SimpleNamespace())
    assert t.is_connected() is False


async def test_pool_asks_azure_instead_of_probing_its_task():
    # The regression: a live Azure active transcriber must not read as dead just because
    # transcription_task is None, or the standby's close ends the call.
    active, standby = _azure(), _azure()
    await active.session_started_handler(SimpleNamespace())
    pool = TranscriberPool(
        transcribers={"kn": active, "hi": standby},
        shared_input_queue=asyncio.Queue(),
        output_queue=asyncio.Queue(),
        active_label="kn",
        multilingual_config={},
    )
    assert active.transcription_task is None
    assert pool.is_active_transcriber_alive() is True

    await active.canceled_handler(_cancel_event())
    assert pool.is_active_transcriber_alive() is False


async def test_single_transcriber_reconnects_until_the_cap():
    tm = SimpleNamespace(
        single_transcriber_reconnect_count=0,
        MAX_SINGLE_TRANSCRIBER_RECONNECTS=TaskManager.MAX_SINGLE_TRANSCRIBER_RECONNECTS,
        tools={"transcriber": MagicMock(run=AsyncMock())},
    )
    for attempt in range(1, TaskManager.MAX_SINGLE_TRANSCRIBER_RECONNECTS + 1):
        assert await TaskManager.reconnect_single_transcriber(tm) is True
        assert tm.single_transcriber_reconnect_count == attempt
    # Budget spent — the call ends rather than reconnecting forever.
    assert await TaskManager.reconnect_single_transcriber(tm) is False


async def test_single_transcriber_reconnect_failure_does_not_consume_budget():
    tm = SimpleNamespace(
        single_transcriber_reconnect_count=0,
        MAX_SINGLE_TRANSCRIBER_RECONNECTS=TaskManager.MAX_SINGLE_TRANSCRIBER_RECONNECTS,
        tools={"transcriber": MagicMock(run=AsyncMock(side_effect=RuntimeError("connect refused")))},
    )
    assert await TaskManager.reconnect_single_transcriber(tm) is False
    assert tm.single_transcriber_reconnect_count == 0


async def test_reconnect_does_not_leave_the_previous_audio_pump_running():
    # The single-transcriber reconnect path above calls run() repeatedly. Each call used to add
    # a consumer on the same input_queue while only tracking the newest, so the pumps raced
    # frames into the shared push stream.
    t = _azure()

    async def skip_native_sdk():
        t.connection_error = None

    t.initialize_connection = skip_native_sdk

    for _ in range(TaskManager.MAX_SINGLE_TRANSCRIBER_RECONNECTS):
        await t.run()
        await asyncio.sleep(0)

    pumps = [task for task in asyncio.all_tasks() if "send_audio_to_transcriber" in str(task.get_coro())]
    assert len(pumps) == 1
    assert t.send_audio_to_transcriber_task is pumps[0]
    await t.cancel_audio_pump()


async def test_the_single_transcriber_reconnect_clears_a_stale_connection_error():
    # Runs for any provider; non-azure ones never clear it themselves.
    transcriber = MagicMock(run=AsyncMock(), connection_error="socket died")
    tm = SimpleNamespace(
        single_transcriber_reconnect_count=0,
        MAX_SINGLE_TRANSCRIBER_RECONNECTS=TaskManager.MAX_SINGLE_TRANSCRIBER_RECONNECTS,
        tools={"transcriber": transcriber},
    )
    assert await TaskManager.reconnect_single_transcriber(tm) is True
    assert transcriber.connection_error is None
