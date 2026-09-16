"""Azure reports its own connection liveness, and a dead single transcriber reconnects.

Azure is SDK-driven and never sets transcription_task, so TranscriberPool's old task probe read it
as permanently dead: every standby close was misread as the active transcriber dying and burned the
reconnect budget. During the 2026-09 centralindia outage that ended calls with
hangup_reason=transcriber_connection_error in ~5s. Single-transcriber agents had no recovery at all.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

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


async def test_reconnect_does_not_leave_the_previous_audio_pump_running():
    # TranscriberPool.reconnect_active() calls run() repeatedly. Each call used to add a consumer
    # on the same input_queue while only tracking the newest, so the pumps raced frames into the
    # shared push stream.
    t = _azure()

    async def skip_native_sdk():
        t.connection_error = None

    t.initialize_connection = skip_native_sdk

    for _ in range(TranscriberPool._MAX_RECONNECTS_PER_CALL):
        await t.run()
        await asyncio.sleep(0)

    pumps = [task for task in asyncio.all_tasks() if "send_audio_to_transcriber" in str(task.get_coro())]
    assert len(pumps) == 1
    assert t.send_audio_to_transcriber_task is pumps[0]
    await t.cancel_audio_pump()
