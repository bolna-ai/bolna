"""AzureTranscriber tears the native recognizer down off the event loop.

The SDK's recognizer teardown blocks for seconds against a degraded endpoint. Running it inline
on the loop freezes every other call on the worker, which stays silent until /healthz starves.
These pin the end-of-stream teardown to a worker thread so a slow teardown cannot block the loop.
"""

import asyncio
import threading
import time

from bolna.transcriber.azure_transcriber import AzureTranscriber


def _transcriber():
    # telephony_provider="twilio" takes the 8k mulaw telephony branch; no network until run().
    return AzureTranscriber(
        telephony_provider="twilio",
        input_queue=asyncio.Queue(),
        output_queue=asyncio.Queue(),
    )


async def test_end_of_stream_tears_down_off_the_loop():
    t = _transcriber()
    ran_on_main = {}

    def slow_cleanup():
        ran_on_main["value"] = threading.current_thread() is threading.main_thread()
        time.sleep(0.2)

    t._sync_cleanup = slow_cleanup

    ticks = 0

    async def heartbeat():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    hb = asyncio.create_task(heartbeat())
    result = await t._check_and_process_end_of_stream({"meta_info": {"eos": True}})
    hb.cancel()

    assert result is True
    assert ran_on_main["value"] is False
    assert ticks > 0


async def test_non_end_of_stream_leaves_the_connection_open():
    t = _transcriber()
    called = False

    def cleanup():
        nonlocal called
        called = True

    t._sync_cleanup = cleanup
    result = await t._check_and_process_end_of_stream({"meta_info": {}})

    assert result is False
    assert called is False
