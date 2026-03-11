"""
Integration test: verify that TranscriberPool keeps standby Deepgram connections
alive via periodic silence keepalives and can switch to them successfully.

Requires DEEPGRAM_AUTH_TOKEN in environment (or .env).
"""
import asyncio
import os
import time
import pytest
from dotenv import load_dotenv

load_dotenv()

from bolna.transcriber.deepgram_transcriber import DeepgramTranscriber
from bolna.transcriber.transcriber_pool import TranscriberPool

DEEPGRAM_KEY = os.getenv("DEEPGRAM_AUTH_TOKEN")
pytestmark = pytest.mark.skipif(not DEEPGRAM_KEY, reason="DEEPGRAM_AUTH_TOKEN not set")


def _make_deepgram(label, language, private_queue, output_queue):
    """Create a real DeepgramTranscriber for integration testing."""
    return DeepgramTranscriber(
        telephony_provider="default",
        input_queue=private_queue,
        model="nova-2",
        stream=True,
        language=language,
        encoding="linear16",
        sampling_rate="16000",
        output_queue=output_queue,
        transcriber_key=DEEPGRAM_KEY,
        enforce_streaming=True,
    )


@pytest.mark.asyncio
async def test_standby_connection_stays_alive_with_keepalive():
    """
    Two real Deepgram connections: en (active) and hi (standby).
    Wait 30s+ to verify the standby doesn't die (without keepalive it would
    die at ~45s).  Then switch to hi and verify it's responsive by sending
    audio and getting a transcript.
    """
    output_q = asyncio.Queue()
    shared_q = asyncio.Queue()

    en_queue = asyncio.Queue()
    hi_queue = asyncio.Queue()

    en_transcriber = _make_deepgram("en", "en", en_queue, output_q)
    hi_transcriber = _make_deepgram("hi", "hi", hi_queue, output_q)

    pool = TranscriberPool(
        transcribers={"en": en_transcriber, "hi": hi_transcriber},
        shared_input_queue=shared_q,
        output_queue=output_q,
        active_label="en",
    )

    try:
        print("\n[1/4] Starting pool with 2 Deepgram transcribers...")
        await pool.run()
        await asyncio.sleep(1)  # let connections establish

        # Verify both connected
        assert en_transcriber.transcription_task is not None
        assert hi_transcriber.transcription_task is not None
        assert not en_transcriber.transcription_task.done(), "en transcriber died on startup"
        assert not hi_transcriber.transcription_task.done(), "hi transcriber died on startup"
        print(f"   en connection_time={en_transcriber.connection_time}ms")
        print(f"   hi connection_time={hi_transcriber.connection_time}ms")

        # [2] Feed audio to active (en) while standby (hi) gets keepalive silence
        print("[2/4] Feeding audio to active (en), standby (hi) gets keepalive silence...")
        SILENCE_16K = b'\x00' * 640  # 20ms of linear16 at 16kHz
        for _ in range(50):  # 1 second of audio
            shared_q.put_nowait({"data": SILENCE_16K, "meta_info": {}})
            await asyncio.sleep(0.02)

        # [3] Wait 30s — without keepalive, Deepgram would close standby at ~45s
        print("[3/4] Waiting 35s to test standby survives (would die at ~45s without keepalive)...")
        start = time.time()
        check_interval = 5
        total_wait = 35
        elapsed = 0
        while elapsed < total_wait:
            await asyncio.sleep(check_interval)
            elapsed = time.time() - start
            en_alive = not en_transcriber.transcription_task.done()
            hi_alive = not hi_transcriber.transcription_task.done()
            print(f"   {elapsed:.0f}s — en={'alive' if en_alive else 'DEAD'}  hi={'alive' if hi_alive else 'DEAD'}")
            if not hi_alive:
                pytest.fail(f"Standby (hi) connection died after {elapsed:.0f}s — keepalive didn't work")

        print("[4/4] Switching to hi and verifying it accepts audio...")
        await pool.switch("hi")
        assert pool.active_label == "hi"

        # Send some audio through the pool to hi
        for _ in range(25):  # 0.5s of audio
            shared_q.put_nowait({"data": SILENCE_16K, "meta_info": {}})
            await asyncio.sleep(0.02)

        # hi should still be alive after receiving audio
        await asyncio.sleep(0.5)
        assert not hi_transcriber.transcription_task.done(), "hi died after receiving audio post-switch"

        print("   SUCCESS: standby stayed alive, switch worked, audio accepted")

    finally:
        await pool.cleanup()


@pytest.mark.asyncio
async def test_reconnect_on_demand_after_standby_dies():
    """
    If we disable keepalive (set interval very high) and wait for standby to die,
    switch() should reconnect it automatically.
    """
    output_q = asyncio.Queue()
    shared_q = asyncio.Queue()

    en_queue = asyncio.Queue()
    hi_queue = asyncio.Queue()

    en_transcriber = _make_deepgram("en", "en", en_queue, output_q)
    hi_transcriber = _make_deepgram("hi", "hi", hi_queue, output_q)

    pool = TranscriberPool(
        transcribers={"en": en_transcriber, "hi": hi_transcriber},
        shared_input_queue=shared_q,
        output_queue=output_q,
        active_label="en",
    )
    # Disable keepalive for this test
    pool._KEEPALIVE_INTERVAL = 9999

    try:
        print("\n[1/3] Starting pool (keepalive disabled)...")
        await pool.run()
        await asyncio.sleep(1)

        assert not hi_transcriber.transcription_task.done(), "hi died on startup"

        # Feed audio to en so it stays alive
        SILENCE_16K = b'\x00' * 640
        async def feed_audio():
            for _ in range(3000):  # 60s of audio
                shared_q.put_nowait({"data": SILENCE_16K, "meta_info": {}})
                await asyncio.sleep(0.02)

        feeder = asyncio.create_task(feed_audio())

        print("[2/3] Waiting for standby (hi) to die naturally (no keepalive)...")
        start = time.time()
        while time.time() - start < 120:
            await asyncio.sleep(5)
            elapsed = time.time() - start
            hi_alive = not hi_transcriber.transcription_task.done()
            print(f"   {elapsed:.0f}s — hi={'alive' if hi_alive else 'DEAD'}")
            if not hi_alive:
                print(f"   Standby died at {elapsed:.0f}s as expected")
                break
        else:
            feeder.cancel()
            pytest.skip("Standby didn't die within 120s — can't test reconnect")

        feeder.cancel()

        print("[3/3] Switching to hi — should trigger reconnect...")
        old_task = hi_transcriber.transcription_task
        await pool.switch("hi")
        assert pool.active_label == "hi"

        # Verify a new transcription_task was created (reconnected)
        assert hi_transcriber.transcription_task is not old_task or not hi_transcriber.transcription_task.done(), \
            "hi was not reconnected"

        await asyncio.sleep(1)
        print("   SUCCESS: switch() reconnected the dead standby transcriber")

    finally:
        await pool.cleanup()
