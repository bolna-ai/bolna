import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from bolna.transcriber.transcriber_pool import TranscriberPool


def _make_mock_transcriber(label, encoding="linear16"):
    """Create a mock transcriber with the interface TranscriberPool expects."""
    t = MagicMock()
    t.input_queue = asyncio.Queue()
    t.connection_time = 42.0
    t.turn_latencies = [{"turn_id": f"{label}_turn1"}]
    t.get_meta_info.return_value = {"label": label}
    t.encoding = encoding
    t.transcription_task = MagicMock()
    t.transcription_task.done.return_value = False  # connection alive
    t.run = AsyncMock()
    t.toggle_connection = AsyncMock()
    t.cleanup = AsyncMock()
    return t


@pytest.fixture
def pool():
    """Standard two-transcriber pool (en active, hi standby)."""
    transcribers = {
        "en": _make_mock_transcriber("en"),
        "hi": _make_mock_transcriber("hi"),
    }
    shared_q = asyncio.Queue()
    output_q = asyncio.Queue()
    p = TranscriberPool(
        transcribers=transcribers,
        shared_input_queue=shared_q,
        output_queue=output_q,
        active_label="en",
    )
    return p


@pytest.mark.asyncio
async def test_audio_routes_to_active_only(pool):
    """Audio placed in shared queue should only reach the active transcriber's queue."""
    await pool.run()
    # Let router start
    await asyncio.sleep(0.05)

    pool.shared_input_queue.put_nowait({"data": b"audio1", "meta_info": {}})
    pool.shared_input_queue.put_nowait({"data": b"audio2", "meta_info": {}})
    await asyncio.sleep(0.05)

    # Active (en) should have both packets
    assert pool.transcribers["en"].input_queue.qsize() == 2
    # Standby (hi) should have none
    assert pool.transcribers["hi"].input_queue.qsize() == 0

    await pool.cleanup()


@pytest.mark.asyncio
async def test_switch_changes_routing(pool):
    """After switch, audio should go to the new active transcriber."""
    await pool.run()
    await asyncio.sleep(0.05)

    # Send one packet to en
    pool.shared_input_queue.put_nowait({"data": b"to_en", "meta_info": {}})
    await asyncio.sleep(0.05)
    assert pool.transcribers["en"].input_queue.qsize() == 1

    # Switch to hi
    await pool.switch("hi")
    assert pool.active_label == "hi"

    # Send another packet — should go to hi
    pool.shared_input_queue.put_nowait({"data": b"to_hi", "meta_info": {}})
    await asyncio.sleep(0.05)
    assert pool.transcribers["hi"].input_queue.qsize() == 1
    # en should still have only 1
    assert pool.transcribers["en"].input_queue.qsize() == 1

    await pool.cleanup()


@pytest.mark.asyncio
async def test_switch_no_op_same_label(pool):
    """Switching to the already-active label should be a no-op."""
    await pool.switch("en")
    assert pool.active_label == "en"


@pytest.mark.asyncio
async def test_switch_invalid_label_raises(pool):
    """Switching to a nonexistent label should raise ValueError."""
    with pytest.raises(ValueError, match="Unknown transcriber label"):
        await pool.switch("fr")


@pytest.mark.asyncio
async def test_cleanup_stops_all(pool):
    """cleanup() should cancel the router, keepalive, and clean up every transcriber."""
    await pool.run()
    await asyncio.sleep(0.05)

    await pool.cleanup()

    # Router task should be done
    assert pool._router_task.done()
    # Keepalive task should be done
    assert pool._keepalive_task.done()
    # Each transcriber's cleanup should have been called
    for t in pool.transcribers.values():
        t.cleanup.assert_awaited_once()


@pytest.mark.asyncio
async def test_connection_time_delegates_to_active(pool):
    """connection_time should return the active transcriber's value."""
    pool.transcribers["en"].connection_time = 100
    pool.transcribers["hi"].connection_time = 200
    assert pool.connection_time == 100

    await pool.switch("hi")
    assert pool.connection_time == 200


@pytest.mark.asyncio
async def test_turn_latencies_aggregates_all(pool):
    """turn_latencies should aggregate from all transcribers."""
    pool.transcribers["en"].turn_latencies = [{"turn_id": "en1"}, {"turn_id": "en2"}]
    pool.transcribers["hi"].turn_latencies = [{"turn_id": "hi1"}]
    latencies = pool.turn_latencies
    assert len(latencies) == 3
    turn_ids = {l["turn_id"] for l in latencies}
    assert turn_ids == {"en1", "en2", "hi1"}


@pytest.mark.asyncio
async def test_toggle_connection_stops_all(pool):
    """toggle_connection() should call toggle on every transcriber."""
    await pool.toggle_connection()
    for t in pool.transcribers.values():
        t.toggle_connection.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_meta_info_delegates_to_active(pool):
    """get_meta_info should delegate to the active transcriber."""
    result = pool.get_meta_info()
    assert result == {"label": "en"}
    pool.transcribers["en"].get_meta_info.assert_called_once()


# ------------------------------------------------------------------
# Standby keepalive tests
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_standby_keepalive_sends_silence_to_standby():
    """Keepalive task should send silence frames to standby transcribers, not active."""
    transcribers = {
        "en": _make_mock_transcriber("en"),
        "hi": _make_mock_transcriber("hi"),
        "ta": _make_mock_transcriber("ta"),
    }
    shared_q = asyncio.Queue()
    output_q = asyncio.Queue()
    p = TranscriberPool(
        transcribers=transcribers,
        shared_input_queue=shared_q,
        output_queue=output_q,
        active_label="en",
    )
    # Use a very short interval for testing
    p._KEEPALIVE_INTERVAL = 0.1

    await p.run()
    # Wait for at least one keepalive cycle
    await asyncio.sleep(0.25)

    # Active (en) should have received NO keepalive silence
    assert p.transcribers["en"].input_queue.qsize() == 0
    # Standby (hi, ta) should each have received at least 1 silence packet
    assert p.transcribers["hi"].input_queue.qsize() >= 1
    assert p.transcribers["ta"].input_queue.qsize() >= 1

    # Verify the packet is silence
    pkt = p.transcribers["hi"].input_queue.get_nowait()
    assert pkt["data"] == b'\x00' * 320  # linear16 silence
    assert pkt["meta_info"] == {}

    await p.cleanup()


@pytest.mark.asyncio
async def test_standby_keepalive_uses_correct_encoding():
    """Keepalive should use mulaw silence (0xFF) for mulaw transcribers."""
    transcribers = {
        "en": _make_mock_transcriber("en", encoding="linear16"),
        "hi": _make_mock_transcriber("hi", encoding="mulaw"),
    }
    shared_q = asyncio.Queue()
    output_q = asyncio.Queue()
    p = TranscriberPool(
        transcribers=transcribers,
        shared_input_queue=shared_q,
        output_queue=output_q,
        active_label="en",
    )
    p._KEEPALIVE_INTERVAL = 0.1

    await p.run()
    await asyncio.sleep(0.2)

    pkt = p.transcribers["hi"].input_queue.get_nowait()
    assert pkt["data"] == b'\xff' * 320  # mulaw silence

    await p.cleanup()


@pytest.mark.asyncio
async def test_standby_keepalive_skips_dead_transcriber():
    """Keepalive should skip transcribers whose connection already dropped."""
    transcribers = {
        "en": _make_mock_transcriber("en"),
        "hi": _make_mock_transcriber("hi"),
    }
    # Mark hi's transcription_task as done (connection dropped)
    transcribers["hi"].transcription_task.done.return_value = True

    shared_q = asyncio.Queue()
    output_q = asyncio.Queue()
    p = TranscriberPool(
        transcribers=transcribers,
        shared_input_queue=shared_q,
        output_queue=output_q,
        active_label="en",
    )
    p._KEEPALIVE_INTERVAL = 0.1

    await p.run()
    await asyncio.sleep(0.25)

    # Dead standby should NOT receive keepalive
    assert p.transcribers["hi"].input_queue.qsize() == 0

    await p.cleanup()


# ------------------------------------------------------------------
# Reconnect-on-demand tests
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_switch_reconnects_dead_transcriber():
    """switch() should call run() on a transcriber whose connection has dropped."""
    transcribers = {
        "en": _make_mock_transcriber("en"),
        "hi": _make_mock_transcriber("hi"),
    }
    # Mark hi as dead
    transcribers["hi"].transcription_task.done.return_value = True

    shared_q = asyncio.Queue()
    output_q = asyncio.Queue()
    p = TranscriberPool(
        transcribers=transcribers,
        shared_input_queue=shared_q,
        output_queue=output_q,
        active_label="en",
    )

    await p.switch("hi")

    # Should have reconnected
    transcribers["hi"].run.assert_awaited_once()
    assert p.active_label == "hi"


@pytest.mark.asyncio
async def test_switch_does_not_reconnect_alive_transcriber():
    """switch() should NOT call run() if the target transcriber is still connected."""
    transcribers = {
        "en": _make_mock_transcriber("en"),
        "hi": _make_mock_transcriber("hi"),
    }
    # hi is alive
    transcribers["hi"].transcription_task.done.return_value = False

    shared_q = asyncio.Queue()
    output_q = asyncio.Queue()
    p = TranscriberPool(
        transcribers=transcribers,
        shared_input_queue=shared_q,
        output_queue=output_q,
        active_label="en",
    )

    await p.switch("hi")

    # Should NOT have called run()
    transcribers["hi"].run.assert_not_awaited()
    assert p.active_label == "hi"


@pytest.mark.asyncio
async def test_silence_frame_linear16():
    """_silence_frame should return zeros for linear16."""
    frame = TranscriberPool._silence_frame("linear16")
    assert frame == b'\x00' * 320
    assert len(frame) == 320


@pytest.mark.asyncio
async def test_silence_frame_mulaw():
    """_silence_frame should return 0xFF bytes for mulaw."""
    frame = TranscriberPool._silence_frame("mulaw")
    assert frame == b'\xff' * 320
    assert len(frame) == 320
