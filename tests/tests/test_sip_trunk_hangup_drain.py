"""Regression: the sip-trunk hangup must not cut off audio still buffered in Asterisk.

Asterisk gets audio faster than real time (the output handler paces at 1.5x), so when
teardown runs on the output handler's duration estimate, roughly a third of a long
goodbye is still sitting in Asterisk's frame queue. HANGUP discards it. The fix asks
Asterisk for a QUEUE_DRAINED report and holds HANGUP until it lands.
"""

import asyncio

import pytest

from bolna.input_handlers.telephony_providers import sip_trunk as sip_trunk_input
from bolna.input_handlers.telephony_providers.sip_trunk import SipTrunkInputHandler


class _FakeWebSocket:
    """Records control frames in order; client_state.value == 1 means connected."""

    class _State:
        value = 1

    def __init__(self, fail_send=False):
        self.sent = []
        self.closed = False
        self.fail_send = fail_send
        self.client_state = self._State()

    async def send_text(self, text):
        if self.fail_send:
            raise RuntimeError("socket gone")
        self.sent.append(text)

    async def close(self):
        self.closed = True


def _make_handler(websocket, listen_task):
    handler = SipTrunkInputHandler.__new__(SipTrunkInputHandler)
    handler.websocket = websocket
    handler.websocket_listen_task = listen_task
    handler.channel_id = "test-channel_1"
    handler.running = True
    handler._dtmf_timer_task = None
    handler._queue_drained = asyncio.Event()
    return handler


async def _live_task():
    """Stands in for a _listen() loop that is still receiving."""
    await asyncio.sleep(30)


@pytest.fixture(autouse=True)
def fast_timings(monkeypatch):
    monkeypatch.setattr(sip_trunk_input, "HANGUP_DRAIN_TIMEOUT_S", 0.30)
    monkeypatch.setattr(sip_trunk_input, "HANGUP_DRAIN_SETTLE_S", 0.05)
    monkeypatch.setattr(sip_trunk_input, "HANGUP_SETTLE_S", 0.0)
    monkeypatch.setattr(sip_trunk_input, "HANGUP_DRAIN_MAX_WAIT_S", 3.0)


@pytest.mark.asyncio
async def test_hangup_waits_for_queue_drained_before_disconnecting():
    ws = _FakeWebSocket()
    listen_task = asyncio.create_task(_live_task())
    handler = _make_handler(ws, listen_task)

    async def _drain_soon():
        await asyncio.sleep(0.10)
        await handler._handle_control_message("QUEUE_DRAINED")

    drainer = asyncio.create_task(_drain_soon())
    start = asyncio.get_running_loop().time()
    await handler.stop_handler()
    elapsed = asyncio.get_running_loop().time() - start
    await drainer
    listen_task.cancel()

    assert ws.sent == ["REPORT_QUEUE_DRAINED", "HANGUP"]
    # Held for the drain (0.10s) plus the settle (0.05s), not the full grace.
    assert 0.13 < elapsed < 0.30


@pytest.mark.asyncio
async def test_hangup_is_bounded_when_queue_drained_never_arrives():
    ws = _FakeWebSocket()
    listen_task = asyncio.create_task(_live_task())
    handler = _make_handler(ws, listen_task)

    start = asyncio.get_running_loop().time()
    await asyncio.wait_for(handler.stop_handler(), timeout=5.0)  # hard-fails if it hangs
    elapsed = asyncio.get_running_loop().time() - start
    listen_task.cancel()

    assert ws.sent == ["REPORT_QUEUE_DRAINED", "HANGUP"]
    assert 0.28 < elapsed < 1.0  # bounded by the grace, and no settle on top of it


@pytest.mark.asyncio
async def test_early_queue_drained_is_not_consumed_from_a_previous_report():
    # A QUEUE_DRAINED seen before teardown must not satisfy the teardown wait.
    ws = _FakeWebSocket()
    listen_task = asyncio.create_task(_live_task())
    handler = _make_handler(ws, listen_task)
    await handler._handle_control_message("QUEUE_DRAINED")
    assert handler._queue_drained.is_set()

    start = asyncio.get_running_loop().time()
    await handler.stop_handler()
    elapsed = asyncio.get_running_loop().time() - start
    listen_task.cancel()

    assert elapsed > 0.28  # waited for a fresh report rather than reusing the stale one


@pytest.mark.asyncio
async def test_no_drain_wait_when_receive_loop_already_gone():
    # Caller hung up first: nothing is left to deliver QUEUE_DRAINED, so asking would
    # just burn the full grace on every user-initiated hangup.
    ws = _FakeWebSocket()
    listen_task = asyncio.create_task(asyncio.sleep(0))
    await listen_task
    handler = _make_handler(ws, listen_task)

    start = asyncio.get_running_loop().time()
    await handler.stop_handler()
    elapsed = asyncio.get_running_loop().time() - start

    assert ws.sent == ["HANGUP"]
    assert elapsed < 0.10


@pytest.mark.asyncio
async def test_hangup_still_sent_when_teardown_is_cancelled_during_the_drain():
    # Teardown is routinely cancelled mid-flight (the end_call tool tears the call down from
    # inside the LLM task). Losing HANGUP here leaves the caller on a live channel until the
    # far end gives up, so it has to go out even on the way past a CancelledError.
    ws = _FakeWebSocket()
    listen_task = asyncio.create_task(_live_task())
    handler = _make_handler(ws, listen_task)

    stopping = asyncio.create_task(handler.stop_handler())
    await asyncio.sleep(0.05)  # let it reach the drain wait
    stopping.cancel()
    await asyncio.gather(stopping, return_exceptions=True)
    listen_task.cancel()

    assert ws.sent == ["REPORT_QUEUE_DRAINED", "HANGUP"]
    assert handler.running is False


@pytest.mark.asyncio
async def test_drain_waits_for_output_handler_to_finish_writing():
    # The output handler paces sends at 1.5x real time, so a long goodbye is often still
    # being WRITTEN at teardown. Asking Asterisk for QUEUE_DRAINED before the tail is even
    # sent would hang up over it.
    ws = _FakeWebSocket()
    listen_task = asyncio.create_task(_live_task())
    handler = _make_handler(ws, listen_task)

    class _FakeOutput:
        unsent = True

        def has_unsent_audio(self):
            return self.unsent

    output = _FakeOutput()
    handler.output_handler_ref = output

    async def _finish_sending_then_drain():
        await asyncio.sleep(0.15)
        assert ws.sent == []  # REPORT_QUEUE_DRAINED must not go out while still writing
        output.unsent = False
        await asyncio.sleep(0.15)
        await handler._handle_control_message("QUEUE_DRAINED")

    sender = asyncio.create_task(_finish_sending_then_drain())
    start = asyncio.get_running_loop().time()
    await handler.stop_handler()
    elapsed = asyncio.get_running_loop().time() - start
    await sender
    listen_task.cancel()

    assert ws.sent == ["REPORT_QUEUE_DRAINED", "HANGUP"]
    assert elapsed > 0.30  # held through the write (0.15s) and the playout (0.15s)


@pytest.mark.asyncio
async def test_drain_timeout_scales_with_pending_mark_durations():
    # A flat grace clips any goodbye whose unplayed tail is longer than it; the wait must
    # cover what the pending marks say is still queued in Asterisk.
    ws = _FakeWebSocket()
    listen_task = asyncio.create_task(_live_task())
    handler = _make_handler(ws, listen_task)

    class _Meta:
        mark_event_meta_data = {
            "m1": {"type": "agent_response", "duration": 0.5},
            "m2": {"type": "agent_response", "duration": 0.3},
        }

    handler.mark_event_meta_data = _Meta()

    start = asyncio.get_running_loop().time()
    await asyncio.wait_for(handler.stop_handler(), timeout=5.0)
    elapsed = asyncio.get_running_loop().time() - start
    listen_task.cancel()

    assert ws.sent == ["REPORT_QUEUE_DRAINED", "HANGUP"]
    # 0.8s of unplayed audio + 0.3s grace, not the bare 0.3s grace.
    assert 1.0 < elapsed < 2.0


@pytest.mark.asyncio
async def test_hangup_still_sent_when_report_cannot_be_delivered():
    ws = _FakeWebSocket(fail_send=True)
    listen_task = asyncio.create_task(_live_task())
    handler = _make_handler(ws, listen_task)

    start = asyncio.get_running_loop().time()
    await handler.stop_handler()
    elapsed = asyncio.get_running_loop().time() - start
    listen_task.cancel()

    assert elapsed < 0.10  # gave up immediately instead of waiting on a dead socket
    assert ws.closed
