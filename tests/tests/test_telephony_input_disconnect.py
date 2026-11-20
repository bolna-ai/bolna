"""Tests for TelephonyInputHandler._listen()'s WebSocketDisconnect handling.

The WebSocketDisconnect branch used to only log the hangup, without breaking the
loop or emitting an EOS packet -- unlike the sibling `stop`-event and generic
except-Exception branches, which both do. Since Starlette raises RuntimeError on
any receive() call after a disconnect has already been received, the loop would
call receive_text() again, hit that RuntimeError, and fall into the generic
except branch: a spurious traceback for what was a completely normal hangup, and
EOS sent one loop iteration late. These drive the real method against a real
starlette.websockets.WebSocket (fed through a fake ASGI receive channel), so the
disconnect-state transition being asserted on is Starlette's actual behavior, not
a guess about it.
"""

import io
import sys

import pytest
from starlette.websockets import WebSocket, WebSocketState

from bolna.input_handlers.telephony import TelephonyInputHandler


class _FakeQueue:
    def __init__(self):
        self.items = []

    def put_nowait(self, item):
        self.items.append(item)


class _OneShotDisconnectChannel:
    """Fake ASGI receive() callable returning a single disconnect message."""

    def __init__(self, code):
        self.code = code
        self.call_count = 0

    async def __call__(self):
        self.call_count += 1
        assert self.call_count == 1, "receive() invoked again -- _listen() did not break"
        return {"type": "websocket.disconnect", "code": self.code}


def _make_handler(code):
    channel = _OneShotDisconnectChannel(code)
    ws = WebSocket(scope={"type": "websocket"}, receive=channel, send=None)
    ws.client_state = WebSocketState.CONNECTED
    ws.application_state = WebSocketState.CONNECTED

    queues = {"transcriber": _FakeQueue(), "dtmf": _FakeQueue()}
    handler = TelephonyInputHandler(queues=queues, websocket=ws, input_types={"audio": 0})
    return handler, channel


@pytest.mark.asyncio
@pytest.mark.parametrize("code", [1000, 1001, 1006])
async def test_disconnect_breaks_loop_and_sends_eos(code):
    """Every disconnect code -- clean (1000/1001) or abnormal (1006) -- must break
    the loop and emit exactly one EOS packet, with no second receive() call.

    NOTE: the buggy (pre-fix) code also ends up with one EOS packet and one
    channel call -- it just gets there via a SECOND receive_text() call that
    raises RuntimeError('Cannot call "receive" once a disconnect message has
    been received.'), caught by the generic except-Exception branch (which
    also emits EOS+break). The queue/call-count alone can't tell the two apart;
    the real signal is whether that spurious RuntimeError/traceback fired at
    all, which is what's actually asserted on below.
    """
    handler, channel = _make_handler(code)

    captured_stderr = io.StringIO()
    real_stderr = sys.stderr
    sys.stderr = captured_stderr
    try:
        await handler._listen()
    finally:
        sys.stderr = real_stderr

    printed = captured_stderr.getvalue()
    assert "RuntimeError" not in printed and "Cannot call" not in printed, (
        f"disconnect (code={code}) fell through to the generic except-Exception "
        f"branch via a spurious RuntimeError instead of being handled directly "
        f"by the WebSocketDisconnect branch:\n{printed}"
    )

    assert channel.call_count == 1
    items = handler.queues["transcriber"].items
    assert len(items) == 1
    assert items[0]["meta_info"]["eos"] is True
    assert items[0]["data"] is None
