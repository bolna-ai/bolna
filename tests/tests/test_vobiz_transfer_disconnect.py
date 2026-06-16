"""Regression tests for the VoBiz transfer disconnect bug.

Bug (exec 7260ebd8-...): VoBiz `disconnect_stream()` issues DELETE on the whole call,
not just the bot's audio stream (unlike Plivo's `delete_all_streams`). After a transfer
the call is bridged to the transfer target, so when the bot's leg tears down the DELETE
hangs up the now-bridged original caller. Fix: skip disconnect_stream entirely when a
transfer has occurred (`call_transferred` flag set by the task manager).
"""

from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from bolna.input_handlers.telephony_providers.vobiz import VobizInputHandler


def _make_handler():
    handler = VobizInputHandler(queues=None, websocket=AsyncMock())
    handler.call_sid = "0edd02a5-2d7f-442c-86a0-f65e05d68ad1"
    handler.stream_sid = "56711996-c912-4571-9a6a-9b7e9"
    return handler


@pytest.mark.asyncio
async def test_disconnect_stream_skips_call_delete_after_transfer():
    """When a transfer occurred, the bot must NOT delete the bridged VoBiz call."""
    handler = _make_handler()
    handler.call_transferred = True

    with patch.dict(
        "os.environ", {"VOBIZ_API_KEY": "key", "VOBIZ_API_SECRET": "secret"}
    ), patch("bolna.input_handlers.telephony_providers.vobiz.requests") as mock_requests:
        await handler.disconnect_stream()

    # No DELETE to the VoBiz API, and no stop event on the websocket — the call is
    # owned by the telephony layer now and must be left untouched.
    mock_requests.delete.assert_not_called()
    handler.websocket.send_text.assert_not_called()


@pytest.mark.asyncio
async def test_disconnect_stream_deletes_call_on_normal_end():
    """No transfer: normal teardown still deletes the VoBiz call (unchanged behavior)."""
    handler = _make_handler()
    assert handler.call_transferred is False  # default

    with patch.dict(
        "os.environ", {"VOBIZ_API_KEY": "key", "VOBIZ_API_SECRET": "secret"}
    ), patch("bolna.input_handlers.telephony_providers.vobiz.requests") as mock_requests:
        mock_requests.delete.return_value = MagicMock(status_code=204)
        await handler.disconnect_stream()

    mock_requests.delete.assert_called_once()
    called_url = mock_requests.delete.call_args[0][0]
    assert handler.call_sid in called_url
    # stop event for the media stream is still sent on a normal end
    handler.websocket.send_text.assert_awaited()


def test_call_transferred_defaults_false():
    """A fresh handler does not suppress disconnect (only an explicit transfer does)."""
    assert _make_handler().call_transferred is False
