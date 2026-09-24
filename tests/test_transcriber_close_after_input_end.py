"""A transcriber socket closing after the caller's audio has ended is teardown, not a connection failure."""

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from bolna.agent_manager.task_manager import TaskManager
from bolna.enums import HangupReason
from bolna.input_handlers.default import DefaultInputHandler

NO_CLOSE_FRAME = "sent 1000 (OK); no close frame received"


def _tm(input_stream_ended):
    tm = MagicMock()
    tm.task_config = {"tools_config": {"transcriber": {"provider": "elevenlabs"}}}
    tm.tools = {"input": SimpleNamespace(input_stream_ended=input_stream_ended)}
    tm.transcriber_error_events = []
    tm.conversation_start_init_ts = time.time() * 1000
    tm._end_call_on_component_error = AsyncMock()
    tm._component_model = MagicMock(return_value="scribe_v2_realtime")
    return tm


def test_close_error_mid_call_ends_the_call_as_connection_error():
    tm = _tm(input_stream_ended=False)

    asyncio.run(TaskManager._log_transcriber_connection_error(tm, NO_CLOSE_FRAME))

    assert tm.transcriber_error_events[0]["event"] == "error"
    tm._end_call_on_component_error.assert_awaited_once()
    assert tm._end_call_on_component_error.await_args.args[1] == HangupReason.TRANSCRIBER_CONNECTION_ERROR


@pytest.mark.parametrize("connection_error", [NO_CLOSE_FRAME, None])
def test_close_after_input_ended_is_recorded_as_drop(connection_error):
    tm = _tm(input_stream_ended=True)

    asyncio.run(TaskManager._log_transcriber_connection_error(tm, connection_error))

    assert tm.transcriber_error_events[0]["event"] == "drop"
    assert tm.transcriber_error_events[0]["error"] == connection_error
    tm._end_call_on_component_error.assert_not_awaited()


def test_ending_the_input_stream_flags_it_and_sends_eos():
    handler = DefaultInputHandler.__new__(DefaultInputHandler)
    handler.queues = {"transcriber": asyncio.Queue()}
    handler.input_stream_ended = False

    handler._end_input_stream("plivo", sequence=3)

    assert handler.input_stream_ended is True
    packet = handler.queues["transcriber"].get_nowait()
    assert packet["meta_info"]["eos"] is True
    assert packet["meta_info"]["io"] == "plivo"
    assert packet["meta_info"]["sequence"] == 3
