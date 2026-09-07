"""__cleanup_downstream_tasks must return even when the output socket is half-dead.

It awaits output.handle_interruption() ahead of task cancellation, history sync and flag
resets, so an unbounded await there freezes _listen_transcriber and everything gated on
cleanup returning. The send timeout is what bounds it.
"""

import asyncio

import pytest
from unittest.mock import AsyncMock, MagicMock

from bolna.helpers.mark_event_meta_data import MarkEventMetaData
from bolna.output_handlers.telephony_providers.plivo import PlivoOutputHandler
import bolna.output_handlers.telephony as telephony_module


class _HangingWebSocket:
    async def send_text(self, payload):
        await asyncio.Event().wait()  # never set: hangs forever, never raises


def _make_input_tool():
    inp = MagicMock()
    inp.welcome_message_played = MagicMock(return_value=True)
    inp.is_welcome_message_played = True
    inp.reset_response_heard_by_user = MagicMock()
    return inp


def _make_task_manager(output_tool):
    from bolna.agent_manager.task_manager import TaskManager

    tm = MagicMock()
    tm.tools = {
        "input": _make_input_tool(),
        "output": output_tool,
        "synthesizer": AsyncMock(),
    }
    tm.mark_event_meta_data = MarkEventMetaData()
    tm.sync_history = AsyncMock()
    tm.interruption_manager = MagicMock()
    tm.interruption_manager.invalidate_pending_responses = MagicMock()
    tm._drop_all_staged_assistant_history = MagicMock()
    tm._cancel_in_flight_llm_response = MagicMock()
    tm.response_in_pipeline = True
    tm.output_task = MagicMock()
    tm.llm_task = MagicMock()
    tm.eager_llm_task = None
    tm.first_message_task = None
    tm.voicemail_handler = MagicMock()
    tm.voicemail_handler.cancel_task = MagicMock()
    tm.synthesizer_tasks = [MagicMock(), MagicMock()]
    tm.buffered_output_queue = asyncio.Queue()
    tm._turn_audio_flushed = MagicMock()
    tm._turn_audio_flushed.set = MagicMock()
    tm.started_transmitting_audio = True
    tm.last_transmitted_timestamp = 0.0

    tm._TaskManager__cleanup_downstream_tasks = TaskManager._TaskManager__cleanup_downstream_tasks.__get__(
        tm, TaskManager
    )
    return tm


@pytest.fixture(autouse=True)
def _fast_send_timeout(monkeypatch):
    if hasattr(telephony_module, "OUTPUT_SEND_TIMEOUT_S"):
        monkeypatch.setattr(telephony_module, "OUTPUT_SEND_TIMEOUT_S", 0.05)


@pytest.fixture(autouse=True)
def _patch_create_task(monkeypatch):
    monkeypatch.setattr(asyncio, "create_task", lambda coro: MagicMock())
    yield


async def test_cleanup_finishes_and_cancels_tasks_despite_dead_output_socket():
    output_tool = PlivoOutputHandler(websocket=_HangingWebSocket(), mark_event_meta_data=MarkEventMetaData())
    output_tool.stream_sid = "test-stream"
    tm = _make_task_manager(output_tool)
    llm_task, output_task = tm.llm_task, tm.output_task
    synth_tasks = list(tm.synthesizer_tasks)

    await asyncio.wait_for(tm._TaskManager__cleanup_downstream_tasks(), timeout=1.0)

    assert output_tool.is_closed() is True  # the dead socket was actually detected
    output_task.cancel.assert_called_once()
    llm_task.cancel.assert_called_once()
    for t in synth_tasks:
        t.cancel.assert_called_once()
    tm._turn_audio_flushed.set.assert_called_once()  # watchdog gate cleared, not left stuck
    assert tm.response_in_pipeline is False
