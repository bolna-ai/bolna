"""TaskManager-level tests for the interruption-hint + cancel-in-flight wiring.

Verifies that barge-in no longer drops the Responses API chain at the
task_manager layer:
  * sync_history sets the hint instead of invalidating.
  * cleanup_before_user_append no longer invalidates the chain.
  * __cleanup_downstream_tasks cancels the in-flight LLM response.
"""

import asyncio
import inspect
import pytest
from unittest.mock import AsyncMock, MagicMock


def _make_input_tool(welcome_played=True):
    inp = MagicMock()
    inp.is_welcome_message_played = welcome_played
    inp.welcome_message_played = MagicMock(side_effect=lambda: inp.is_welcome_message_played)
    inp.io_provider = "plivo"
    inp.is_audio_being_played_to_user = MagicMock(return_value=False)
    inp.update_is_audio_being_played = MagicMock()
    inp.reset_response_heard_by_user = MagicMock()
    inp.get_call_sid = MagicMock(return_value="call-1")
    return inp


def _make_output_tool():
    out = MagicMock()
    out.handle_interruption = AsyncMock()
    out.flush_synthesizer_stream = AsyncMock()
    out.get_provider = MagicMock(return_value="plivo")
    return out


def _make_synth_tool():
    synth = MagicMock()
    synth.handle_interruption = AsyncMock()
    synth.flush_synthesizer_stream = AsyncMock()
    return synth


def _make_llm_agent():
    agent = MagicMock()
    agent.llm = MagicMock()
    agent.llm.set_interruption_hint = MagicMock()
    agent.llm.invalidate_response_chain = MagicMock()
    agent.llm.cancel_in_flight_response = MagicMock()
    return agent


def _make_history(length=2):
    h = MagicMock()
    h.__len__ = MagicMock(return_value=length)
    h.is_duplicate_user = MagicMock(return_value=False)
    h.append_user = MagicMock()
    h.pop_unheard_responses = MagicMock()
    h.pop_and_merge_user = MagicMock(side_effect=lambda m: m)
    h.get_copy = MagicMock(return_value=[{"role": "system", "content": "x"}])
    h.sync_interim = MagicMock()
    return h


def _make_task_manager():
    from bolna.agent_manager.task_manager import TaskManager

    tm = MagicMock()
    tm.tools = {
        "input": _make_input_tool(welcome_played=True),
        "output": _make_output_tool(),
        "synthesizer": _make_synth_tool(),
        "llm_agent": _make_llm_agent(),
    }
    mem = MagicMock()
    mem.fetch_cleared_mark_event_data = MagicMock(return_value={})
    tm.mark_event_meta_data = mem
    tm.conversation_history = _make_history(length=4)
    tm.discard_pre_welcome_utterance = False
    tm._speech_started_before_welcome = False
    tm.generate_precise_transcript = False
    tm.response_in_pipeline = False
    tm.run_id = "test-run"
    tm.llm_task = None
    tm.eager_llm_task = None
    tm.first_message_task = None
    tm.output_task = None
    tm.synthesizer_tasks = []
    tm.buffered_output_queue = asyncio.Queue()
    tm._turn_audio_flushed = MagicMock()
    tm._turn_audio_flushed.set = MagicMock()
    tm.started_transmitting_audio = True
    tm.last_transmitted_timestamp = 0.0
    tm.transcriber_provider = "deepgram"
    tm.voicemail_handler = MagicMock()
    tm.voicemail_handler.detected = False
    tm.voicemail_handler.cancel_task = MagicMock()
    tm.voicemail_handler.trigger_check = MagicMock()
    tm.interruption_manager = MagicMock()
    tm.interruption_manager.invalidate_pending_responses = MagicMock()
    tm.interruption_manager.has_pending_responses_excluding = MagicMock(return_value=False)
    tm.language_detector = MagicMock()
    tm.language_detector.collect_transcript = AsyncMock()
    tm._has_interruptible_mark_activity = MagicMock(return_value=False)
    tm._drop_all_staged_assistant_history = MagicMock()
    tm._retire_dropped_response = MagicMock()
    tm._trigger_voicemail_check = MagicMock()
    tm.execute_function_call_task = None
    tm.task_config = {
        "task_type": "conversation",
        "tools_config": {
            "transcriber": {"provider": "deepgram"},
            "llm_agent": {"agent_flow_type": "preprocessed"},
            "output": {"provider": "plivo", "format": "pcm"},
        },
    }
    tm.__process_output_loop = AsyncMock()

    tm.sync_history = AsyncMock()
    tm._set_interruption_hint = TaskManager._set_interruption_hint.__get__(tm, TaskManager)
    tm._cancel_in_flight_llm_response = TaskManager._cancel_in_flight_llm_response.__get__(tm, TaskManager)
    tm._invalidate_response_chain = TaskManager._invalidate_response_chain.__get__(tm, TaskManager)
    tm._TaskManager__cleanup_downstream_tasks = TaskManager._TaskManager__cleanup_downstream_tasks.__get__(
        tm, TaskManager
    )
    return tm


@pytest.fixture(autouse=True)
def _patch_create_task(monkeypatch):
    monkeypatch.setattr(asyncio, "create_task", lambda coro: MagicMock())
    yield


class TestHelperRouting:
    def test_set_interruption_hint_routes_to_llm(self):
        tm = _make_task_manager()
        tm._set_interruption_hint("the user heard this")
        tm.tools["llm_agent"].llm.set_interruption_hint.assert_called_once_with("the user heard this")

    def test_cancel_in_flight_routes_to_llm(self):
        tm = _make_task_manager()
        tm._cancel_in_flight_llm_response()
        tm.tools["llm_agent"].llm.cancel_in_flight_response.assert_called_once_with()

    def test_invalidate_still_routes_to_llm(self):
        tm = _make_task_manager()
        tm._invalidate_response_chain()
        tm.tools["llm_agent"].llm.invalidate_response_chain.assert_called_once_with()

    def test_set_interruption_hint_no_llm_agent_is_safe(self):
        tm = _make_task_manager()
        tm.tools.pop("llm_agent")
        tm._set_interruption_hint("text")  # must not raise

    def test_cancel_in_flight_no_llm_agent_is_safe(self):
        tm = _make_task_manager()
        tm.tools.pop("llm_agent")
        tm._cancel_in_flight_llm_response()  # must not raise


class TestCleanupDownstream:
    @pytest.mark.asyncio
    async def test_cleanup_cancels_in_flight_llm(self):
        tm = _make_task_manager()
        await tm._TaskManager__cleanup_downstream_tasks()
        tm.tools["llm_agent"].llm.cancel_in_flight_response.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_cleanup_does_not_invalidate_chain(self):
        tm = _make_task_manager()
        await tm._TaskManager__cleanup_downstream_tasks()
        tm.tools["llm_agent"].llm.invalidate_response_chain.assert_not_called()


class TestCallSites:
    """Source-level guards: catches accidental re-introduction of the
    barge-in invalidation we deliberately removed."""

    def test_sync_history_does_not_invalidate(self):
        from bolna.agent_manager.task_manager import TaskManager

        src = inspect.getsource(TaskManager.sync_history)
        assert "_invalidate_response_chain" not in src, (
            "sync_history must use _set_interruption_hint instead of dropping the chain"
        )
        assert "_set_interruption_hint" in src

    def test_handle_transcriber_output_does_not_invalidate(self):
        from bolna.agent_manager.task_manager import TaskManager

        src = inspect.getsource(TaskManager._handle_transcriber_output)
        assert "_invalidate_response_chain" not in src, (
            "_handle_transcriber_output must not drop the chain; sync_history already set the hint"
        )

    def test_cleanup_downstream_cancels_in_flight(self):
        from bolna.agent_manager.task_manager import TaskManager

        src = inspect.getsource(TaskManager._TaskManager__cleanup_downstream_tasks)
        assert "_cancel_in_flight_llm_response" in src
