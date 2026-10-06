"""An eager end of turn starts a speculative reply only for a transcriber that declares the capability."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from bolna.agent_manager.interruption_manager import InterruptionManager
from bolna.agent_manager.task_manager import TaskManager

_EAGER = {
    "data": {"type": "eager_end_of_turn", "content": "haan ji", "confidence": None},
    "meta_info": {"io": "plivo", "sequence_id": 2, "request_id": "req-1"},
}


def _make_tm(transcriber):
    tm = MagicMock()
    tm.hangup_triggered = False
    tm._end_call_in_progress = False
    tm.has_transfer = False
    tm.stream = True
    tm.history = []
    tm.response_in_pipeline = False
    tm.function_call_in_flight = False
    tm.output_task = MagicMock()
    tm.eager_llm_task = None
    tm.transcriber_output_queue = asyncio.Queue()
    tm.process_transcriber_request = AsyncMock(return_value=0)
    tm._set_call_details = MagicMock()
    tm._get_next_step = MagicMock(return_value="llm")
    tm.tools = {"input": MagicMock(), "transcriber": transcriber}
    tm.tools["input"].welcome_message_played = MagicMock(return_value=True)
    tm.tools["input"].is_audio_being_played_to_user = MagicMock(return_value=False)
    tm.interruption_manager = InterruptionManager(number_of_words_for_interruption=2)
    tm.regen_settle_armed = MagicMock(return_value=False)
    tm._run_llm_task = AsyncMock()
    tm._TaskManager__get_updated_meta_info = MagicMock(side_effect=lambda m: dict(m))
    tm.task_config = {"tools_config": {"transcriber": {"provider": "bolna"}}}
    tm._should_ignore_transcriber_input = TaskManager._should_ignore_transcriber_input.__get__(tm, TaskManager)
    tm._listen_transcriber = TaskManager._listen_transcriber.__get__(tm, TaskManager)
    return tm


async def _drive(tm):
    await tm.transcriber_output_queue.put(_EAGER)
    try:
        await asyncio.wait_for(tm._listen_transcriber(), timeout=0.3)
    except asyncio.TimeoutError:
        pass


async def test_a_transcriber_with_eager_turns_starts_a_speculative_reply():
    tm = _make_tm(SimpleNamespace(eager_end_of_turn=True, eager_eot_threshold=None, current_turn_id=1))
    await _drive(tm)
    assert tm.eager_llm_task is not None
    assert tm.history[-1]["content"] == "haan ji"


async def test_a_transcriber_without_eager_turns_never_speculates():
    tm = _make_tm(SimpleNamespace(eager_end_of_turn=False, eager_eot_threshold=None, current_turn_id=1))
    await _drive(tm)
    assert tm.eager_llm_task is None
    tm._run_llm_task.assert_not_called()


async def test_flux_below_its_confidence_threshold_never_speculates():
    tm = _make_tm(SimpleNamespace(eager_end_of_turn=True, eager_eot_threshold=0.6, current_turn_id=1))
    low = {**_EAGER, "data": {**_EAGER["data"], "confidence": 0.4}}
    await tm.transcriber_output_queue.put(low)
    try:
        await asyncio.wait_for(tm._listen_transcriber(), timeout=0.3)
    except asyncio.TimeoutError:
        pass
    assert tm.eager_llm_task is None
