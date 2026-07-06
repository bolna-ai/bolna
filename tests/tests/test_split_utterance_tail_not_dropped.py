"""Short speech_final transcripts must not be dropped while the agent is only
thinking (response_in_pipeline=True, no audio playing).

Reproduces call 032a233d: Deepgram force-finalized "hello" mid-utterance and
dispatched it to the LLM; the real speech_final tail "हिंदी में" (2 words) arrived
1s later and was discarded as a "false interruption" because the guard counted
response_in_pipeline as audio playing. The user's request for Hindi never
reached the LLM. The fix: only actual audio playback gates the drop —
_handle_transcriber_output already merges the tail and supersedes the in-flight
LLM turn (pop_and_merge_user + cancel/regenerate).
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from bolna.agent_manager.task_manager import TaskManager
from bolna.agent_manager.interruption_manager import InterruptionManager


_TAIL_FINAL = {
    "data": {"type": "transcript", "content": " हिंदी में"},
    "meta_info": {"io": "plivo", "sequence_id": 2, "request_id": "req-1"},
}


def _make_tm(*, audio_playing, response_in_pipeline, function_call_in_flight=False):
    tm = MagicMock()
    tm.hangup_triggered = False
    tm._end_call_in_progress = False
    tm.has_transfer = False
    tm.stream = True
    tm.history = []
    tm.response_in_pipeline = response_in_pipeline
    tm.function_call_in_flight = function_call_in_flight
    tm.output_task = MagicMock()
    tm.eager_llm_task = None
    tm.transcriber_output_queue = asyncio.Queue()
    tm.process_transcriber_request = AsyncMock(return_value=0)
    tm._set_call_details = MagicMock()
    tm._get_next_step = MagicMock(return_value="llm")
    tm.tools = {"input": MagicMock(), "transcriber": MagicMock()}
    tm.tools["input"].welcome_message_played = MagicMock(return_value=True)
    tm.tools["input"].is_audio_being_played_to_user = MagicMock(return_value=audio_playing)
    tm.interruption_manager = InterruptionManager(number_of_words_for_interruption=2)
    tm._maybe_update_tts_language = AsyncMock()
    tm._handle_transcriber_output = AsyncMock()
    tm._TaskManager__get_updated_meta_info = MagicMock(side_effect=lambda m: m)
    tm._TaskManager__cleanup_downstream_tasks = AsyncMock()
    tm._end_call_on_component_error = AsyncMock()
    tm.task_config = {"tools_config": {"transcriber": {"provider": "deepgram"}}}
    tm._should_ignore_transcriber_input = TaskManager._should_ignore_transcriber_input.__get__(tm, TaskManager)
    tm._listen_transcriber = TaskManager._listen_transcriber.__get__(tm, TaskManager)
    return tm


async def _drive(tm, message=_TAIL_FINAL):
    await tm.transcriber_output_queue.put(message)
    try:
        await asyncio.wait_for(tm._listen_transcriber(), timeout=0.3)
    except asyncio.TimeoutError:
        pass


@pytest.mark.asyncio
async def test_tail_processed_while_llm_generating_no_audio():
    # The 032a233d repro: agent thinking, nothing playing — tail must reach the LLM path.
    tm = _make_tm(audio_playing=False, response_in_pipeline=True)
    await _drive(tm)
    tm._handle_transcriber_output.assert_awaited_once()


@pytest.mark.asyncio
async def test_short_final_still_dropped_while_audio_playing():
    # Real barge-in protection unchanged: short finals during agent SPEECH stay dropped.
    tm = _make_tm(audio_playing=True, response_in_pipeline=False)
    await _drive(tm)
    tm._handle_transcriber_output.assert_not_awaited()


@pytest.mark.asyncio
async def test_tail_deferred_not_processed_during_tool_call():
    # function_call_in_flight ordering intact: no cancel/regenerate mid tool call.
    tm = _make_tm(audio_playing=False, response_in_pipeline=True, function_call_in_flight=True)
    await _drive(tm)
    tm._handle_transcriber_output.assert_not_awaited()
    tm._TaskManager__cleanup_downstream_tasks.assert_not_awaited()


@pytest.mark.asyncio
async def test_long_final_processed_while_audio_playing():
    # 3+ words exceeds number_of_words_for_interruption=2 — never treated as false interruption.
    tm = _make_tm(audio_playing=True, response_in_pipeline=False)
    long_final = {
        "data": {"type": "transcript", "content": "मुझे हिंदी में बात करनी है"},
        "meta_info": {"io": "plivo", "sequence_id": 2, "request_id": "req-1"},
    }
    await _drive(tm, long_final)
    tm._handle_transcriber_output.assert_awaited_once()
