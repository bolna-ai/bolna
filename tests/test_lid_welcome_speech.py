"""Speech over the welcome must never reach the language-switch judge.

The welcome is non-interruptible: _handle_transcriber_output drops every turn spoken over it. The
detector tap still buffered the same speech, and with no main turn to drain it the idle-flush
judged it alone, with no LIVE transcript. A caller's "Hello" over a Tamil welcome read as English
and switched the call.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

from bolna.agent_manager.task_manager import TaskManager
from bolna.lid.base import LIDBackend
from bolna.transcriber.transcriber_pool import TranscriberPool


def _tm(lid, *, welcome_played, welcome_ended_ms=None):
    tm = MagicMock()
    tm.conversation_ended = False
    tm.hangup_triggered = False
    tm._end_call_in_progress = False
    tm.has_transfer = False
    tm.language = "ta"
    tm.handle_language_switch = AsyncMock()
    tm.interruption_manager = MagicMock()
    tm.interruption_manager.callee_speaking = False

    # Pool over a real detector buffer, so a drain empties it the way it does in a call.
    pool = MagicMock(spec=TranscriberPool)
    pool.lid_buffer_age.side_effect = lid.buffer_age_seconds
    pool.lid_buffer_language.side_effect = lid.buffer_language
    pool.lid_buffer_event.side_effect = lid.buffer_event
    pool.lid_buffer_segments.side_effect = lid.buffer_segments
    pool.take_lid_transcript.side_effect = lid.take_turn_transcript
    tm.tools = {"transcriber": pool, "input": MagicMock()}
    tm.tools["input"].welcome_message_played.return_value = welcome_played
    tm.tools["input"].welcome_message_played_ts = welcome_ended_ms

    tm._should_ignore_transcriber_input = TaskManager._should_ignore_transcriber_input.__get__(tm, TaskManager)
    tm._TaskManager__lid_idle_watcher = TaskManager._TaskManager__lid_idle_watcher.__get__(tm, TaskManager)
    tm._TaskManager__buffered_language_evidence = TaskManager._TaskManager__buffered_language_evidence
    tm._TaskManager__spoken_over_welcome = TaskManager._TaskManager__spoken_over_welcome.__get__(tm, TaskManager)
    return tm


async def _stop(task):
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


async def test_hello_over_the_welcome_is_dropped_not_judged(monkeypatch):
    monkeypatch.setenv("LANGUAGE_SWITCH_MISMATCH_IDLE_FLUSH_S", "0.2")
    lid = LIDBackend(on_language=None, config={})
    tm = _tm(lid, welcome_played=False)
    task = asyncio.create_task(tm._TaskManager__lid_idle_watcher())
    await asyncio.sleep(0.05)
    lid._accumulate("Hello", "en", 0.736, 0.901)  # a short greeting the detector tags as English
    await asyncio.sleep(0.1)
    # Not dropped on arrival: if the main pipeline keeps this turn, its own decide must still find it.
    assert lid.buffer_age_seconds() is not None
    await asyncio.sleep(0.3)
    # Nor once the welcome ends: the dropped speech must not be judged late either.
    tm.tools["input"].welcome_message_played.return_value = True
    tm.tools["input"].welcome_message_played_ts = time.time() * 1000
    await asyncio.sleep(0.4)
    await _stop(task)
    tm.handle_language_switch.assert_not_awaited()
    assert lid.buffer_age_seconds() is None


async def test_hello_from_over_the_welcome_delivered_after_it_ends_is_dropped(monkeypatch):
    # The detector can deliver after the welcome ends; the speech still began during it.
    monkeypatch.setenv("LANGUAGE_SWITCH_MISMATCH_IDLE_FLUSH_S", "0.2")
    lid = LIDBackend(on_language=None, config={})
    tm = _tm(lid, welcome_played=True, welcome_ended_ms=time.time() * 1000)
    task = asyncio.create_task(tm._TaskManager__lid_idle_watcher())
    await asyncio.sleep(0.2)
    lid._accumulate("Hello", "en", 0.736, 0.901)  # spoken from ~0.5s before the welcome ended
    await asyncio.sleep(0.4)
    await _stop(task)
    tm.handle_language_switch.assert_not_awaited()
    assert lid.buffer_age_seconds() is None


async def test_speech_after_the_welcome_is_still_judged(monkeypatch):
    monkeypatch.setenv("LANGUAGE_SWITCH_MISMATCH_IDLE_FLUSH_S", "0.2")
    lid = LIDBackend(on_language=None, config={})
    tm = _tm(lid, welcome_played=True, welcome_ended_ms=(time.time() - 5) * 1000)
    task = asyncio.create_task(tm._TaskManager__lid_idle_watcher())
    await asyncio.sleep(0.05)
    lid._accumulate("Can you speak in English please", "en", 1.8, 0.95)
    await asyncio.sleep(0.4)
    await _stop(task)
    tm.handle_language_switch.assert_awaited()


async def test_calls_without_a_welcome_timestamp_are_still_judged(monkeypatch):
    # No welcome message (or one cut short) never gets an ACK timestamp; switching must stay live.
    monkeypatch.setenv("LANGUAGE_SWITCH_MISMATCH_IDLE_FLUSH_S", "0.2")
    lid = LIDBackend(on_language=None, config={})
    tm = _tm(lid, welcome_played=True, welcome_ended_ms=None)
    task = asyncio.create_task(tm._TaskManager__lid_idle_watcher())
    await asyncio.sleep(0.05)
    lid._accumulate("Can you speak in English please", "en", 1.8, 0.95)
    await asyncio.sleep(0.4)
    await _stop(task)
    tm.handle_language_switch.assert_awaited()


async def test_pre_welcome_turn_dropped_after_the_welcome_takes_its_detector_copy():
    # discard_pre_welcome_utterance on: the main pipeline drops this turn after the welcome, so drain here.
    lid = LIDBackend(on_language=None, config={})
    lid._accumulate("Hello", "en", 0.736, 0.901)
    pool = MagicMock(spec=TranscriberPool)
    pool.take_lid_transcript.side_effect = lid.take_turn_transcript
    tm = TaskManager.__new__(TaskManager)
    tm.tools = {"input": MagicMock(**{"welcome_message_played.return_value": True}), "transcriber": pool}
    tm._speech_started_before_welcome = True
    tm._retire_dropped_response = MagicMock()
    await TaskManager._handle_transcriber_output(tm, "llm", "ஹலோ.", {"sequence_id": 1, "turn_id": 1})
    tm._retire_dropped_response.assert_called_once_with(
        {"sequence_id": 1, "turn_id": 1}, "speech_started_before_welcome_finished"
    )
    assert lid.buffer_age_seconds() is None
