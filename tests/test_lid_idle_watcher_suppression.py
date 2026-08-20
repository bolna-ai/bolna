"""Idle-flush mid-utterance suppression.

The watcher fired 1.2s after a detector segment even while main-ASR interims showed the
caller mid-sentence — slicing one utterance across two decides (an idle-flush judging the
opening ack, then the turn-boundary decide waiting out its lock). While callee_speaking is
true the flush defers; past LANGUAGE_SWITCH_SPEAKING_STALE_CAP_S of buffer age the flag is
treated as stale (the detector hears the same audio and produced nothing that long) and the
flush fires anyway, so a wedged final can never starve the safety net.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock


from bolna.agent_manager.task_manager import TaskManager
from bolna.transcriber.transcriber_pool import TranscriberPool


def _tm(*, buffer_age, speaking, buffered_lang="hi", active="mr", segments=None):
    tm = MagicMock()
    tm.conversation_ended = False
    tm.hangup_triggered = False
    tm._end_call_in_progress = False
    tm.has_transfer = False
    tm.language = active
    tm.handle_language_switch = AsyncMock()
    tm.interruption_manager = MagicMock()
    tm.interruption_manager.callee_speaking = speaking

    pool = MagicMock(spec=TranscriberPool)
    pool.lid_buffer_age.return_value = buffer_age
    pool.lid_buffer_language.return_value = buffered_lang
    pool.lid_buffer_event.return_value = None
    pool.lid_buffer_segments.return_value = segments or [{"lang": buffered_lang, "prob": 0.9, "audio_s": 1.2}]
    tm.tools = {"transcriber": pool}

    tm._should_ignore_transcriber_input = TaskManager._should_ignore_transcriber_input.__get__(tm, TaskManager)
    tm._TaskManager__lid_idle_watcher = TaskManager._TaskManager__lid_idle_watcher.__get__(tm, TaskManager)
    tm._TaskManager__buffered_language_evidence = TaskManager._TaskManager__buffered_language_evidence
    return tm


async def _run_watcher_for(tm, seconds):
    task = asyncio.create_task(tm._TaskManager__lid_idle_watcher())
    await asyncio.sleep(seconds)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


async def test_flush_deferred_while_caller_is_speaking():
    # Aged mismatched buffer, but interims say mid-utterance: the coming turn will drain it.
    tm = _tm(buffer_age=1.5, speaking=True)
    await _run_watcher_for(tm, 0.8)
    tm.handle_language_switch.assert_not_awaited()


async def test_flush_fires_when_caller_not_speaking():
    tm = _tm(buffer_age=1.5, speaking=False)
    await _run_watcher_for(tm, 0.8)
    tm.handle_language_switch.assert_awaited()


async def test_stale_speaking_flag_cannot_starve_the_flush():
    # Flag stuck true (lost final / wedged socket) but the detector has been silent past the
    # cap — the safety net must still fire.
    tm = _tm(buffer_age=2.6, speaking=True)
    await _run_watcher_for(tm, 0.8)
    tm.handle_language_switch.assert_awaited()


async def test_suppression_resumes_firing_when_speech_ends():
    tm = _tm(buffer_age=1.5, speaking=True)
    task = asyncio.create_task(tm._TaskManager__lid_idle_watcher())
    await asyncio.sleep(0.5)
    assert not tm.handle_language_switch.await_count  # deferred while speaking
    tm.interruption_manager.callee_speaking = False  # final transcript landed... or VAD closed
    await asyncio.sleep(0.5)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    tm.handle_language_switch.assert_awaited()


async def test_missing_interruption_manager_behaves_like_today():
    # A provider path with no speaking state must leave the watcher exactly as before.
    tm = _tm(buffer_age=1.5, speaking=False)
    del tm.interruption_manager.callee_speaking
    tm.interruption_manager = MagicMock(spec=[])  # no callee_speaking attribute at all
    await _run_watcher_for(tm, 0.8)
    tm.handle_language_switch.assert_awaited()
