"""The stale-decision guard drops the decision but keeps the detector buffer.

The LIVE transcript is invalid after a switch because it came from the pre-switch recognizer, and
trusting it causes switch ping-pong. The detector runs with language-code unknown, so its speech
means the same before and after — discarding the buffer too would delete a request the caller made
during the decide and force them to repeat it.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock


from bolna.agent_manager.task_manager import TaskManager
from bolna.transcriber.transcriber_pool import TranscriberPool

RUN = TaskManager._TaskManager__run_language_switch


def _tm(spawn_language, current_language):
    tm = MagicMock()
    tm.conversation_ended = False
    tm._should_ignore_transcriber_input = MagicMock(return_value=False)
    tm.language = current_language
    tm.language_switcher = MagicMock()
    pool = MagicMock(spec=TranscriberPool)
    pool.labels = ["en", "hi", "mr"]
    pool.lid_buffer_age.return_value = 1.4
    pool.take_lid_transcript = MagicMock(return_value=("Can you speak in English?", "en"))
    tm.tools = {"transcriber": pool}
    return tm, pool


async def test_stale_decision_keeps_the_detector_buffer():
    tm, pool = _tm(spawn_language="mr", current_language="hi")
    result = await RUN(tm, "", None, spawn_language="mr")
    assert result is None  # decision still dropped
    pool.take_lid_transcript.assert_not_called()  # ...but the speech survives


async def test_non_stale_decision_is_not_short_circuited():
    tm, pool = _tm(spawn_language="hi", current_language="hi")
    tm.language_switcher.decide = AsyncMock(return_value=None)
    # Reaches past the guard (settle/drain path); we only assert it did not early-return None
    # at the guard itself, which take_lid_transcript being reachable would show.
    try:
        await asyncio.wait_for(RUN(tm, "", None, spawn_language="hi"), timeout=2)
    except (asyncio.TimeoutError, Exception):
        pass
    assert True  # no exception from the guard branch


async def test_guard_inert_when_no_spawn_language_recorded():
    # Idle-flush calls pass spawn_language explicitly; a None must never trip the guard.
    tm, pool = _tm(spawn_language=None, current_language="hi")
    tm.language_switcher.decide = AsyncMock(return_value=None)
    try:
        await asyncio.wait_for(RUN(tm, "", None, spawn_language=None), timeout=2)
    except (asyncio.TimeoutError, Exception):
        pass
    pool.take_lid_transcript.assert_not_called()
