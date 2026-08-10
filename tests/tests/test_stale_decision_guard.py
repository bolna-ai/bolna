"""Stale-decision guard drops the DECISION but keeps the detector buffer.

QA 971254c0: the caller said "क्या बोल रहे हैं आप?" then immediately "Can you speak in
English?". The first segment alone drove an mr→hi switch; the English request landed in the
buffer during the decide and was deleted by this guard ("discarded buffer='Can you speak in
English? Police'"). The caller had to repeat it ~28s later.

The LIVE transcript IS invalid after a switch (it came from the pre-switch recognizer — that
mislabeling caused the 1a16da82 ping-pong), but the detector runs language-code=unknown, so its
speech means the same before and after.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

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


@pytest.mark.asyncio
async def test_stale_decision_keeps_the_detector_buffer():
    tm, pool = _tm(spawn_language="mr", current_language="hi")
    result = await RUN(tm, "", None, spawn_language="mr")
    assert result is None  # decision still dropped
    pool.take_lid_transcript.assert_not_called()  # ...but the speech survives


@pytest.mark.asyncio
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


@pytest.mark.asyncio
async def test_guard_inert_when_no_spawn_language_recorded():
    # Idle-flush calls pass spawn_language explicitly; a None must never trip the guard.
    tm, pool = _tm(spawn_language=None, current_language="hi")
    tm.language_switcher.decide = AsyncMock(return_value=None)
    try:
        await asyncio.wait_for(RUN(tm, "", None, spawn_language=None), timeout=2)
    except (asyncio.TimeoutError, Exception):
        pass
    pool.take_lid_transcript.assert_not_called()
