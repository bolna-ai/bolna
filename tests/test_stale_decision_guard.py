"""The stale-decision guard drops the decision but keeps the detector buffer.

The LIVE transcript is invalid after a switch because it came from the pre-switch recognizer, and
trusting it causes switch ping-pong. The detector runs with language-code unknown, so its speech
means the same before and after — discarding the buffer too would delete a request the caller made
during the decide and force them to repeat it.

The guard keys on the language the decide was spawned on, so it fires only once that language is
no longer active. An unrecorded spawn language must leave the switch path alone.
"""

from bolna.agent_manager.task_manager import TaskManager

RUN = TaskManager._TaskManager__run_language_switch
LIVE = "garbled hi"
META = {"sequence_id": 1}


async def test_stale_decision_is_dropped_but_the_buffer_survives(language_switch_tm):
    """Spawned on "mr" while "hi" is active: the decision goes, the caller's speech stays."""
    tm = language_switch_tm()

    result = await RUN(tm, LIVE, META, "mr")

    assert result is None
    tm.switch_language.assert_not_awaited()
    tm.tools["transcriber"].take_lid_transcript.assert_not_called()


async def test_a_current_spawn_language_runs_the_switch(language_switch_tm):
    """The guard must not swallow the ordinary case it shares a code path with."""
    tm = language_switch_tm()

    await RUN(tm, LIVE, META, "hi")

    tm.switch_language.assert_awaited_once()
    tm.tools["transcriber"].take_lid_transcript.assert_called()


async def test_unrecorded_spawn_language_leaves_the_switch_alone(language_switch_tm):
    """Idle-flush passes no spawn language, and a missing one is not evidence of staleness."""
    tm = language_switch_tm()

    await RUN(tm, LIVE, META, None)

    tm.switch_language.assert_awaited_once()
    tm.tools["transcriber"].take_lid_transcript.assert_called()
