"""Guards the end_call hangup actuation against barge-in resurrection.

Reproduces the failure where the end_call tool fired and generated a goodbye,
but a user barge-in during/after the goodbye cancelled the conversation turn
task before the disconnect ran. hangup_triggered was never set, so the
transcriber kept feeding new turns and the agent looped goodbyes until the
user dropped (actual_hangup_reason stayed null in the experiment outcome).

The fix: _end_call_in_progress is set the instant the end_call tool fires
(before the goodbye is generated), and _listen_transcriber drops all user
speech while a hangup or end_call actuation is underway, so barge-in can no
longer cancel the in-flight hangup.
"""

import inspect
from types import SimpleNamespace

from bolna.agent_manager.task_manager import TaskManager


def _ignore(hangup_triggered, end_call_in_progress):
    fake = SimpleNamespace(hangup_triggered=hangup_triggered, _end_call_in_progress=end_call_in_progress)
    return TaskManager._should_ignore_transcriber_input(fake)


def test_ignores_input_during_end_call_actuation():
    # end_call fired but hangup_triggered not yet set (goodbye still generating).
    assert _ignore(hangup_triggered=False, end_call_in_progress=True) is True


def test_ignores_input_after_hangup_locked():
    assert _ignore(hangup_triggered=True, end_call_in_progress=False) is True


def test_processes_input_during_normal_conversation():
    assert _ignore(hangup_triggered=False, end_call_in_progress=False) is False


class TestSourceGuards:
    """Catch accidental removal of the wiring that closes the barge-in race."""

    def test_end_call_branch_sets_in_progress_flag(self):
        src = inspect.getsource(TaskManager._TaskManager__execute_function_call)
        assert "_end_call_in_progress = True" in src, (
            "end_call branch must set _end_call_in_progress before generating the goodbye"
        )

    def test_listen_transcriber_uses_the_guard(self):
        src = inspect.getsource(TaskManager._listen_transcriber)
        assert "_should_ignore_transcriber_input" in src, (
            "_listen_transcriber must drop user speech while a hangup/end_call actuation is underway"
        )
