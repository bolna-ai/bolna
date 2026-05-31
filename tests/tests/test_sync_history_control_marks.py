"""Regression tests: a leftover control mark (pre-mark/backchanneling) must not
become the sync_history trim target. A fully-played turn whose pre-mark lingered
unacked was being removed from the transcript as 'unheard' after a language
switch, causing the agent to repeat the vanished reply."""

import pytest

from bolna.agent_manager.task_manager import TaskManager
from bolna.helpers.conversation_history import ConversationHistory


def _mark(mark_type="", turn_id=None, response_uid=None, counter=0, duration=0.0):
    return {
        "type": mark_type,
        "turn_id": turn_id,
        "response_uid": response_uid,
        "counter": counter,
        "duration": duration,
    }


class TestLatestFromMarksSkipsControlMarks:
    def test_pre_mark_only_yields_no_target(self):
        marks = [("m0", _mark("pre_mark_message", turn_id=5, response_uid="r5", counter=0))]
        assert TaskManager._get_latest_turn_id_from_marks(marks) is None
        assert TaskManager._get_latest_response_uid_from_marks(marks) is None

    def test_backchanneling_only_yields_no_target(self):
        marks = [("m0", _mark("backchanneling", turn_id=5, response_uid="r5", counter=0))]
        assert TaskManager._get_latest_turn_id_from_marks(marks) is None
        assert TaskManager._get_latest_response_uid_from_marks(marks) is None

    def test_audio_mark_is_a_valid_target(self):
        marks = [
            ("m0", _mark("pre_mark_message", turn_id=5, response_uid="r5", counter=0)),
            ("m1", _mark("", turn_id=5, response_uid="r5", counter=1)),
        ]
        assert TaskManager._get_latest_turn_id_from_marks(marks) == 5
        assert TaskManager._get_latest_response_uid_from_marks(marks) == "r5"

    def test_audio_mark_wins_over_later_pre_mark(self):
        marks = [
            ("m0", _mark("", turn_id=4, response_uid="r4", counter=10)),
            ("m1", _mark("pre_mark_message", turn_id=5, response_uid="r5", counter=14)),
        ]
        assert TaskManager._get_latest_turn_id_from_marks(marks) == 4
        assert TaskManager._get_latest_response_uid_from_marks(marks) == "r4"


class _InputStub:
    last_heard_turn_id = None
    last_heard_response_uid = None
    response_heard_by_user = ""

    def get_response_heard_for_response(self, _uid):
        return ""

    def get_response_heard_for_turn(self, _turn_id):
        return ""

    def get_current_mark_started_time(self):
        return 0.0


class _MarkMetaStub:
    def get_heard_text_for_response(self, _uid):
        return ""

    def get_heard_text_for_turn(self, _turn_id):
        return ""


def _make_task_manager(history):
    tm = TaskManager.__new__(TaskManager)
    tm.conversation_history = history
    tm.tools = {"input": _InputStub()}
    tm.mark_event_meta_data = _MarkMetaStub()
    tm._turn_msg_map = {}
    return tm


@pytest.mark.asyncio
async def test_lingering_pre_mark_keeps_played_turn():
    history = ConversationHistory()
    history.append_user("switch to hindi please")
    history.append_assistant("agent reply in hindi", turn_id=5, response_uid="r5")
    tm = _make_task_manager(history)

    leftover = [("m0", _mark("pre_mark_message", turn_id=5, response_uid="r5", counter=0))]
    await tm.sync_history(leftover, interruption_processed_at=1000.0)

    assert any(m.get("content") == "agent reply in hindi" for m in history.messages)


@pytest.mark.asyncio
async def test_no_marks_keeps_played_turn():
    history = ConversationHistory()
    history.append_assistant("agent reply", turn_id=5, response_uid="r5")
    tm = _make_task_manager(history)

    await tm.sync_history([], interruption_processed_at=1000.0)

    assert any(m.get("content") == "agent reply" for m in history.messages)
