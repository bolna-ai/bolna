"""The caller hung up while the reply was in flight: its pre-mark was recorded but the audio
send failed on the closed socket, so no audio mark exists. End-of-call sync kept the full
unheard reply in the transcript because the lone pre-mark was not treated as evidence."""

import time

from bolna.agent_manager.task_manager import TaskManager
from bolna.helpers.conversation_history import ConversationHistory
from bolna.helpers.mark_event_meta_data import MarkEventMetaData

HEARD_REPLY = "hello, I am calling from the support team"
UNSENT_REPLY = "I will forward your query to the support team and they will reply to you by email"


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


def _pre_mark(turn_id, response_uid, category=""):
    return {
        "type": "pre_mark_message",
        "sequence_id": turn_id,
        "turn_id": turn_id,
        "response_uid": response_uid,
        "message_category": category,
    }


def _audio_mark(turn_id, response_uid, text):
    return {
        "type": "",
        "turn_id": turn_id,
        "response_uid": response_uid,
        "sequence_id": turn_id,
        "duration": 2.0,
        "text_synthesized": text,
        "sent_ts": time.time() - 10,
    }


def _make_tm():
    tm = TaskManager.__new__(TaskManager)
    history = ConversationHistory()
    history.append_assistant(HEARD_REPLY, turn_id=4, response_uid="r4")
    history.append_user("hello? yes")
    history.append_assistant(UNSENT_REPLY, turn_id=15, response_uid="r15")
    tm.conversation_history = history
    tm.tools = {"input": _InputStub()}
    tm._turn_msg_map = {}
    tm.mark_event_meta_data = MarkEventMetaData()
    return tm, history


def _assistant_contents(history):
    return [m["content"] for m in history.messages if m["role"] == "assistant"]


async def _end_of_call_sync(tm):
    marks = tm.mark_event_meta_data
    await tm.sync_history(marks.mark_event_meta_data.items(), time.time(), extend_with_playback_estimate=True)


async def test_unsent_reply_removed_at_end_of_call():
    tm, history = _make_tm()
    tm.mark_event_meta_data.update_data("p15", _pre_mark(15, "r15"))

    await _end_of_call_sync(tm)

    assert UNSENT_REPLY not in _assistant_contents(history)
    assert HEARD_REPLY in _assistant_contents(history)


async def test_played_reply_with_lingering_pre_mark_kept():
    tm, history = _make_tm()
    marks = tm.mark_event_meta_data
    marks.update_data("a15", _audio_mark(15, "r15", UNSENT_REPLY))
    marks.fetch_data("a15")
    marks.update_data("p15", _pre_mark(15, "r15"))

    await _end_of_call_sync(tm)

    assert UNSENT_REPLY in _assistant_contents(history)


async def test_backchannel_pre_mark_does_not_trim():
    tm, history = _make_tm()
    tm.mark_event_meta_data.update_data("p15", _pre_mark(15, "r15", category="backchanneling"))

    await _end_of_call_sync(tm)

    assert UNSENT_REPLY in _assistant_contents(history)


async def test_pre_mark_without_response_uid_does_not_trim():
    tm, history = _make_tm()
    tm.mark_event_meta_data.update_data("p15", _pre_mark(15, None))

    await _end_of_call_sync(tm)

    assert UNSENT_REPLY in _assistant_contents(history)
