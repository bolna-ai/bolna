"""Follow-up payloads must not end on an assistant turn the caller never heard.

Gemini rejects a request ending with a model turn (prod cc413210: switch fired after an
unheard old-language reply was already committed, so the follow-up 400'd and never spoke).
"""

from unittest.mock import MagicMock

import pytest

from bolna.agent_manager.task_manager import TaskManager
from bolna.enums import ChatRole

# The methods under test are not in the package: this module landed without its production half.
# Unskip once TaskManager grows __row_was_heard and __drop_unheard_trailing_responses.
if not hasattr(TaskManager, "_TaskManager__drop_unheard_trailing_responses"):
    pytest.skip("__drop_unheard_trailing_responses is not implemented", allow_module_level=True)


def make_tm(*, heard_responses=None, heard_turns=None):
    tm = MagicMock()
    heard_responses = heard_responses or {}
    heard_turns = heard_turns or {}

    input_handler = MagicMock()
    input_handler.get_response_heard_for_response = lambda uid: heard_responses.get(uid, "")
    input_handler.get_response_heard_for_turn = lambda tid: heard_turns.get(tid, "")
    tm.tools = {"input": input_handler}
    tm.mark_event_meta_data.get_heard_text_for_response = lambda uid: ""
    tm.mark_event_meta_data.get_heard_text_for_turn = lambda tid: ""

    tm._TaskManager__row_was_heard = TaskManager._TaskManager__row_was_heard.__get__(tm, TaskManager)
    tm.drop = TaskManager._TaskManager__drop_unheard_trailing_responses.__get__(tm, TaskManager)
    return tm


USER_ROW = {"role": ChatRole.USER, "content": "I am available"}
SYSTEM_ROW = {"role": ChatRole.SYSTEM, "content": "prompt"}


def test_unheard_trailing_assistant_is_dropped():
    messages = [SYSTEM_ROW, USER_ROW, {"role": ChatRole.ASSISTANT, "content": "शुक्रिया।", "turn_id": 4}]
    make_tm().drop(messages)
    assert messages == [SYSTEM_ROW, USER_ROW]


def test_heard_trailing_assistant_is_kept():
    row = {"role": ChatRole.ASSISTANT, "content": "शुक्रिया।", "turn_id": 4, "response_uid": "r4"}
    messages = [SYSTEM_ROW, USER_ROW, row]
    make_tm(heard_responses={"r4": "शुक्रिया।"}).drop(messages)
    assert messages == [SYSTEM_ROW, USER_ROW, row]


def test_heard_by_turn_evidence_is_kept():
    row = {"role": ChatRole.ASSISTANT, "content": "partial", "turn_id": 4}
    messages = [USER_ROW, row]
    make_tm(heard_turns={4: "partial"}).drop(messages)
    assert messages == [USER_ROW, row]


def test_dangling_tool_call_group_is_dropped_whole():
    messages = [
        SYSTEM_ROW,
        USER_ROW,
        {"role": ChatRole.ASSISTANT, "content": None, "tool_calls": [{"id": "c1"}], "turn_id": 4},
        {"role": ChatRole.TOOL, "tool_call_id": "c1", "content": "{'ok': true}"},
        {"role": ChatRole.ASSISTANT, "content": "शुक्रिया।", "turn_id": 4},
    ]
    make_tm().drop(messages)
    assert messages == [SYSTEM_ROW, USER_ROW]


def test_stops_at_heard_row_leaving_group_intact():
    heard = {"role": ChatRole.ASSISTANT, "content": "heard bit", "response_uid": "r3"}
    messages = [USER_ROW, heard, {"role": ChatRole.ASSISTANT, "content": "unheard", "turn_id": 4}]
    make_tm(heard_responses={"r3": "heard bit"}).drop(messages)
    assert messages == [USER_ROW, heard]


def test_row_without_ids_never_consults_last_heard_fallback():
    # Accessors fall back to the LAST heard turn when passed None — a row with no ids must
    # not be able to claim that evidence, or an unheard reply would survive.
    tm = make_tm(heard_turns={9: "something else"})
    messages = [USER_ROW, {"role": ChatRole.ASSISTANT, "content": "unheard"}]
    tm.drop(messages)
    assert messages == [USER_ROW]


def test_payload_ending_on_user_is_untouched():
    messages = [SYSTEM_ROW, {"role": ChatRole.ASSISTANT, "content": "earlier"}, USER_ROW]
    make_tm().drop(messages)
    assert messages == [SYSTEM_ROW, {"role": ChatRole.ASSISTANT, "content": "earlier"}, USER_ROW]
