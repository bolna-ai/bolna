"""Regression tests for BLT-018 — an action tool re-emitted on rapid closing acks re-hits its API.

Fix: __execute_function_call skips a repeat whose signature (tool name + resolved args) was already
executed this call, recording an "already done" result instead (same shape as the transfer_call guard).
These tests pin _tool_call_signature: identical calls collapse (deduped), genuinely different calls do
not (still run), and OpenAI plumbing keys never affect the identity.
"""

from bolna.agent_manager.task_manager import TaskManager


sig = TaskManager._tool_call_signature  # static method, call directly


def _reschedule(mentioned_time, **extra):
    return {
        "function_name": "reschedule_call",
        "execution_id": "run-1",
        "mentionedTime": mentioned_time,
        "model_response": [{"id": "call_x", "type": "function"}],
        "tool_call_id": "call_x",
        "textual_response": "Your call has been successfully rescheduled.",
        **extra,
    }


def test_identical_reschedule_calls_share_a_signature():
    # The BLT-018 case: reschedule re-emitted with the same args -> deduped (second is a no-op).
    assert sig("custom_task_reschedule_call", _reschedule("later")) == sig(
        "custom_task_reschedule_call", _reschedule("later")
    )


def test_plumbing_keys_do_not_affect_identity():
    # A fresh generation gives new model_response / tool_call_id / textual_response, but it's the
    # same action -> must still match so the duplicate is caught.
    a = _reschedule("later", model_response=[{"id": "call_1"}], tool_call_id="call_1")
    b = _reschedule("later", model_response=[{"id": "call_2"}], tool_call_id="call_2")
    assert sig("custom_task_reschedule_call", a) == sig("custom_task_reschedule_call", b)


def test_different_args_are_distinct_calls():
    # A different requested time is a genuinely different reschedule -> must NOT be deduped.
    assert sig("custom_task_reschedule_call", _reschedule("later")) != sig(
        "custom_task_reschedule_call", _reschedule("tomorrow 5pm")
    )


def test_non_purchase_reason_with_different_reasons_not_deduped():
    # Documents the deliberate boundary: different reason text = a distinct call, so it still runs.
    r1 = {"function_name": "non_purchase_reason", "execution_id": "run-1", "reason": "too expensive"}
    r2 = {"function_name": "non_purchase_reason", "execution_id": "run-1", "reason": "already bought"}
    assert sig("custom_task_non_purchase_reason", r1) != sig("custom_task_non_purchase_reason", r2)


def test_different_tools_are_distinct():
    payload = {"execution_id": "run-1", "function_name": "x", "query": "jackets"}
    assert sig("custom_task_product_search", payload) != sig("custom_task_inventory_question", payload)


def test_signature_is_order_independent():
    # Same args in a different dict order must produce the same signature.
    a = {"execution_id": "run-1", "function_name": "reschedule_call", "mentionedTime": "later"}
    b = {"mentionedTime": "later", "function_name": "reschedule_call", "execution_id": "run-1"}
    assert sig("custom_task_reschedule_call", a) == sig("custom_task_reschedule_call", b)
