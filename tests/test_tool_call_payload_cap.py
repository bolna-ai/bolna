"""Tool payloads are capped where they are captured: nothing downstream rejects one on size."""

import json

import pytest

from bolna.agent_manager.task_manager import (
    MAX_TOOL_CALL_FIELD_BYTES,
    TOOL_CALL_PREVIEW_CHARS,
    TaskManager,
    cap_tool_payload,
)


def _oversized_text(over=2048):
    return "x" * (MAX_TOOL_CALL_FIELD_BYTES + over)


def test_small_payload_passes_through_unchanged():
    value = {"order_id": "A-1", "status": "shipped"}
    assert cap_tool_payload(value, "response_json") == value


def test_small_payload_is_copied_not_aliased():
    value = {"nested": {"k": "v"}}
    out = cap_tool_payload(value, "request_body")
    value["nested"]["k"] = "mutated"
    assert out["nested"]["k"] == "v"


def test_none_stays_none():
    assert cap_tool_payload(None, "response_body") is None


def test_oversized_payload_is_replaced_with_a_marker():
    out = cap_tool_payload({"blob": _oversized_text()}, "response_body")
    assert out["_truncated"] is True
    assert out["_original_bytes"] > MAX_TOOL_CALL_FIELD_BYTES
    assert len(out["_preview"]) == TOOL_CALL_PREVIEW_CHARS


def test_replacement_is_valid_json():
    out = cap_tool_payload({"blob": _oversized_text()}, "response_body")
    assert json.loads(json.dumps(out)) == out


def test_replacement_is_far_under_the_cap():
    out = cap_tool_payload({"blob": _oversized_text(over=5_000_000)}, "response_body")
    assert len(json.dumps(out).encode("utf-8")) < MAX_TOOL_CALL_FIELD_BYTES


def test_raw_string_response_is_measured_unescaped():
    assert cap_tool_payload("x" * 100, "response_body") == "x" * 100
    assert cap_tool_payload(_oversized_text(), "response_body")["_truncated"] is True


def test_unserialisable_value_still_gets_measured():
    class Blob:
        def __repr__(self):
            return "b" * (MAX_TOOL_CALL_FIELD_BYTES + 10)

    assert cap_tool_payload(Blob(), "runtime_args")["_truncated"] is True


@pytest.mark.parametrize("size", [0, 1, MAX_TOOL_CALL_FIELD_BYTES - 3])
def test_boundary_values_are_kept(size):
    value = "x" * size
    assert cap_tool_payload(value, "request_body") == value


def test_finalize_caps_both_halves_of_an_oversized_response():
    payload = json.dumps({"rows": [{"v": _oversized_text(over=0)}]})
    detail = {"started_at": None}
    TaskManager._finalize_api_call_detail(detail, response=payload, status_code=200)

    assert detail["response_body"]["_truncated"] is True
    assert detail["response_json"]["_truncated"] is True


def test_finalize_keeps_a_normal_response_intact():
    detail = {"started_at": None}
    TaskManager._finalize_api_call_detail(detail, response='{"success": true}', status_code=200)

    assert detail["response_body"] == '{"success": true}'
    assert detail["response_json"] == {"success": True}


def test_finalize_leaves_response_json_none_for_non_json():
    detail = {"started_at": None}
    TaskManager._finalize_api_call_detail(detail, response="<html>not json</html>", status_code=405)

    assert detail["response_body"] == "<html>not json</html>"
    assert detail["response_json"] is None
