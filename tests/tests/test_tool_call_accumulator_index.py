"""Tests for ToolCallAccumulator.build_api_payload index handling.

final_tool_calls is keyed by the provider's streamed tool-call index, not by
list position. build_api_payload used to read final_tool_calls[0], which assumes
the first tool call always has index 0. That holds for OpenAI/Azure but is not
guaranteed across providers (LiteLLM normalizes several backends), so a non-zero
first index raised KeyError and dropped the whole function call. These verify the
first accumulated call is used regardless of its index, and index 0 still works.
"""

import json
from types import SimpleNamespace

from bolna.llms.tool_call_accumulator import ToolCallAccumulator


def _delta(index, call_id, name, arguments):
    """Mimic one streamed tool-call delta chunk (OpenAI/LiteLLM shape)."""
    return SimpleNamespace(
        index=index,
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _accumulator_for(func_name):
    api_params = {
        func_name: {
            "url": "https://example.com/api",
            "method": "POST",
            "param": None,
            "api_token": None,
            "headers": None,
        }
    }
    tools = [
        {
            "type": "function",
            "function": {
                "name": func_name,
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]
    return ToolCallAccumulator(
        api_params=api_params, tools=tools, language="en", model="gpt-4o-mini", run_id="run-1"
    )


def _no_op_request_log(monkeypatch):
    # convert_to_request_log schedules an asyncio task and writes trace logs;
    # stub it so the unit test needs no event loop and touches no files.
    monkeypatch.setattr(
        "bolna.llms.tool_call_accumulator.convert_to_request_log",
        lambda *args, **kwargs: None,
    )


def test_build_api_payload_with_nonzero_first_index(monkeypatch):
    _no_op_request_log(monkeypatch)
    acc = _accumulator_for("get_weather")
    # Provider streams the only tool call with index 1 (not 0).
    acc.process_delta([_delta(1, "call_abc", "get_weather", json.dumps({"city": "Paris"}))])

    payload = acc.build_api_payload(model_args={"model": "gpt-4o-mini"}, meta_info={}, answer="")

    assert payload is not None
    assert payload.called_fun == "get_weather"
    assert payload.tool_call_id == "call_abc"
    assert getattr(payload, "city") == "Paris"


def test_build_api_payload_zero_index_still_works(monkeypatch):
    _no_op_request_log(monkeypatch)
    acc = _accumulator_for("get_weather")
    acc.process_delta([_delta(0, "call_xyz", "get_weather", json.dumps({"city": "Berlin"}))])

    payload = acc.build_api_payload(model_args={"model": "gpt-4o-mini"}, meta_info={}, answer="")

    assert payload is not None
    assert payload.tool_call_id == "call_xyz"
    assert getattr(payload, "city") == "Berlin"


def test_build_api_payload_uses_insertion_order_not_numeric_min(monkeypatch):
    """Deltas can arrive with any index the provider assigns; build_api_payload must use
    the FIRST one accumulated (insertion order), not search for the numerically smallest
    index. Regression guard against a fix that swaps [0] for min(final_tool_calls) instead
    of next(iter(...values()))."""
    _no_op_request_log(monkeypatch)
    acc = _accumulator_for("get_weather")
    # First delta received has index=3; a second (different-index) delta arrives after.
    acc.process_delta([_delta(3, "call_first", "get_weather", json.dumps({"city": "Rome"}))])
    acc.process_delta([_delta(1, "call_second", "get_weather", json.dumps({"city": "Oslo"}))])

    payload = acc.build_api_payload(model_args={"model": "gpt-4o-mini"}, meta_info={}, answer="")

    assert payload is not None
    assert payload.tool_call_id == "call_first"
    assert getattr(payload, "city") == "Rome"
