"""Streamed tool-call deltas becoming one function call, shared by the OpenAI, Azure and LiteLLM paths.

Arguments arrive split across chunks, so they must be rejoined in order or the tool receives
malformed JSON. A call whose required arguments never arrived must not run at all, and the
user-facing filler must be offered exactly once, since a second one talks over the first.
"""

from types import SimpleNamespace

import pytest
from unittest.mock import patch

from bolna.llms.tool_call_accumulator import ToolCallAccumulator

TOOLS = [
    {"function": {"name": "book", "parameters": {"required": ["date"], "properties": {}}}},
    {"function": {"name": "lookup", "parameters": {"required": [], "properties": {}}}},
]
API_PARAMS = {
    "book": {"url": "https://example.test/book", "method": "POST"},
    "lookup": {"url": "https://example.test/lookup", "method": "GET"},
    "custom": {"url": "https://example.test/c", "method": "POST", "pre_call_message": "One moment please"},
    "end_call_now": {"url": "https://example.test/e", "method": "POST"},
    "switch_language": {"url": "https://example.test/s", "method": "POST"},
}


@pytest.fixture(autouse=True)
def _no_request_logging():
    """The success path logs the call, which needs a live loop and is not what these tests pin."""
    with patch("bolna.llms.tool_call_accumulator.convert_to_request_log"):
        yield


def _delta(index, name=None, arguments=None, call_id=None):
    return SimpleNamespace(index=index, id=call_id, function=SimpleNamespace(name=name, arguments=arguments))


def _accumulator(language="en"):
    return ToolCallAccumulator(API_PARAMS, TOOLS, language, "gpt-4o", "run-1")


def _booking(*chunks, call_id="call-1"):
    acc = _accumulator()
    acc.process_delta([_delta(0, "book", chunks[0], call_id=call_id)])
    for chunk in chunks[1:]:
        acc.process_delta([_delta(0, None, chunk)])
    return acc


def test_arguments_split_across_chunks_are_rejoined_in_order():
    acc = _booking('{"da', 'te": ', '"friday"}')
    assert acc.final_tool_calls[0]["function"]["arguments"] == '{"date": "friday"}'


def test_parsed_arguments_land_on_the_payload():
    payload = _booking('{"date": "friday"}').build_api_payload({"model": "m"}, {"request_id": "q"}, "")
    assert payload.date == "friday"
    assert payload.called_fun == "book"
    assert payload.tool_call_id == "call-1"
    assert payload.url == "https://example.test/book"


def test_method_is_lowercased_for_the_http_layer():
    assert _booking('{"date": "friday"}').build_api_payload({}, {}, "").method == "post"


def test_a_missing_required_argument_is_never_applied():
    """Running the tool with a half-filled payload is worse than not running it."""
    payload = _booking('{"note": "no date here"}').build_api_payload({}, {}, "")
    assert not hasattr(payload, "date")


def test_malformed_arguments_do_not_raise():
    payload = _booking('{"date": ').build_api_payload({}, {}, "")
    assert payload is not None
    assert not hasattr(payload, "date")


def test_a_function_outside_api_params_yields_no_payload():
    """The model can name anything; only configured tools may be called."""
    acc = _accumulator()
    acc.process_delta([_delta(0, "definitely_not_configured", "{}", call_id="c")])
    assert acc.build_api_payload({}, {}, "") is None


def test_no_tool_call_yields_no_payload():
    assert _accumulator().build_api_payload({}, {}, "") is None


def test_a_textual_response_is_carried_only_when_one_was_streamed():
    acc = _booking('{"date": "friday"}')
    assert acc.build_api_payload({}, {}, "  spoken words  ").textual_response is None

    acc = _booking('{"date": "friday"}')
    acc.received_textual = True
    assert acc.build_api_payload({}, {}, "  spoken words  ").textual_response == "spoken words"


def test_the_filler_is_offered_once_only():
    """A second filler for the same call would talk over the first."""
    acc = _booking("{}")
    message, name, _ = acc.get_pre_call_message({})
    assert message and name == "book"
    assert acc.get_pre_call_message({}) is None


def test_end_call_gets_no_filler():
    """The goodbye is the model's own text; a filler would speak over it."""
    acc = _accumulator()
    acc.process_delta([_delta(0, "end_call_now", "{}", call_id="c")])
    assert acc.get_pre_call_message({}) is None


def test_switching_language_is_silent():
    acc = _accumulator()
    acc.process_delta([_delta(0, "switch_language", "{}", call_id="c")])
    assert acc.get_pre_call_message({})[0] == ""


def test_text_already_streamed_suppresses_the_filler():
    acc = _accumulator()
    acc.received_textual = True
    acc.process_delta([_delta(0, "book", "{}", call_id="c")])
    assert acc.get_pre_call_message({}) is None


def test_a_per_tool_message_overrides_the_default():
    acc = _accumulator()
    acc.process_delta([_delta(0, "custom", "{}", call_id="c")])
    assert acc.get_pre_call_message({})[0] == "One moment please"


def test_the_detected_language_picks_the_variant():
    acc = _booking("{}")
    assert acc.get_pre_call_message({"detected_language": "ge"})[0].startswith("Geben")


def test_an_unknown_language_falls_back_to_english_rather_than_silence():
    acc = _booking("{}")
    assert acc.get_pre_call_message({"detected_language": "xx"})[0].startswith("Just give me")


def test_parallel_calls_accumulate_independently():
    acc = _accumulator()
    acc.process_delta([_delta(0, "book", '{"date":"fri"}', call_id="i0"), _delta(1, "lookup", '{"q":', call_id="i1")])
    acc.process_delta([_delta(1, None, '"hours"}')])

    assert [v["function"]["arguments"] for v in acc.final_tool_calls.values()] == ['{"date":"fri"}', '{"q":"hours"}']
    assert len(acc.build_api_payload({}, {}, "").model_response) == 2
