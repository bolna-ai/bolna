import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from bolna.llms.openai_llm import OpenAiLLM


def _make_llm(**overrides):
    defaults = {
        "model": "gpt-4o",
        "previous_response_id": None,
        "_pending_call_ids": set(),
        "_interruption_hint": None,
    }
    defaults.update(overrides)
    with patch.object(OpenAiLLM, "__init__", lambda self, **kw: None):
        llm = OpenAiLLM.__new__(OpenAiLLM)
    llm.model = defaults["model"]
    llm.max_tokens = 100
    llm.buffer_size = 40
    llm.temperature = 0.1
    llm.run_id = "run_123"
    llm.language = "en"
    llm.trigger_function_call = False
    llm.api_params = {}
    llm.tools = []
    llm.started_streaming = False
    llm.llm_host = None
    llm.use_responses_api = True
    llm.previous_response_id = defaults["previous_response_id"]
    llm._pending_call_ids = defaults["_pending_call_ids"]
    llm.compact_threshold = None
    llm._interruption_hint = defaults["_interruption_hint"]
    llm._ws_transport = None
    llm.async_client = AsyncMock()
    llm.model_args = {"model": llm.model, "max_tokens": llm.max_tokens, "temperature": llm.temperature}
    return llm


SYSTEM_USER_ASSISTANT_USER = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": "hello there"},
    {"role": "user", "content": "tell me a joke"},
]


class TestInterruptionHintInjection:
    def test_barge_in_preserves_previous_response_id(self):
        llm = _make_llm(previous_response_id="resp_prev")
        llm.set_interruption_hint("hello th")

        _, items = llm._build_responses_input(SYSTEM_USER_ASSISTANT_USER)

        assert llm.previous_response_id == "resp_prev"
        assert items[0]["role"] == "developer"
        assert "hello th" in items[0]["content"]
        assert items[1]["role"] == "user"
        assert items[1]["content"] == "tell me a joke"

    def test_hint_is_one_shot(self):
        llm = _make_llm(previous_response_id="resp_prev")
        llm.set_interruption_hint("partial")

        _, items_first = llm._build_responses_input(SYSTEM_USER_ASSISTANT_USER)
        _, items_second = llm._build_responses_input(SYSTEM_USER_ASSISTANT_USER)

        assert items_first[0]["role"] == "developer"
        assert items_second[0]["role"] == "user"
        assert llm._interruption_hint is None

    def test_hint_not_injected_without_chain(self):
        llm = _make_llm(previous_response_id=None)
        llm.set_interruption_hint("ignored")

        _, items = llm._build_responses_input(SYSTEM_USER_ASSISTANT_USER)

        assert all(item.get("role") != "developer" for item in items)
        assert llm._interruption_hint is None

    def test_hint_consumed_on_pending_tool_fallback(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "check order"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call_1", "function": {"name": "get_order", "arguments": "{}"}}],
            },
        ]
        llm = _make_llm(
            previous_response_id="resp_with_tool",
            _pending_call_ids={"call_1"},
        )
        llm.set_interruption_hint("let me che")

        _, items = llm._build_responses_input(messages)

        assert llm.previous_response_id is None
        assert llm._interruption_hint is None
        assert all(item.get("role") != "developer" for item in items)

    def test_hint_injected_when_tool_outputs_present(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "check order"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call_1", "function": {"name": "get_order", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "shipped"},
            {"role": "user", "content": "thanks"},
        ]
        llm = _make_llm(
            previous_response_id="resp_with_tool",
            _pending_call_ids={"call_1"},
        )
        llm.set_interruption_hint("your order is shi")

        _, items = llm._build_responses_input(messages)

        assert llm.previous_response_id == "resp_with_tool"
        assert items[0]["role"] == "developer"
        assert "your order is shi" in items[0]["content"]
        types = [item.get("type") for item in items]
        assert "function_call_output" in types

    def test_invalidate_clears_hint(self):
        llm = _make_llm(previous_response_id="resp_prev")
        llm.set_interruption_hint("heard text")
        assert llm._interruption_hint == "heard text"

        llm.invalidate_response_chain()

        assert llm._interruption_hint is None
        assert llm.previous_response_id is None
        assert llm._pending_call_ids == set()

    def test_set_interruption_hint_with_none(self):
        llm = _make_llm(previous_response_id="resp_prev")
        llm.set_interruption_hint(None)
        assert llm._interruption_hint == ""

        _, items = llm._build_responses_input(SYSTEM_USER_ASSISTANT_USER)
        assert items[0]["role"] == "developer"
        assert '""' in items[0]["content"]


class TestCancelInFlightResponse:
    def test_cancel_in_flight_no_transport_is_noop(self):
        llm = _make_llm(previous_response_id="resp_1")
        llm._ws_transport = None
        llm.cancel_in_flight_response()
        assert llm.previous_response_id == "resp_1"

    def test_cancel_in_flight_no_response_id_is_noop(self):
        llm = _make_llm(previous_response_id=None)
        llm._ws_transport = MagicMock()
        llm.cancel_in_flight_response()
        llm._ws_transport.cancel_response.assert_not_called()

    def test_cancel_in_flight_does_not_invalidate_chain(self):
        import asyncio

        llm = _make_llm(previous_response_id="resp_1")
        llm._ws_transport = MagicMock()
        llm._ws_transport.cancel_response = AsyncMock()

        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            llm.cancel_in_flight_response()
            pending = asyncio.all_tasks(loop)
            loop.run_until_complete(asyncio.gather(*pending))
        finally:
            loop.close()
            asyncio.set_event_loop(None)

        assert llm.previous_response_id == "resp_1"
        assert llm._pending_call_ids == set()
        llm._ws_transport.cancel_response.assert_called_once_with("resp_1")
