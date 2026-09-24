"""A tool call that names a configured tool imperfectly still runs that tool.

A model may call a custom tool without its custom_task_ prefix, padded, namespaced or re-cased.
Such a call must reach the configured tool instead of going out without its configuration.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from bolna.agent_manager.task_manager import TaskManager
from bolna.helpers.function_calling_helpers import resolve_tool_name
from bolna.llms.openai_llm import OpenAiLLM
from bolna.llms.tool_call_accumulator import ToolCallAccumulator

URL = "https://api.example.com/tools"
TOOL = "custom_task_lookup_order"
TOOLS_PARAMS = {
    TOOL: {
        "url": URL,
        "method": "POST",
        "param": {"order_id": "%(order_id)s", "execution_id": "%(execution_id)s"},
        "api_token": "Bearer tok",
        "headers": {"X-Api-Key": "key"},
    },
    "end_call": {"pre_call_message": None},
}
PARAMETERS = {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}
DECLARATION = {
    "type": "function",
    "function": {"name": TOOL, "description": "Look up an order", "parameters": PARAMETERS},
}


class TestResolveToolName:
    def test_an_exact_key_is_returned_unchanged(self):
        assert resolve_tool_name(TOOL, TOOLS_PARAMS) == TOOL

    def test_an_exact_builtin_is_never_remapped_to_a_custom_tool_of_the_same_name(self):
        assert resolve_tool_name("end_call", {"end_call": {}, "custom_task_end_call": {}}) == "end_call"

    def test_a_bare_name_resolves_to_its_prefixed_custom_tool(self):
        assert resolve_tool_name("lookup_order", TOOLS_PARAMS) == TOOL

    def test_other_prefixes_are_not_guessed(self):
        assert resolve_tool_name("lookup", {"book_appointment_lookup": {}}) == "lookup"

    @pytest.mark.parametrize("emitted", [" lookup_order ", "functions.lookup_order", "Lookup_Order", TOOL.upper()])
    def test_padding_namespaces_and_case_are_forgiven(self, emitted):
        assert resolve_tool_name(emitted, TOOLS_PARAMS) == TOOL

    def test_an_ambiguous_name_is_left_unresolved(self):
        """Running the wrong tool is worse than running none."""
        params = {"custom_task_Lookup": {}, "custom_task_LOOKUP": {}}
        assert resolve_tool_name("lookup", params) == "lookup"

    @pytest.mark.parametrize("emitted", ["order", "_order", "lookup", "ghost"])
    def test_a_partial_or_unknown_name_is_left_unresolved(self, emitted):
        assert resolve_tool_name(emitted, TOOLS_PARAMS) == emitted

    @pytest.mark.parametrize("tools_params", [None, {}])
    def test_no_configured_tools_leaves_the_name_alone(self, tools_params):
        assert resolve_tool_name("lookup_order", tools_params) == "lookup_order"


class _StopAfterRequest(Exception):
    pass


def _make_tm():
    tm = TaskManager.__new__(TaskManager)
    tm.run_id = "exec-123"
    tm.kwargs = {"api_tools": {"tools_params": TOOLS_PARAMS}}
    tm.hangup_triggered = False
    tm.conversation_ended = False
    tm.function_tool_api_call_details = []
    tm.wait_for_current_message = AsyncMock()
    return tm


async def _execute(tm, called_fun, url=None, method="get", param=None):
    """Run a call handed over with no config, as the LLM layer does for an unknown name."""
    api = AsyncMock(side_effect=_StopAfterRequest)
    with (
        patch("bolna.agent_manager.task_manager.trigger_api", new=api),
        patch("bolna.agent_manager.task_manager.convert_to_request_log"),
        pytest.raises(_StopAfterRequest),
    ):
        await tm._TaskManager__execute_function_call(
            url,
            method,
            param,
            None,  # api_token
            None,  # headers
            {"model": "m"},
            {"turn_id": 1, "sequence_id": 1},
            "llm",
            called_fun,
            model_response=[{"id": "call_1", "function": {"name": called_fun}, "type": "function"}],
            tool_call_id="call_1",
            textual_response=None,
            order_id="A-100",
            execution_id="exec-123",
        )
    return api.await_args.kwargs


class TestExecuteFunctionCall:
    async def test_a_bare_name_runs_the_configured_tool(self):
        tm = _make_tm()
        sent = await _execute(tm, "lookup_order")

        assert sent["url"] == URL
        assert sent["method"] == "post"
        assert sent["param"] == TOOLS_PARAMS[TOOL]["param"]
        assert sent["api_token"] == "Bearer tok"
        assert sent["headers_data"] == {"X-Api-Key": "key"}
        assert sent["called_fun"] == TOOL
        assert sent["order_id"] == "A-100"
        assert tm.function_tool_api_call_details[0]["tool_name"] == TOOL

    async def test_an_exact_name_keeps_the_config_the_llm_layer_resolved(self):
        sent = await _execute(_make_tm(), TOOL, url="https://llm-layer.example/x", method="post", param={"a": 1})
        assert sent["url"] == "https://llm-layer.example/x"
        assert sent["param"] == {"a": 1}

    async def test_an_unknown_name_gets_no_configuration(self):
        sent = await _execute(_make_tm(), "ghost")
        assert sent["url"] is None
        assert sent["called_fun"] == "ghost"


class TestSharedLlmLayer:
    """OpenAI, Azure and every LiteLLM provider share ToolCallAccumulator; OpenAI and Azure also share
    the Responses API builder and the text tool-call rescue."""

    def test_chat_completions_run_the_configured_tool(self):
        acc = ToolCallAccumulator(TOOLS_PARAMS, [DECLARATION], "en", "m", "run-1")
        fn = SimpleNamespace(name="lookup_order", arguments='{"order_id": "A-100"}')
        acc.process_delta([SimpleNamespace(index=0, id="call_1", function=fn)])
        with patch("bolna.llms.tool_call_accumulator.convert_to_request_log"):
            payload = acc.build_api_payload({}, {}, "")

        assert (payload.called_fun, payload.url, payload.order_id) == (TOOL, URL, "A-100")
        assert payload.model_response[0]["function"]["name"] == "lookup_order"

    @staticmethod
    def _openai():
        return OpenAiLLM(
            model="gpt-4.1-mini", llm_key="sk-dummy", api_tools={"tools": [DECLARATION], "tools_params": TOOLS_PARAMS}
        )

    def test_a_responses_api_call_runs_the_configured_tool(self):
        responses_tools = [{"type": "function", "name": TOOL, "parameters": PARAMETERS}]
        with patch("bolna.llms.openai_base.convert_to_request_log"):
            chunk = self._openai()._build_function_call_chunk(
                {"fc_1": '{"order_id": "A-100"}'},
                {"fc_1": "lookup_order"},
                {"fc_1": "call_1"},
                responses_tools,
                {},
                {},
                "",
                False,
                None,
            )

        assert (chunk.data.called_fun, chunk.data.url, chunk.data.order_id) == (TOOL, URL, "A-100")

    def test_a_tool_call_written_as_text_is_rescued_as_the_configured_tool(self):
        chunk = self._openai()._try_rescue_text_tool_call(
            'functions.lookup_order({"order_id": "A-100"})', {"tools": [DECLARATION]}, {}, "", None
        )

        assert chunk is not None and chunk.is_function_call
        assert (chunk.data.called_fun, chunk.data.url, chunk.data.order_id) == (TOOL, URL, "A-100")
