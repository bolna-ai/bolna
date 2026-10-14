"""A tool call that names a configured tool imperfectly still runs that tool.

A model may call a custom tool without its custom_task_ prefix, padded, namespaced or re-cased.
Such a call must reach the configured tool, with its own filler, config and required-fields check,
instead of going out without its configuration.
"""

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from google.genai import types

from bolna.helpers.function_calling_helpers import resolve_tool_name
from bolna.llms.gemini_llm import GeminiLLM
from bolna.llms.openai_llm import OpenAiLLM
from bolna.llms.tool_call_accumulator import ToolCallAccumulator

URL = "https://api.example.com/tools"
TOOL = "custom_task_lookup_order"
FILLER = "One moment please"
TOOLS_PARAMS = {
    TOOL: {
        "url": URL,
        "method": "POST",
        "param": {"order_id": "%(order_id)s"},
        "api_token": "Bearer tok",
        "headers": {"X-Api-Key": "key"},
        "pre_call_message": FILLER,
    },
    "transfer_call_support": {"url": "https://api.example.com/transfer"},
    "end_call": {"pre_call_message": None},
}
PARAMETERS = {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}
DECLARATION = {
    "type": "function",
    "function": {"name": TOOL, "description": "Look up an order", "parameters": PARAMETERS},
}


async def _aiter(items):
    for item in items:
        yield item


class TestResolveToolName:
    def test_an_exact_key_is_returned_unchanged(self):
        assert resolve_tool_name(TOOL, TOOLS_PARAMS) == TOOL

    def test_an_exact_builtin_is_never_remapped_to_a_custom_tool_of_the_same_name(self):
        assert resolve_tool_name("end_call", {"end_call": {}, "custom_task_end_call": {}}) == "end_call"

    def test_a_bare_name_resolves_to_its_prefixed_custom_tool(self):
        assert resolve_tool_name("lookup_order", TOOLS_PARAMS) == TOOL

    def test_a_recased_builtin_resolves_to_the_builtin(self):
        assert resolve_tool_name("Transfer_Call_Support", TOOLS_PARAMS) == "transfer_call_support"

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


class TestGemini:
    @staticmethod
    def _llm():
        return GeminiLLM(
            model="gemini-2.5-flash", llm_key="k", api_tools={"tools": [DECLARATION], "tools_params": TOOLS_PARAMS}
        )

    @staticmethod
    async def _stream(llm, name="lookup_order", args=None):
        call = types.FunctionCall(name=name, args={"order_id": "A-100"} if args is None else args, id="call_1")
        part = types.Part(function_call=call, thought_signature=b"signature")
        chunk = SimpleNamespace(
            usage_metadata=None,
            candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part]))],
            response_id="r1",
            text=None,
        )
        llm.client = MagicMock()
        llm.client.aio.models.generate_content_stream = AsyncMock(return_value=_aiter([chunk]))
        with patch("bolna.llms.gemini_llm.convert_to_request_log"):
            return [c async for c in llm.generate_stream([{"role": "user", "content": "hi"}], meta_info={})]

    @staticmethod
    def _payload(chunks):
        return next(c.data for c in chunks if c.is_function_call)

    async def test_a_bare_name_runs_the_configured_tool(self):
        payload = self._payload(await self._stream(self._llm()))

        assert payload.called_fun == TOOL
        assert (payload.url, payload.method, payload.param) == (URL, "post", TOOLS_PARAMS[TOOL]["param"])
        assert (payload.api_token, payload.headers) == ("Bearer tok", {"X-Api-Key": "key"})
        assert payload.order_id == "A-100"

    async def test_the_filler_is_the_configured_tools_own(self):
        filler = next(c for c in await self._stream(self._llm()) if c.function_name)
        assert (filler.data, filler.function_name) == (FILLER, TOOL)

    async def test_the_required_fields_check_runs_against_the_configured_tool(self, caplog):
        with caplog.at_level(logging.WARNING, logger="bolna.llms.gemini_llm"):
            await self._stream(self._llm(), args={})
        assert "missing=['order_id']" in caplog.text

    async def test_a_recased_builtin_reaches_its_own_branch(self):
        """task_manager branches on the called_fun prefix, so a builtin must arrive resolved."""
        payload = self._payload(await self._stream(self._llm(), name="Transfer_Call_Support"))
        assert payload.called_fun == "transfer_call_support"

    async def test_history_keeps_the_emitted_name_with_and_without_the_cached_part(self):
        """The thought_signature belongs to the call as emitted, so history must replay that name."""
        llm = self._llm()
        payload = self._payload(await self._stream(llm))
        assert payload.model_response[-1]["function"]["name"] == "lookup_order"

        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": None, "tool_calls": payload.model_response},
            {"role": "tool", "tool_call_id": "call_1", "content": '{"status": "shipped"}'},
        ]
        for cache_hit in (True, False):
            if not cache_hit:
                llm._native_function_parts.clear()
            _, history = llm._prepare_history(messages)
            call_part = next(p for p in history[1].parts if p.function_call)

            assert call_part.function_call.name == "lookup_order"
            assert call_part.thought_signature == b"signature"
            assert history[2].parts[0].function_response.name == "lookup_order"


class TestChatCompletions:
    """OpenAI, Azure and every LiteLLM provider share ToolCallAccumulator."""

    @staticmethod
    def _accumulator():
        acc = ToolCallAccumulator(TOOLS_PARAMS, [DECLARATION], "en", "m", "run-1")
        fn = SimpleNamespace(name="lookup_order", arguments='{"order_id": "A-100"}')
        acc.process_delta([SimpleNamespace(index=0, id="call_1", function=fn)])
        return acc

    def test_a_bare_name_runs_the_configured_tool_and_history_records_it(self):
        with patch("bolna.llms.tool_call_accumulator.convert_to_request_log"):
            payload = self._accumulator().build_api_payload({}, {}, "")

        assert (payload.called_fun, payload.url, payload.order_id) == (TOOL, URL, "A-100")
        assert payload.model_response[0]["function"]["name"] == TOOL

    def test_the_filler_is_the_configured_tools_own(self):
        assert self._accumulator().get_pre_call_message({}) == (FILLER, TOOL, FILLER)


class TestResponsesApi:
    """OpenAI and Azure share the Responses API stream (HTTP and WebSocket) and the text rescue."""

    @staticmethod
    def _openai():
        llm = OpenAiLLM(
            model="gpt-4.1-mini", llm_key="sk-dummy", api_tools={"tools": [DECLARATION], "tools_params": TOOLS_PARAMS}
        )
        llm.use_responses_api = True
        return llm

    @staticmethod
    def _assert_resolved(chunks):
        filler = next(c for c in chunks if c.function_name)
        payload = next(c.data for c in chunks if c.is_function_call)

        assert (filler.data, filler.function_name) == (FILLER, TOOL)
        assert (payload.called_fun, payload.url, payload.order_id) == (TOOL, URL, "A-100")
        assert payload.model_response[0]["function"]["name"] == TOOL

    async def test_http_stream_resolves_the_filler_and_the_call(self):
        llm = self._openai()
        item = SimpleNamespace(type="function_call", id="fc_1", name="lookup_order", call_id="call_1")
        events = [
            SimpleNamespace(type="response.output_item.added", item=item),
            SimpleNamespace(
                type="response.function_call_arguments.delta", item_id="fc_1", delta='{"order_id": "A-100"}'
            ),
            SimpleNamespace(type="response.completed", response=SimpleNamespace(id="resp_1")),
        ]
        llm.async_client.responses.create = AsyncMock(return_value=_aiter(events))
        with patch("bolna.llms.openai_base.convert_to_request_log"):
            chunks = [
                c async for c in llm._generate_stream_responses([{"role": "user", "content": "hi"}], meta_info={})
            ]

        self._assert_resolved(chunks)

    async def test_websocket_stream_resolves_the_filler_and_the_call(self):
        llm = self._openai()
        item = {"type": "function_call", "id": "fc_1", "name": "lookup_order", "call_id": "call_1"}
        events = [
            {"type": "response.output_item.added", "item": item},
            {"type": "response.function_call_arguments.delta", "item_id": "fc_1", "delta": '{"order_id": "A-100"}'},
            {"type": "response.completed", "response": {"id": "resp_1"}},
        ]
        llm._ws_transport = SimpleNamespace(stream_response=lambda params: _aiter(events))
        with patch("bolna.llms.openai_base.convert_to_request_log"):
            chunks = [
                c async for c in llm._generate_stream_ws_responses([{"role": "user", "content": "hi"}], meta_info={})
            ]

        self._assert_resolved(chunks)

    def test_a_tool_call_written_as_text_is_rescued_as_the_configured_tool(self):
        chunk = self._openai()._try_rescue_text_tool_call(
            'functions.lookup_order({"order_id": "A-100"})', {"tools": [DECLARATION]}, {}, "", None
        )

        assert chunk is not None and chunk.is_function_call
        assert (chunk.data.called_fun, chunk.data.url, chunk.data.order_id) == (TOOL, URL, "A-100")
