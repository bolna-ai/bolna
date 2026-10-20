"""Model tool arguments remain request data across provider and executor boundaries.

HTTP coverage uses loopback and dummy credentials; no provider request is made.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import web
from google.genai import types

from bolna.agent_manager.task_manager import TaskManager
from bolna.helpers.conversation_history import ConversationHistory
from bolna.llms.gemini_llm import GeminiLLM
from bolna.llms.openai_llm import OpenAiLLM
from bolna.llms.tool_call_accumulator import ToolCallAccumulator
from bolna.llms.types import FunctionCallPayload

TOOL = "custom_task_save_link"
META = {"turn_id": 1, "sequence_id": 1, "response_uid": "original-response"}
PROVIDERS = ["chat", "responses", "text_rescue", "gemini"]
COLLISIONS = {
    "url": "https://example.test/caller-link",
    "method": "DELETE",
    "param": {"replaced": True},
    "api_token": "Bearer model-token",
    "headers": {"Authorization": "Bearer model-header"},
    "model_args": {"model": "model-override"},
    "meta_info": {"turn_id": 999},
    "called_fun": "end_call",
    "model_response": [],
    "tool_call_id": "model-call-id",
    "textual_response": "model-text",
    "resp": "model-response",
    "next_step": "model-step",
    "run_id": "model-run",
    "headers_data": {"Authorization": "Bearer model-header-data"},
    "return_response_metadata": False,
    "arguments": {"url": "nested-data"},
    "execution_id": "model-execution",
    "date": "friday",
}


@pytest.fixture(autouse=True)
def no_request_logs(monkeypatch):
    for module in (
        "llms.tool_call_accumulator",
        "llms.openai_base",
        "llms.gemini_llm",
        "agent_manager.task_manager",
        "helpers.function_calling_helpers",
    ):
        monkeypatch.setattr(f"bolna.{module}.convert_to_request_log", MagicMock())


async def build_payload(provider, config, args):
    schema = {
        "type": "object",
        "properties": {key: {} for key in COLLISIONS},
        "required": [],
        "additionalProperties": False,
    }
    declaration = {
        "type": "function",
        "function": {"name": TOOL, "description": "Save caller data", "parameters": schema},
    }
    model_args = {"model": "offline"}
    if provider == "chat":
        accumulator = ToolCallAccumulator({TOOL: config}, [declaration], "en", "offline", "local-test")
        accumulator.process_delta(
            [SimpleNamespace(index=0, id="call-1", function=SimpleNamespace(name=TOOL, arguments=json.dumps(args)))]
        )
        return accumulator.build_api_payload(model_args, dict(META), "")
    if provider in ("responses", "text_rescue"):
        llm = OpenAiLLM(
            model="gpt-4.1-mini", llm_key="sk-dummy", api_tools={"tools": [declaration], "tools_params": {TOOL: config}}
        )
        if provider == "text_rescue":
            return llm._try_rescue_text_tool_call(
                f"functions.{TOOL}({json.dumps(args)})", {"tools": [declaration]}, dict(META), "", None
            ).data
        return llm._build_function_call_chunk(
            {"item-1": json.dumps(args)},
            {"item-1": TOOL},
            {"item-1": "call-1"},
            [{"name": TOOL, "parameters": schema}],
            model_args,
            dict(META),
            "",
            False,
            None,
        ).data
    llm = GeminiLLM(
        model="gemini-2.5-flash", llm_key="dummy", api_tools={"tools": [declaration], "tools_params": {TOOL: config}}
    )

    async def stream():
        yield SimpleNamespace(
            usage_metadata=None,
            response_id="response-1",
            text=None,
            candidates=[
                SimpleNamespace(
                    content=SimpleNamespace(
                        parts=[types.Part(function_call=types.FunctionCall(name=TOOL, args=args, id="call-1"))]
                    )
                )
            ],
        )

    llm.client = MagicMock()
    llm.client.aio.models.generate_content_stream = AsyncMock(return_value=stream())
    return next(
        chunk.data
        for chunk in [
            c async for c in llm.generate_stream([{"role": "user", "content": "save this"}], meta_info=dict(META))
        ]
        if chunk.is_function_call
    )


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("field", COLLISIONS)
async def test_model_arguments_cannot_replace_execution_fields(provider, field):
    config = {
        "url": "https://example.test/configured",
        "method": "POST",
        "api_token": "Bearer dummy-token",
        "headers": {"X-Configured": "yes"},
        "param": {"event": "save"},
    }
    baseline = await build_payload(provider, config, {})
    args = {field: COLLISIONS[field]}
    payload = await build_payload(provider, config, args)
    for name in FunctionCallPayload.model_fields:
        if name not in ("arguments", "model_response"):
            assert getattr(payload, name) == getattr(baseline, name), name
    assert payload.arguments == args
    assert json.loads(payload.model_response[0]["function"]["arguments"]) == args
    assert not payload.model_extra


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("method", ["POST", "GET"])
@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("template", ["markers", "legacy"])
async def test_only_configured_endpoint_receives_credentials_and_arguments(
    provider, template, graph, method, monkeypatch
):
    received = []

    async def record(request):
        received.append(
            (
                request.method,
                request.path,
                request.headers.get("Authorization"),
                request.headers.get("X-Configured"),
                dict(request.query) if request.method == "GET" else await request.json(),
            )
        )
        return web.json_response({"ok": True})

    app = web.Application()
    app.router.add_route("*", "/{path}", record)
    runner = web.AppRunner(app)
    await runner.setup()
    try:
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        base = f"http://127.0.0.1:{runner.addresses[0][1]}"
        args = {**COLLISIONS, "url": base + "/caller-link"}
        param = (
            {key: {"$var": key} for key in args}
            if template == "markers"
            else {key: f"%({key})s" for key in args if isinstance(args[key], str)}
        )
        config = {
            "url": base + "/configured",
            "method": method,
            "param": param,
            "api_token": "Bearer dummy-token",
            "headers": {"X-Configured": "yes"},
        }
        payload = await build_payload(provider, config, args)
        history = ConversationHistory()
        history.append_user("save this")
        tm = SimpleNamespace(
            run_id="local-test",
            kwargs={},
            hangup_triggered=False,
            conversation_ended=False,
            conversation_config={},
            conversation_history=history,
            llm_config={"model": "offline"},
            wait_for_current_message=AsyncMock(),
            _caller_disconnected=lambda: False,
            _TaskManager__is_graph_agent=lambda: graph,
            context_data={},
            tools={"llm_agent": SimpleNamespace(context_data={})},
            _extract_api_call_runtime_args=TaskManager._extract_api_call_runtime_args,
            _start_api_call_detail=MagicMock(return_value={}),
            _finalize_api_call_detail=MagicMock(),
            _spawn_followup_meta_info=lambda value: dict(value),
            _TaskManager__do_llm_generation=AsyncMock(),
        )
        monkeypatch.setattr("bolna.helpers.function_calling_helpers._ALLOWLISTED_HOSTS", frozenset({"127.0.0.1"}))
        await TaskManager._TaskManager__execute_function_call(tm, next_step="synthesizer", **payload.model_dump())
        expected = {key: args[key] for key in param}
        # execution_id is bound to this call rather than trusting the generated value.
        expected["execution_id"] = tm.run_id
        if method == "GET":
            expected = {key: str(value) for key, value in expected.items()}
        assert received == [(method, "/configured", "Bearer dummy-token", "yes", expected)]
        assert tm._start_api_call_detail.call_args.kwargs["url"] == config["url"]
        assert tm._TaskManager__do_llm_generation.await_args.args[1]["turn_id"] == META["turn_id"]
        assert history.get_copy()[-1]["tool_call_id"] == payload.tool_call_id
        if graph:
            assert tm.context_data == tm.tools["llm_agent"].context_data == {"ok": True}
    finally:
        await runner.cleanup()


@pytest.mark.parametrize(
    "tool,args",
    [
        ("end_call", {"reason": "caller finished"}),
        ("switch_language", {"language": "en"}),
        ("web_search", {"query": "opening hours"}),
    ],
)
async def test_builtin_consumers_read_the_argument_mapping(tool, args, monkeypatch):
    history = ConversationHistory()
    history.append_user("help")
    tm = SimpleNamespace(
        run_id="local-test",
        conversation_history=history,
        interruptible_hangup_message=False,
        wait_for_current_message=AsyncMock(),
        _enter_hangup_state=MagicMock(),
        _TaskManager__arm_end_call_teardown=MagicMock(),
        _hangup_cancelled=False,
        language="en",
        web_search_config={},
        hangup_triggered=False,
        conversation_ended=False,
        conversation_config={},
        _run_web_search=AsyncMock(return_value="open at nine"),
        _spawn_followup_meta_info=lambda value: dict(value),
        _TaskManager__do_llm_generation=AsyncMock(),
    )
    log = MagicMock()
    monkeypatch.setattr("bolna.agent_manager.task_manager.convert_to_request_log", log)
    payload = FunctionCallPayload(
        called_fun=tool,
        meta_info=dict(META),
        arguments=args,
        textual_response="Goodbye" if tool == "end_call" else None,
        tool_call_id="builtin-call",
        model_response=[
            {"id": "builtin-call", "type": "function", "function": {"name": tool, "arguments": json.dumps(args)}}
        ],
    )
    await TaskManager._TaskManager__execute_function_call(tm, next_step="synthesizer", **payload.model_dump())
    if tool == "end_call":
        assert json.loads(log.call_args_list[0].args[0])["reason"] == args["reason"]
        tm._TaskManager__arm_end_call_teardown.assert_called_once()
    elif tool == "switch_language":
        assert "Already speaking in en" in history.get_copy()[-1]["content"]
    else:
        tm._run_web_search.assert_awaited_once_with(args["query"], META, "builtin-call")
    assert history.get_copy()[-1]["tool_call_id"] == "builtin-call"
