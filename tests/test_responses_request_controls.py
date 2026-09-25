"""Native Responses controls preserve visible context and cannot disguise a strict WS failure."""

import asyncio
import json
import time
from copy import deepcopy
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from openai import APIError
from pydantic import ValidationError
from websockets.protocol import State

from bolna.agent_manager.task_manager import TaskManager
from bolna.llms.openai_llm import OpenAIWSConnection
from bolna.models import LlmAgent, SimpleLlmAgent


HISTORY = [
    {"role": "system", "content": "Be concise."},
    {"role": "user", "content": "Hello", "turn_id": 1},
    {"role": "assistant", "content": "Hello."},
    {"role": "user", "content": "Continue", "turn_id": 2},
]
OPTIONS = {
    "use_responses_api": True,
    "responses_store": False,
    "responses_history": "full",
    "responses_omit_parameters": ["truncation", "include"],
    "prompt_cache_key": "example-agent-v1",
    "omit_request_parameters": ["temperature", "stop", "verbosity"],
    "strict_websocket": True,
}
USAGE = {
    "input_tokens": 24,
    "output_tokens": 8,
    "total_tokens": 32,
    "input_tokens_details": {"cached_tokens": 16},
    "output_tokens_details": {"reasoning_tokens": 3},
}


def native_config(**options):
    agent = LlmAgent.model_validate(
        {
            "agent_type": "simple_llm_agent",
            "agent_flow_type": "streaming",
            "llm_config": {
                "provider": "openai",
                "model": "gpt-5",
                "max_tokens": 4096,
                "reasoning_effort": "medium",
                "use_responses_api": True,
                **options,
            },
        }
    ).model_dump(mode="json")
    task = {
        "task_type": "conversation",
        "toolchain": {"execution": "sequential", "pipelines": [["llm"]]},
        "tools_config": {
            "llm_agent": agent,
            "synthesizer": {"provider": "elevenlabs", "provider_config": {}, "stream": True, "buffer_size": 100},
            "transcriber": {
                "provider": "deepgram",
                "model": "nova-3",
                "language": "en",
                "stream": True,
                "encoding": "linear16",
                "sampling_rate": 16000,
                "endpointing": 250,
            },
            "input": {"provider": "default"},
            "output": {"provider": "default"},
        },
        "task_config": {},
    }
    # Exercise production configuration assembly without starting unrelated audio services.
    with (
        patch.object(TaskManager, "_TaskManager__setup_transcriber"),
        patch.object(TaskManager, "_TaskManager__setup_synthesizer"),
        patch.object(TaskManager, "_TaskManager__setup_llm"),
        patch.object(TaskManager, "_TaskManager__setup_tasks"),
        patch.object(TaskManager, "_TaskManager__setup_input_handlers"),
        patch.object(TaskManager, "_TaskManager__setup_output_handlers"),
        patch.object(TaskManager, "message_task_new", new_callable=AsyncMock),
    ):
        return TaskManager("example", 0, task, MagicMock(), turn_based_conversation=True).llm_config


class Socket:
    """A deterministic socket boundary; framing, locking and reset remain production code."""

    state = State.OPEN

    def __init__(self, events):
        self.events = iter(events)
        self.sent = []
        self.closed = False

    async def send(self, frame):
        self.sent.append(json.loads(frame))

    def __aiter__(self):
        return self

    async def __anext__(self):
        event = next(self.events, None)
        if event is None:
            raise StopAsyncIteration
        if isinstance(event, BaseException):
            raise event
        return json.dumps(event)

    async def close(self):
        self.closed = True


def terminal(status="completed", **response):
    return {
        "type": "response." + status,
        "response": {"id": "resp_example", "status": status, "model": "gpt-5", "service_tier": "default", **response},
    }


@pytest.fixture
async def native(monkeypatch):
    providers = []
    http_requests = []

    def reject_http(request):
        http_requests.append(request)
        raise AssertionError("No HTTP request belongs in this WebSocket test")

    client = httpx.AsyncClient(transport=httpx.MockTransport(reject_http))
    monkeypatch.setattr("bolna.llms.openai_llm.get_shared_http_client", lambda **kwargs: client)
    monkeypatch.setattr(OpenAIWSConnection, "start_connect", lambda self: None)

    def build(events=(), runtime=None, **options):
        config = native_config(**options)
        tm = MagicMock(task_config={"tools_config": {"llm_agent": {}}}, language="en", kwargs=runtime or {})
        llm = TaskManager._TaskManager__setup_llm(tm, config)
        providers.append(llm)
        socket = Socket(events)
        if llm._ws_transport:
            llm._ws_transport._ws = socket
            llm._ws_transport._connected_at = time.monotonic()
        return llm, socket

    yield build
    for llm in providers:
        await llm.close()
    await asyncio.sleep(0)  # Let abandoned-socket close tasks settle.
    await client.aclose()
    assert http_requests == []


@pytest.mark.parametrize(
    "options",
    [
        {"responses_history": "unknown"},
        {"responses_omit_parameters": ["input"]},
        {"responses_omit_parameters": ["store"]},
        {"responses_omit_parameters": "include"},
    ],
)
def test_schema_rejects_unsupported_state_and_omissions(options):
    with pytest.raises(ValidationError):
        SimpleLlmAgent(**options)


async def test_native_full_history_exact_ws_request_and_usage(native):
    events = [
        terminal("created"),
        {"type": "response.output_text.delta", "delta": "Hello."},
        terminal(usage=USAGE),
    ]
    llm, socket = native(events, **OPTIONS)
    llm.previous_response_id = "resp_do_not_chain"
    original_history, original_args = deepcopy(HISTORY), deepcopy(llm.model_args)
    meta = {}
    chunks = [chunk async for chunk in llm.generate_stream(HISTORY, synthesize=False, meta_info=meta)]
    assert socket.sent == [
        {
            "type": "response.create",
            "model": "gpt-5",
            "input": [{"type": "message", "role": m["role"], "content": m["content"]} for m in HISTORY],
            "store": False,
            "previous_response_id": None,
            "max_output_tokens": 4096,
            "service_tier": "default",
            "reasoning": {"effort": "medium"},
            "text": {"format": {"type": "text"}},
            "prompt_cache_key": "example-agent-v1",
        }
    ]
    assert HISTORY == original_history and llm.model_args == original_args
    assert llm.previous_response_id == "resp_example"  # Still available for cancellation, not chaining.
    assert chunks[-1].data == "Hello." and chunks[-1].end_of_stream
    assert (chunks[-1].input_tokens, chunks[-1].output_tokens, chunks[-1].cached_tokens) == (24, 8, 16)
    attempt = meta["responses_ws_attempts"][0]
    assert len(meta["responses_ws_attempts"]) == 1
    assert attempt["transport"] == "websocket" and attempt["status"] == "completed"
    assert attempt["usage"] == USAGE and attempt["response_id"] == "resp_example"
    assert attempt["model"] == "gpt-5" and attempt["service_tier"] == "default"
    assert 0 <= attempt["first_visible_text_ms"] <= attempt["duration_ms"]
    assert not llm._ws_transport._needs_reset


async def test_default_responses_request_retains_chaining_and_tuning(native):
    llm, _ = native()
    llm.previous_response_id = "resp_previous"
    params, _ = llm._build_responses_create_kwargs(HISTORY, None, False, None)
    assert params == {
        "model": "gpt-5",
        "input": [{"type": "message", "role": "user", "content": "Continue"}],
        "store": True,
        "truncation": "auto",
        "max_output_tokens": 4096,
        "temperature": 1,
        "service_tier": "default",
        "reasoning": {"effort": "medium"},
        "text": {"verbosity": "low"},
        "include": ["reasoning.encrypted_content"],
        "previous_response_id": "resp_previous",
    }


async def test_storage_and_history_are_independent_and_json_format_survives(native):
    llm, _ = native(responses_store=False)
    llm.previous_response_id = "resp_previous"
    params, _ = llm._build_responses_create_kwargs(HISTORY, None, False, None, store=True)
    assert params["store"] is False and params["previous_response_id"] == "resp_previous"
    assert params["input"] == [{"type": "message", "role": "user", "content": "Continue"}]
    llm, _ = native(responses_history="full", omit_request_parameters=["verbosity"])
    llm.previous_response_id = "resp_previous"
    params, _ = llm._build_responses_create_kwargs(HISTORY, None, True, None)
    assert params["store"] is True and params["previous_response_id"] is None
    assert params["text"] == {"format": {"type": "json_object"}}
    assert len(params["input"]) == len(HISTORY)


@pytest.mark.parametrize(
    "options",
    [
        {"use_responses_api": False},
        {"base_url": "https://example.com/v1"},
        {"provider": "custom"},
        {"provider": "azure"},
    ],
)
async def test_strict_mode_rejects_non_ws_configuration(native, options):
    with pytest.raises(ValueError, match="[Ww]eb[Ss]ocket|strict"):
        native(strict_websocket=True, **options)


async def test_strict_mode_rejects_http_dispatch_and_nonstream_generate(native):
    llm, _ = native(**OPTIONS)
    with pytest.raises(ValueError, match="[Ww]eb[Ss]ocket|strict"):
        await llm.generate(HISTORY)
    transport, llm._ws_transport = llm._ws_transport, None
    try:
        with pytest.raises(ValueError, match="[Ww]eb[Ss]ocket|strict"):
            _ = [chunk async for chunk in llm.generate_stream(HISTORY)]
    finally:
        llm._ws_transport = transport


@pytest.mark.parametrize(
    "events,status",
    [
        ([{"type": "error", "error": {"code": "previous_response_not_found", "message": "gone"}}], "error"),
        ([{"type": "error", "error": {"type": "invalid_request_error", "param": "input"}}], "error"),
        ([terminal("failed", error={"message": "failed"})], "failed"),
        ([terminal("incomplete", usage={}, incomplete_details={"reason": "max_output_tokens"})], "incomplete"),
        ([OSError("socket interrupted")], "error"),
        ([], "error"),
    ],
)
async def test_strict_failure_is_one_ws_attempt_without_replay_or_http(native, events, status):
    llm, socket = native(events, strict_websocket=True)
    llm.previous_response_id = "resp_previous"
    meta = {}
    with pytest.raises((APIError, OSError)):
        _ = [chunk async for chunk in llm.generate_stream(HISTORY, synthesize=False, meta_info=meta)]
    assert len(socket.sent) == 1
    assert len(meta["responses_ws_attempts"]) == 1
    attempt = meta["responses_ws_attempts"][0]
    assert attempt["status"] == status and attempt["transport"] == "websocket"
    assert attempt["usage"] == ({} if status == "incomplete" else None)
    assert attempt["first_visible_text_ms"] is None and attempt["duration_ms"] >= 0
    assert llm.previous_response_id is None


@pytest.mark.parametrize(
    "usage",
    [
        None,
        {},
        {
            "input_tokens": 0,
            "output_tokens": 0,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    ],
)
async def test_usage_presence_and_first_visible_text_are_not_inferred_from_other_events(native, usage, monkeypatch):
    events = [
        terminal("created"),
        {"type": "response.reasoning_summary_text.delta", "delta": "Thinking"},
        {"type": "response.output_text.delta", "delta": ""},
        {"type": "response.output_text.delta", "delta": "Visible"},
        terminal(**({"usage": usage} if usage is not None else {})),
    ]
    llm, _ = native(events, **OPTIONS)
    ticks = iter(range(100, 1000, 10))
    monkeypatch.setattr("bolna.llms.openai_llm.now_ms", lambda: next(ticks))
    meta = {}
    chunks = [chunk async for chunk in llm.generate_stream(HISTORY, synthesize=False, meta_info=meta)]
    attempt = meta["responses_ws_attempts"][0]
    assert attempt["usage"] == usage
    assert attempt["first_visible_text_ms"] == 40
    if usage:
        assert (chunks[-1].input_tokens, chunks[-1].cached_tokens, chunks[-1].reasoning_tokens) == (0, 0, 0)


async def test_persistent_full_history_tool_turn_does_not_chain_hidden_state(native):
    tool = {"type": "function", "function": {"name": "lookup", "parameters": {"type": "object", "properties": {}}}}
    events = [
        terminal("created"),
        {
            "type": "response.output_item.added",
            "item": {"type": "function_call", "id": "item_example", "call_id": "call_example", "name": "lookup"},
        },
        {"type": "response.function_call_arguments.delta", "item_id": "item_example", "delta": "{}"},
        terminal(usage={}),
        terminal("created"),
        {"type": "response.output_text.delta", "delta": "Found."},
        terminal(usage=USAGE),
    ]
    llm, socket = native(events, runtime={"api_tools": {"tools": [tool], "tools_params": {"lookup": {}}}}, **OPTIONS)
    meta = {}
    _ = [chunk async for chunk in llm.generate_stream(HISTORY, synthesize=False, meta_info=meta, tools=[tool])]
    assert meta["responses_ws_attempts"][0]["first_visible_text_ms"] is None
    messages = HISTORY + [
        {
            "role": "assistant",
            "tool_calls": [{"id": "call_example", **tool, "function": {"name": "lookup", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "call_example", "content": "Found item"},
    ]
    chunks = [chunk async for chunk in llm.generate_stream(messages, synthesize=False, meta_info=meta, tools=[tool])]
    assert len(socket.sent) == 2 and llm._ws_transport._ws is socket
    assert all(frame["previous_response_id"] is None and frame["store"] is False for frame in socket.sent)
    assert socket.sent[1]["input"] == socket.sent[0]["input"] + [
        {"type": "function_call", "call_id": "call_example", "name": "lookup", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "call_example", "output": "Found item"},
    ]
    assert socket.sent[1]["tools"] == [
        {
            "type": "function",
            "name": "lookup",
            "description": "",
            "parameters": {"type": "object", "properties": {}},
            "strict": False,
        }
    ]
    assert socket.sent[1]["parallel_tool_calls"] is False and socket.sent[1]["tool_choice"] == "auto"
    assert chunks[-1].data == "Found." and len(meta["responses_ws_attempts"]) == 2


async def test_full_history_still_cancels_the_in_flight_response(native):
    llm, socket = native(
        [terminal("created"), {"type": "response.output_text.delta", "delta": "Hello. "}, terminal()], **OPTIONS
    )
    llm.buffer_size = 1
    stream = llm.generate_stream(HISTORY, meta_info={})
    try:
        await stream.__anext__()
        assert llm.previous_response_id == "resp_example"
        llm.cancel_in_flight_response()
        await asyncio.sleep(0)
        assert socket.sent[-1] == {"type": "response.cancel", "response_id": "resp_example"}
        assert llm.previous_response_id is None
    finally:
        await stream.aclose()


async def test_cancelled_strict_attempt_retains_cancellation_and_socket_cleanup(native):
    llm, socket = native([terminal("created"), asyncio.CancelledError()], **OPTIONS)
    meta = {}
    with pytest.raises(asyncio.CancelledError):
        _ = [chunk async for chunk in llm.generate_stream(HISTORY, meta_info=meta)]
    await asyncio.sleep(0)
    assert len(socket.sent) == 1 and socket.closed
    assert meta["responses_ws_attempts"][0]["status"] == "cancelled"


async def test_default_socket_failure_still_falls_back_to_http(native, monkeypatch):
    llm, socket = native([OSError("socket interrupted")])
    fallback_calls = []

    async def fallback(*args, **kwargs):
        fallback_calls.append((args, kwargs))
        yield "http-fallback"

    monkeypatch.setattr(llm, "_generate_stream_responses", fallback)
    assert [chunk async for chunk in llm.generate_stream(HISTORY)] == ["http-fallback"]
    assert len(socket.sent) == len(fallback_calls) == 1
    assert llm.previous_response_id is None
