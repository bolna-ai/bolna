"""Simple-agent opt-ins survive native setup and real SDK streaming without changing defaults."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from pydantic import ValidationError

from bolna.agent_manager.task_manager import TaskManager
from bolna.llms.openai_llm import OpenAiLLM
from bolna.models import Llm, LlmAgent, SimpleLlmAgent


MESSAGES = [{"role": "user", "content": "Hello", "turn_id": 3}]
OPTIONS = {"prompt_cache_key": "example-agent-v1", "omit_request_parameters": ["temperature", "stop", "verbosity"]}
TOOLS = [{"type": "function", "function": {"name": "lookup_item", "parameters": {"type": "object", "properties": {}}}}]
USAGE = {
    "prompt_tokens": 24,
    "completion_tokens": 8,
    "total_tokens": 32,
    "prompt_tokens_details": {"cached_tokens": 16},
    "completion_tokens_details": {"reasoning_tokens": 3},
}


def native_config(**options):
    agent = LlmAgent.model_validate(
        {
            "agent_type": "simple_llm_agent",
            "agent_flow_type": "streaming",
            "llm_config": {"model": "gpt-5", "provider": "openai", **options},
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
    # Keep real config assembly while replacing unrelated audio/task startup.
    with (
        patch.object(TaskManager, "_TaskManager__setup_transcriber"),
        patch.object(TaskManager, "_TaskManager__setup_synthesizer"),
        patch.object(TaskManager, "_TaskManager__setup_llm"),
        patch.object(TaskManager, "_TaskManager__setup_tasks"),
    ):
        return TaskManager("example", 0, task, MagicMock(), turn_based_conversation=True)


@pytest.fixture
async def sdk(monkeypatch):
    clients = []

    def build(config, **runtime):
        requests = []

        def respond(request):
            body = json.loads(request.content)
            assert request.url.path == "/v1/chat/completions" and body["stream"] is True
            requests.append(body)
            delta = {"content": "Hello."}
            if body.get("tools"):
                delta = {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_example",
                            "type": "function",
                            "function": {"name": "lookup_item", "arguments": "{}"},
                        }
                    ]
                }
            chunk = {
                "id": "example",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": body["model"],
                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
            }
            usage = {**chunk, "choices": [], "usage": USAGE}
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content="".join("data: " + json.dumps(c) + "\n\n" for c in [chunk, usage]) + "data: [DONE]\n\n",
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        clients.append(client)
        monkeypatch.setattr("bolna.llms.openai_llm.get_shared_http_client", lambda **kw: client)
        # Invoke the actual constructor boundary, including stored/runtime precedence.
        tm = MagicMock(task_config={"tools_config": {"llm_agent": {}}}, language="en", kwargs=runtime)
        llm = TaskManager._TaskManager__setup_llm(tm, config)
        return llm, requests

    yield build
    for client in clients:
        await client.aclose()


@pytest.mark.parametrize(
    "field,value",
    [
        ("prompt_cache_key", ""),
        ("prompt_cache_key", 123),
        ("omit_request_parameters", ["messages"]),
        ("omit_request_parameters", ["max_tokens"]),
        ("omit_request_parameters", "temperature"),
    ],
)
def test_schema_rejects_invalid_opt_ins(field, value):
    with pytest.raises(ValidationError):
        SimpleLlmAgent(**{field: value})


@pytest.mark.parametrize("transport,expected", [(None, True), (False, False), (True, True)])
async def test_simple_transport_override_and_automatic_default(transport, expected):
    values = {} if transport is None else {"use_responses_api": transport}
    tm = native_config(model="gpt-5.4-mini", **values)
    assert tm.llm_agent_config["use_responses_api"] is transport
    assert tm.llm_config["use_responses_api"] is expected
    assert Llm().use_responses_api is False  # Other agent schemas retain their existing default.


async def test_native_simple_config_reaches_exact_sdk_stream(sdk):
    tm = native_config(max_tokens=4096, reasoning_effort="medium", temperature=0.2, use_responses_api=False, **OPTIONS)
    llm, requests = sdk(tm.llm_config)
    shared_args = llm.model_args.copy()
    chunks = [c async for c in llm.generate_stream(MESSAGES, synthesize=False)]
    assert requests == [
        {
            "model": "gpt-5",
            "max_completion_tokens": 4096,
            "reasoning_effort": "medium",
            "service_tier": "default",
            "prompt_cache_key": "example-agent-v1",
            "messages": [{"role": "user", "content": "Hello"}],
            "response_format": {"type": "text"},
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ]
    assert llm.model_args == shared_args
    assert not llm.use_responses_api and llm._ws_transport is None
    assert chunks[-1].data == "Hello." and chunks[-1].end_of_stream
    assert (
        chunks[-1].input_tokens,
        chunks[-1].output_tokens,
        chunks[-1].reasoning_tokens,
        chunks[-1].cached_tokens,
    ) == (24, 8, 3, 16)


async def test_absent_opt_ins_preserve_legacy_stream(sdk):
    llm, requests = sdk(native_config(model="gpt-4.1-mini", max_tokens=100, temperature=0.2).llm_config)
    _ = [c async for c in llm.generate_stream(MESSAGES, synthesize=False)]
    assert requests == [
        {
            "model": "gpt-4.1-mini",
            "max_tokens": 100,
            "temperature": 0.2,
            "service_tier": "default",
            "messages": [{"role": "user", "content": "Hello"}],
            "response_format": {"type": "text"},
            "stream": True,
            "stream_options": {"include_usage": True},
            "stop": ["User:"],
        }
    ]


async def test_streamed_tools_and_core_fields_survive_runtime_omissions(sdk):
    config = native_config(model="gpt-4.1-mini", max_tokens=100, **OPTIONS).llm_config
    # Runtime overrides also use the real factory, and cannot remove required request fields.
    llm, requests = sdk(
        config,
        prompt_cache_key="runtime-key",
        omit_request_parameters=["temperature", "stop", "tools", "messages", "max_tokens"],
        api_tools={"tools": TOOLS, "tools_params": {"lookup_item": {}}},
    )
    chunks = [c async for c in llm.generate_stream(MESSAGES, synthesize=False, meta_info={})]
    assert requests[0]["prompt_cache_key"] == "runtime-key"
    assert requests[0]["tools"] == TOOLS and requests[0]["tool_choice"] == "auto"
    assert requests[0]["parallel_tool_calls"] is False and requests[0]["max_tokens"] == 100
    assert requests[0]["messages"] == [{"role": "user", "content": "Hello"}]
    assert "temperature" not in requests[0] and "stop" not in requests[0]
    assert config["prompt_cache_key"] == "example-agent-v1"
    calls = [c for c in chunks if c.is_function_call]
    assert len(calls) == 1 and calls[0].data.called_fun == "lookup_item"
    assert calls[0].data.tool_call_id == "call_example"


async def test_custom_endpoint_guard_still_runs_before_stream(sdk, monkeypatch):
    guard = AsyncMock()
    monkeypatch.setattr("bolna.llms.openai_llm.guard_llm_base_url", guard)
    config = native_config(model="gpt-4.1-mini", **OPTIONS).llm_config
    llm, requests = sdk(config, provider="custom", base_url="https://example.com/v1")
    _ = [c async for c in llm.generate_stream(MESSAGES, synthesize=False)]
    guard.assert_awaited_once_with("https://example.com/v1")
    assert requests[0]["prompt_cache_key"] == "example-agent-v1"
