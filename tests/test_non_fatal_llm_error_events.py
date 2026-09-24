"""Every generation's non-fatal LLM errors reach progression_data, including tool and language-switch follow-ups."""

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import bolna.agent_manager.task_manager as task_manager_module
from bolna.agent_manager.task_manager import TaskManager
from bolna.exceptions import LLMError
from bolna.llms.azure_llm import AzureLLM
from bolna.llms.litellm import LiteLLM
from bolna.llms.openai_llm import OpenAiLLM
from bolna.llms.types import FunctionCallPayload, LatencyData, LLMStreamChunk

STALE_ID = {"error_type": "stale_response_id", "error": "previous response not found", "model": "gemma4"}
EMPTY = {"error_type": "empty_response", "error": None, "model": "gemma4"}


def make_tm():
    tm = TaskManager.__new__(TaskManager)
    tm.non_fatal_llm_error_events = []
    return tm


async def generate_via_wrapper(tm, meta_info):
    await TaskManager._TaskManager__do_llm_generation(tm, [], meta_info, "synthesizer")


# ------------------------------------------------------------------ collection per generation


async def test_a_tool_follow_up_keeps_its_empty_turn_event():
    tm = make_tm()
    outer_meta = {"sequence_id": 3, "turn_id": 3}

    async def impl(messages, meta_info, *args):
        if meta_info["sequence_id"] == 3:
            meta_info.setdefault("_non_fatal_errors", []).append(STALE_ID)
            # The real tool path hands the follow-up a copy: FunctionCallPayload -> model_dump -> **kwargs.
            copied = FunctionCallPayload(meta_info=meta_info).model_dump()["meta_info"]
            await generate_via_wrapper(tm, {**copied, "sequence_id": 4, "turn_id": 6})
        else:
            meta_info.setdefault("_non_fatal_errors", []).append(EMPTY)

    tm._TaskManager__do_llm_generation_impl = impl
    await generate_via_wrapper(tm, outer_meta)

    assert EMPTY not in outer_meta["_non_fatal_errors"]  # why draining only the turn's own list lost it
    assert tm.non_fatal_llm_error_events == [
        {**EMPTY, "sequence_id": 4, "turn_id": 6},
        {**STALE_ID, "sequence_id": 3, "turn_id": 3},
    ]


async def test_a_language_switch_follow_up_records_its_own_events():
    tm = make_tm()
    tm._append_eager_llm_stub = MagicMock()
    tm.interruption_manager = MagicMock()

    async def impl(messages, meta_info, *args):
        meta_info.setdefault("_non_fatal_errors", []).append(EMPTY)

    tm._TaskManager__do_llm_generation_impl = impl
    await TaskManager._TaskManager__generate_switch_followup(tm, [], {"sequence_id": 12, "turn_id": 15}, "synthesizer")

    assert tm.non_fatal_llm_error_events == [{**EMPTY, "sequence_id": 12, "turn_id": 15}]


async def test_errors_recorded_before_a_generation_fails_are_kept():
    tm = make_tm()

    async def impl(messages, meta_info, *args):
        meta_info.setdefault("_non_fatal_errors", []).append(STALE_ID)
        raise LLMError("boom", provider="openai", model="gemma4")

    tm._TaskManager__do_llm_generation_impl = impl
    with pytest.raises(LLMError):
        await generate_via_wrapper(tm, {"sequence_id": 7, "turn_id": 9})

    assert tm.non_fatal_llm_error_events == [{**STALE_ID, "sequence_id": 7, "turn_id": 9}]


async def test_errors_recorded_before_a_timeout_are_kept(monkeypatch):
    monkeypatch.setattr(task_manager_module, "LLM_GENERATION_TIMEOUT_S", 0.05)
    tm = make_tm()

    async def impl(messages, meta_info, *args):
        meta_info.setdefault("_non_fatal_errors", []).append(STALE_ID)
        await asyncio.Event().wait()

    tm._TaskManager__do_llm_generation_impl = impl
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(generate_via_wrapper(tm, {"sequence_id": 7, "turn_id": 9}), timeout=1.0)

    assert tm.non_fatal_llm_error_events == [{**STALE_ID, "sequence_id": 7, "turn_id": 9}]


async def test_a_generation_never_inherits_its_parents_errors_or_finish_reason():
    tm = make_tm()
    seen = {}

    async def impl(messages, meta_info, *args):
        seen["errors"] = list(meta_info["_non_fatal_errors"])
        seen["finish_reason"] = meta_info.get("llm_finish_reason", "absent")

    tm._TaskManager__do_llm_generation_impl = impl
    await generate_via_wrapper(
        tm, {"sequence_id": 4, "_non_fatal_errors": [STALE_ID], "llm_finish_reason": "tool_calls"}
    )

    assert seen == {"errors": [], "finish_reason": "absent"}
    assert tm.non_fatal_llm_error_events == []


# ------------------------------------------------------------------ empty turn names the finish reason


async def test_an_empty_turn_names_the_finish_reason():
    tm = TaskManager.__new__(TaskManager)
    tm.hangup_triggered = False
    tm.conversation_ended = False
    tm.stream = True
    tm.turn_based_conversation = False
    tm.language = "en"
    tm.llm_config = {"model": "gemma4"}
    tm.llm_latencies = SimpleNamespace(turn_latencies=[])
    tm.tools = {"input": MagicMock(reset_response_heard_by_user=MagicMock())}
    tm._stamp_llm_latency_dict = MagicMock()
    tm._inject_language_instruction = lambda messages: messages
    tm._handle_llm_output = AsyncMock()
    tm._TaskManager__process_stop_words = lambda data, meta_info: data
    tm._TaskManager__store_into_history = MagicMock()

    async def generate(messages, synthesize, meta_info):
        meta_info["llm_finish_reason"] = "length"  # what a chat-completions client stashes at stream end
        yield LLMStreamChunk(
            data="", end_of_stream=True, latency=LatencyData(sequence_id=4, first_token_latency_ms=820.0)
        )

    tm.tools["llm_agent"] = MagicMock(generate=generate)
    meta_info = {"llm_start_time": time.time(), "sequence_id": 4, "turn_id": 6, "_non_fatal_errors": []}
    await TaskManager._TaskManager__do_llm_generation_impl(
        tm, [{"role": "user", "content": "hi"}], meta_info, "synthesizer"
    )

    assert meta_info["_non_fatal_errors"] == [{"error_type": "empty_response", "error": "length", "model": "gemma4"}]
    assert tm._TaskManager__store_into_history.call_args.kwargs["log_message"] == "LLM returned no output (length)"


# ------------------------------------------------------------------ chat-completions clients stash it


def chat_chunk(finish_reason=None):
    return SimpleNamespace(
        id="c1",
        usage=None,
        service_tier=None,
        choices=[SimpleNamespace(delta=SimpleNamespace(content=None, tool_calls=None), finish_reason=finish_reason)],
    )


async def stream_of(chunks):
    for chunk in chunks:
        yield chunk


def stub_chat_llm():
    llm = MagicMock()
    llm.model = "gemma4"
    llm.model_args = {}
    llm.trigger_function_call = False
    llm.buffer_size = 40
    llm.llm_host = None
    llm.request_log_model = "gemma4"
    llm._find_text_tool_call_start = MagicMock(return_value=-1)
    return llm


async def drain(agen):
    return [chunk async for chunk in agen]


async def test_openai_chat_stream_stashes_the_finish_reason():
    llm = stub_chat_llm()
    llm.async_client.chat.completions.create = AsyncMock(return_value=stream_of([chat_chunk(), chat_chunk("length")]))
    meta_info = {"sequence_id": 4}
    await drain(OpenAiLLM._generate_stream_chat(llm, [{"role": "user", "content": "hi"}], meta_info=meta_info))
    assert meta_info["llm_finish_reason"] == "length"


async def test_azure_chat_stream_stashes_the_finish_reason():
    llm = stub_chat_llm()
    llm._create_completion = AsyncMock(return_value=(stream_of([chat_chunk(), chat_chunk("length")]), False))
    meta_info = {"sequence_id": 4}
    await drain(AzureLLM._generate_stream_chat(llm, [{"role": "user", "content": "hi"}], meta_info=meta_info))
    assert meta_info["llm_finish_reason"] == "length"


async def test_litellm_stream_stashes_the_finish_reason():
    llm = stub_chat_llm()
    chunks = [
        {"choices": [{"delta": SimpleNamespace(content=None, tool_calls=None), "finish_reason": None}]},
        {"choices": [{"delta": SimpleNamespace(content=None, tool_calls=None), "finish_reason": "length"}]},
    ]
    meta_info = {"sequence_id": 4}
    with patch("bolna.llms.litellm.acompletion", AsyncMock(return_value=stream_of(chunks))):
        await drain(LiteLLM.generate_stream(llm, [{"role": "user", "content": "hi"}], meta_info=meta_info))
    assert meta_info["llm_finish_reason"] == "length"


# ------------------------------------------------------------------ fatal error reaches progression_data


async def test_the_error_that_ends_the_call_is_kept_for_progression_data():
    tm = TaskManager.__new__(TaskManager)
    tm._component_error = None
    tm.component_error_event = None
    tm._report_provider_health = AsyncMock()
    tm.run_id = None
    tm.conversation_ended = True
    tm._end_of_conversation_in_progress = False
    tm.conversation_start_init_ts = time.time() * 1000 - 5000

    await tm._end_call_on_component_error(LLMError("timed out", provider="openai", model="gemma4"), "llm_error")
    await tm._end_call_on_component_error(LLMError("second", provider="openai", model="gemma4"), "llm_error")

    event = tm.component_error_event
    assert {k: v for k, v in event.items() if k != "ts_ms"} == {
        "component": "llm",
        "provider": "openai",
        "model": "gemma4",
        "error": "timed out",
    }
    assert 4900 < event["ts_ms"] < 6000
