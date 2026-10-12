"""A tool call naming no configured tool sends at most one end-of-stream and is recorded as a non-fatal LLM error."""

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

from bolna.agent_manager.task_manager import TaskManager
from bolna.llms.openai_llm import OpenAiLLM

SPOKEN = "Sure, please stay on the line, I'm transferring your call to our team."
TOOL = {"type": "function", "function": {"name": "transfer_call_support", "parameters": {"properties": {}}}}


def _chunk(content=None, tool_calls=None, finish_reason=None):
    return SimpleNamespace(
        id="c1",
        usage=None,
        service_tier=None,
        choices=[
            SimpleNamespace(delta=SimpleNamespace(content=content, tool_calls=tool_calls), finish_reason=finish_reason)
        ],
    )


async def _stream(*chunks):
    for chunk in chunks:
        yield chunk


def _llm_streaming(tool_name, spoken):
    """A chat-completions client whose model optionally speaks, then calls ``tool_name``."""
    llm = MagicMock()
    llm.model = "gemma4"
    llm.model_args = {}
    llm.trigger_function_call = True
    llm.api_params = {"transfer_call_support": {"url": None, "method": "POST"}}
    llm.language = "en"
    llm.run_id = "run-1"
    llm.buffer_size = 400
    llm.llm_host = None
    llm.request_log_model = "gemma4"
    llm._parse_tools = MagicMock(return_value=[TOOL])
    llm._find_text_tool_call_start = MagicMock(return_value=-1)
    call = SimpleNamespace(index=0, id="call-1", function=SimpleNamespace(name=tool_name, arguments="{}"))
    chunks = ([_chunk(spoken)] if spoken else []) + [_chunk(tool_calls=[call]), _chunk(finish_reason="tool_calls")]
    llm.async_client.chat.completions.create = MagicMock(side_effect=lambda **_: _async_value(_stream(*chunks)))
    return llm


async def _async_value(value):
    return value


def _task_manager(llm):
    tm = TaskManager.__new__(TaskManager)
    tm.hangup_triggered = False
    tm.conversation_ended = False
    tm.stream = True
    tm.turn_based_conversation = False
    tm.language = "en"
    tm.llm_config = {"model": "gemma4"}
    tm.llm_latencies = SimpleNamespace(turn_latencies=[])
    tm._stamp_llm_latency_dict = MagicMock()
    tm._inject_language_instruction = lambda messages: messages
    tm._TaskManager__process_stop_words = lambda data, meta_info: data
    tm._TaskManager__store_into_history = MagicMock()
    tm._turn_audio_flushed = asyncio.Event()
    tm._turn_audio_flushed.set()
    tm._turn_eos_forwarded = False
    tm.synthesizer_tasks = []
    tm.forwarded = []

    async def _synthesize(packet):
        tm.forwarded.append(packet)

    async def generate(messages, synthesize=True, meta_info=None, **kwargs):
        async for chunk in OpenAiLLM._generate_stream_chat(llm, messages, synthesize=synthesize, meta_info=meta_info):
            yield chunk

    tm._synthesize = _synthesize
    tm.tools = {"input": MagicMock(), "llm_agent": MagicMock(generate=generate)}
    return tm


async def _run_turn(tool_name, spoken=SPOKEN):
    tm = _task_manager(_llm_streaming(tool_name, spoken))
    meta_info = {"llm_start_time": time.time(), "sequence_id": 3, "turn_id": 3, "_non_fatal_errors": []}
    await TaskManager._TaskManager__do_llm_generation_impl(
        tm, [{"role": "user", "content": "when will I get my order"}], meta_info, "synthesizer"
    )
    await asyncio.gather(*tm.synthesizer_tasks)
    return tm, meta_info


async def test_the_spoken_text_ends_the_synthesizer_stream_exactly_once():
    tm, _ = await _run_turn("custom_task_transfer_call")
    assert [p["data"] for p in tm.forwarded] == [SPOKEN]
    assert sum(1 for p in tm.forwarded if p["meta_info"].get("end_of_llm_stream")) == 1


async def test_the_dropped_tool_call_is_recorded_as_a_non_fatal_error():
    _, meta_info = await _run_turn("custom_task_transfer_call")
    assert {"error_type": "unresolved_tool_call", "error": "custom_task_transfer_call", "model": "gemma4"} in (
        meta_info["_non_fatal_errors"]
    )


async def test_a_tool_call_with_no_spoken_text_sends_nothing_to_the_synthesizer():
    tm, meta_info = await _run_turn("custom_task_send_whatsapp", spoken=None)
    assert tm.forwarded == []
    assert {"error_type": "unresolved_tool_call", "error": "custom_task_send_whatsapp", "model": "gemma4"} in (
        meta_info["_non_fatal_errors"]
    )
