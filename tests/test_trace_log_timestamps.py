"""Trace rows must be stamped when the event happened, not when the row is written.

Two rows are written after the fact: the LLM response (logged only once every chunk has
been pushed to TTS) and the graph_routing request (logged only after the hop resolved).
Without an explicit `ts` they land late enough that synthesizer rows sort in front of the
LLM response, which is what makes a graph-agent trace read as non-sequential.
"""

import asyncio
import time
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import bolna.helpers.utils as utils
from bolna.agent_manager.task_manager import TaskManager
from bolna.enums import LogComponent, LogDirection
from bolna.llms.types import LatencyData, LLMStreamChunk


@pytest.fixture
def captured(monkeypatch):
    """Capture the log dicts convert_to_request_log would have written."""
    rows = []

    async def fake_write(log, run_id):
        rows.append(log)

    monkeypatch.setattr(utils, "write_request_logs", fake_write)
    return rows


async def _drain():
    # convert_to_request_log dispatches via create_task
    await asyncio.sleep(0)


async def test_explicit_ts_is_used_for_the_row_time(captured):
    event_ts = datetime(2026, 8, 19, 13, 56, 45, 689336).timestamp()
    utils.convert_to_request_log(
        "response text",
        {"request_id": "leg", "sequence_id": 2},
        "gpt-4.1-mini",
        LogComponent.LLM,
        direction=LogDirection.RESPONSE,
        ts=event_ts,
    )
    await _drain()

    assert captured[0]["time"] == "2026-08-19 13:56:45.689336"


async def test_omitting_ts_falls_back_to_now(captured):
    before = datetime.now()
    utils.convert_to_request_log(
        "response text",
        {"request_id": "leg", "sequence_id": 2},
        "gpt-4.1-mini",
        LogComponent.LLM,
        direction=LogDirection.RESPONSE,
    )
    await _drain()

    stamped = datetime.strptime(captured[0]["time"], "%Y-%m-%d %H:%M:%S.%f")
    assert before <= stamped <= datetime.now()


async def test_explicit_latency_wins_over_meta_info(captured):
    # meta_info["llm_latency"] is never set anywhere in bolna, so the column stayed empty
    # on every LLM row until callers could pass the value directly.
    utils.convert_to_request_log(
        "response text",
        {"request_id": "leg", "sequence_id": 2, "llm_latency": 9.9},
        "gpt-4.1-mini",
        LogComponent.LLM,
        direction=LogDirection.RESPONSE,
        latency=0.833,
    )
    await _drain()

    assert captured[0]["latency"] == 0.833


async def test_zero_latency_is_kept_not_dropped(captured):
    # A deterministic routing hop really does take ~0s; it must not read as "no data".
    utils.convert_to_request_log(
        "Node: a -> b",
        {"request_id": "leg", "sequence_id": 1},
        "deterministic",
        LogComponent.GRAPH_ROUTING,
        direction=LogDirection.RESPONSE,
        latency=0.0,
    )
    await _drain()

    assert captured[0]["latency"] == 0.0


async def test_routing_request_row_precedes_its_response_row(captured):
    """The regression: both graph_routing rows are emitted after the hop finished, so the
    request row must be stamped from routing_started_at or it sorts on top of the response."""
    hop_started_at = datetime(2026, 8, 19, 13, 56, 44, 992414).timestamp()
    meta_info = {"request_id": "leg", "sequence_id": 2}

    utils.convert_to_request_log(
        "routing prompt",
        meta_info,
        "gpt-4.1-mini",
        LogComponent.GRAPH_ROUTING,
        direction=LogDirection.REQUEST,
        ts=hop_started_at,
    )
    utils.convert_to_request_log(
        "Node: a -> b",
        meta_info,
        "gpt-4.1-mini",
        LogComponent.GRAPH_ROUTING,
        direction=LogDirection.RESPONSE,
        latency=0.6918,
    )
    await _drain()

    request_row, response_row = captured
    assert request_row["time"] < response_row["time"]
    assert response_row["latency"] == 0.6918


def streaming_task_manager(generate):
    tm = TaskManager.__new__(TaskManager)
    tm.hangup_triggered = False
    tm.conversation_ended = False
    tm.stream = True
    tm.turn_based_conversation = False
    tm.language = "en"
    tm.llm_config = {"model": "gpt-4.1-mini"}
    tm.llm_latencies = SimpleNamespace(turn_latencies=[])
    tm.tools = {"input": MagicMock(reset_response_heard_by_user=MagicMock()), "llm_agent": MagicMock(generate=generate)}
    tm._stamp_llm_latency_dict = MagicMock()
    tm._inject_language_instruction = lambda messages: messages
    tm._handle_llm_output = AsyncMock()
    tm._TaskManager__process_stop_words = lambda data, meta_info: data
    tm._TaskManager__store_into_history = MagicMock()
    return tm


async def run_generation(tm):
    meta_info = {"llm_start_time": time.time(), "sequence_id": 4, "turn_id": 6, "_non_fatal_errors": []}
    await TaskManager._TaskManager__do_llm_generation_impl(
        tm, [{"role": "user", "content": "hi"}], meta_info, "synthesizer"
    )
    return tm._TaskManager__store_into_history.call_args.kwargs["ts"]


async def test_llm_response_row_is_stamped_when_the_first_text_arrived():
    """TTS starts on the first sentence, so a stream-end stamp sorts the LLM response below
    the synthesizer rows of its own reply."""
    sent = {}

    async def generate(messages, synthesize, meta_info):
        latency = LatencyData(sequence_id=4, first_token_latency_ms=400.0)
        sent["first"] = time.time()
        yield LLMStreamChunk(data="Your refund is approved.", end_of_stream=False, latency=latency)
        await asyncio.sleep(0.05)
        sent["last"] = time.time()
        yield LLMStreamChunk(data="It reaches your account in three days.", end_of_stream=True, latency=latency)

    ts = await run_generation(streaming_task_manager(generate))

    assert sent["first"] <= ts < sent["last"]


async def test_a_reply_with_no_text_keeps_the_stream_end_stamp():
    sent = {}

    async def generate(messages, synthesize, meta_info):
        sent["end"] = time.time()
        yield LLMStreamChunk(
            data="", end_of_stream=True, latency=LatencyData(sequence_id=4, first_token_latency_ms=820.0)
        )

    ts = await run_generation(streaming_task_manager(generate))

    assert ts is not None and ts >= sent["end"]
