"""Trace rows must be stamped when the event happened, not when the row is written.

Two rows are written after the fact: the LLM response (logged only once every chunk has
been pushed to TTS) and the graph_routing request (logged only after the hop resolved).
Without an explicit `ts` they land late enough that synthesizer rows sort in front of the
LLM response, which is what makes a graph-agent trace read as non-sequential.
"""

import asyncio
from datetime import datetime

import pytest

import bolna.helpers.utils as utils
from bolna.enums import LogComponent, LogDirection


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
