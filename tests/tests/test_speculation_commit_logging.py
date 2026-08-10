"""Tests for committed-speculation LLM logging.

A committed speculative follow-up is a real main-LLM call whose reply the caller hears,
so it must appear in the execution logs exactly like a normal turn: an LLM request row,
an LLM response row carrying api-reported token usage, and a turn-latency entry.
Discarded speculations must log nothing — their capture is dropped.
"""

import types
from unittest.mock import MagicMock, patch

import pytest

from bolna.agent_manager.task_manager import TaskManager
from bolna.helpers.conversation_history import ConversationHistory


def _msg(data="", end=False, fc=False, **extra):
    return types.SimpleNamespace(data=data, end_of_stream=end, is_function_call=fc, **extra)


def _make_tm(history, generate):
    tm = MagicMock()
    tm.conversation_history = history
    tm.multilingual_prompts = {"en": "You are a helpful agent.", "te": "Telugu prompt"}
    tm._TaskManager__switch_context_note = TaskManager._TaskManager__switch_context_note.__get__(tm, TaskManager)
    tm.tools = {"llm_agent": MagicMock()}
    tm.tools["llm_agent"].generate = generate
    tm.lid_spec_capture = None
    return tm


def _spec(tm):
    return TaskManager._TaskManager__speculative_followup_text.__get__(tm, TaskManager)


def _log_commit(tm):
    return TaskManager._TaskManager__log_committed_speculation.__get__(tm, TaskManager)


def _usage_generate(input_tokens=120, output_tokens=30, cached_tokens=100, latency=None):
    async def generate(messages, synthesize=False, meta_info=None):
        yield _msg(data="telugu reply", end=False)
        yield _msg(
            data="",
            end=True,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=None,
            cached_tokens=cached_tokens,
            latency=latency,
        )

    return generate


@pytest.mark.asyncio
async def test_completed_speculation_captures_request_and_usage():
    h = ConversationHistory(initial_history=[{"role": "system", "content": "base"}])
    h.append_user("తెలుగులో మాట్లాడదామా?")
    tm = _make_tm(h, _usage_generate())

    text = await _spec(tm)("te", "తెలుగులో మాట్లాడదామా?", "", "తెలుగులో మాట్లాడదామా?")

    assert text == "telugu reply"
    capture = tm.lid_spec_capture
    assert capture is not None
    assert capture["input_tokens"] == 120
    assert capture["output_tokens"] == 30
    assert capture["cached_tokens"] == 100
    assert capture["meta_info"]["origin"] == "language_switch_speculation"
    assert capture["meta_info"]["llm_start_time"] is not None
    assert "తెలుగులో మాట్లాడదామా?" in capture["request_message"]


@pytest.mark.asyncio
async def test_function_call_abort_leaves_no_capture():
    async def generate(messages, synthesize=False, meta_info=None):
        yield _msg(data="", fc=True)

    h = ConversationHistory(initial_history=[{"role": "system", "content": "base"}])
    tm = _make_tm(h, generate)

    text = await _spec(tm)("te", "detector text")

    assert text == ""
    assert tm.lid_spec_capture is None


def test_commit_emits_request_and_response_rows_with_usage():
    tm = MagicMock()
    tm.run_id = "run-1"
    tm.llm_config = {"model": "gpt-4.1-mini"}
    tm.lid_spec_capture = {
        "meta_info": {"request_id": "req-1", "sequence_id": -1, "turn_id": None, "llm_start_time": 123.0},
        "request_message": "formatted request",
        "input_tokens": 120,
        "output_tokens": 30,
        "reasoning_tokens": None,
        "cached_tokens": 100,
        "latency": None,
    }

    with patch("bolna.agent_manager.task_manager.convert_to_request_log") as log_mock:
        _log_commit(tm)("telugu reply")

    assert log_mock.call_count == 2
    request_call, response_call = log_mock.call_args_list
    assert request_call.kwargs["direction"].value == "request"
    assert request_call.kwargs["message"] == "formatted request"
    assert request_call.kwargs["model"] == "gpt-4.1-mini"
    assert response_call.kwargs["direction"].value == "response"
    assert response_call.kwargs["message"] == "telugu reply"
    assert response_call.kwargs["input_tokens"] == 120
    assert response_call.kwargs["output_tokens"] == 30
    assert response_call.kwargs["cached_tokens"] == 100
    # Capture consumed — a later switch can never re-log it.
    assert tm.lid_spec_capture is None


def test_commit_appends_latency_entry_when_present():
    latency = MagicMock()
    latency.model_dump.return_value = {"sequence_id": -1, "first_token_latency_ms": 90.0}
    tm = MagicMock()
    tm.run_id = "run-1"
    tm.llm_config = {"model": "gpt-4.1-mini"}
    tm.llm_latencies.turn_latencies = []
    tm.lid_spec_capture = {
        "meta_info": {"request_id": "req-1", "sequence_id": -1, "turn_id": None, "llm_start_time": 123.0},
        "request_message": "formatted request",
        "input_tokens": 120,
        "output_tokens": 30,
        "reasoning_tokens": None,
        "cached_tokens": 100,
        "latency": latency,
    }

    with patch("bolna.agent_manager.task_manager.convert_to_request_log"):
        _log_commit(tm)("telugu reply")

    tm._stamp_llm_latency_dict.assert_called_once()
    assert len(tm.llm_latencies.turn_latencies) == 1
    assert tm.llm_latencies.turn_latencies[0]["first_token_latency_ms"] == 90.0


def test_commit_without_capture_is_noop():
    tm = MagicMock()
    tm.lid_spec_capture = None

    with patch("bolna.agent_manager.task_manager.convert_to_request_log") as log_mock:
        _log_commit(tm)("anything")

    log_mock.assert_not_called()
