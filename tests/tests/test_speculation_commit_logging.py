"""Tests for committed/discarded speculative follow-up LLM logging.

Committed speculation logs like a normal turn (request/response rows with tokens under
a real sequence id, latency entry with origin, on_turn_usage tally); the capture rides
the spec task's return value so overlapping handlers can't clear it. Discarded-but-
completed speculations log their spend under LLM_LANGUAGE_SWITCH (visible, never billed).
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
    return tm


def _spec(tm):
    return TaskManager._TaskManager__speculative_followup_text.__get__(tm, TaskManager)


def _log_commit(tm):
    return TaskManager._TaskManager__log_committed_speculation.__get__(tm, TaskManager)


def _log_discard(tm):
    return TaskManager._TaskManager__log_discarded_speculation.__get__(tm, TaskManager)


def _capture(**overrides):
    base = {
        "meta_info": {
            "request_id": "req-1",
            "sequence_id": -1,
            "turn_id": None,
            "origin": "language_switch_speculation",
            "llm_start_time": 123.0,
        },
        "request_message": "formatted request",
        "input_tokens": 120,
        "output_tokens": 30,
        "reasoning_tokens": None,
        "cached_tokens": 100,
        "overflowed": False,
        "latency": None,
    }
    base.update(overrides)
    return base


def _commit_tm():
    tm = MagicMock()
    tm.run_id = "run-1"
    tm.task_id = 0
    tm.llm_config = {"model": "gpt-4.1-mini"}
    tm.interruption_manager.get_next_sequence_id.return_value = 4
    tm._usage_tasks = set()
    return tm


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
async def test_completed_speculation_returns_text_and_capture():
    h = ConversationHistory(initial_history=[{"role": "system", "content": "base"}])
    h.append_user("తెలుగులో మాట్లాడదామా?")
    tm = _make_tm(h, _usage_generate())

    text, capture = await _spec(tm)("te", "తెలుగులో మాట్లాడదామా?", "", "తెలుగులో మాట్లాడదామా?")

    assert text == "telugu reply"
    assert capture is not None
    assert capture["input_tokens"] == 120
    assert capture["output_tokens"] == 30
    assert capture["cached_tokens"] == 100
    assert capture["meta_info"]["origin"] == "language_switch_speculation"
    assert capture["meta_info"]["llm_start_time"] is not None
    assert "తెలుగులో మాట్లాడదామా?" in capture["request_message"]


@pytest.mark.asyncio
async def test_function_call_abort_returns_no_capture():
    async def generate(messages, synthesize=False, meta_info=None):
        yield _msg(data="", fc=True)

    h = ConversationHistory(initial_history=[{"role": "system", "content": "base"}])
    tm = _make_tm(h, generate)

    text, capture = await _spec(tm)("te", "detector text")

    assert text == ""
    assert capture is None


def test_commit_emits_rows_with_real_sequence_id_and_usage():
    tm = _commit_tm()

    with patch("bolna.agent_manager.task_manager.convert_to_request_log") as log_mock:
        _log_commit(tm)("telugu reply", _capture())

    assert log_mock.call_count == 2
    request_call, response_call = log_mock.call_args_list
    assert request_call.kwargs["direction"].value == "request"
    assert request_call.kwargs["message"] == "formatted request"
    assert request_call.kwargs["model"] == "gpt-4.1-mini"
    # Log rows carry the minted REAL sequence id, retired immediately so it never
    # counts as pending; the synth meta elsewhere keeps -1.
    assert request_call.kwargs["meta_info"]["sequence_id"] == 4
    tm.interruption_manager.retire_sequence_id.assert_called_once_with(4)
    assert response_call.kwargs["direction"].value == "response"
    assert response_call.kwargs["message"] == "telugu reply"
    assert response_call.kwargs["meta_info"]["sequence_id"] == 4
    assert response_call.kwargs["input_tokens"] == 120
    assert response_call.kwargs["output_tokens"] == 30
    assert response_call.kwargs["cached_tokens"] == 100


@pytest.mark.asyncio
async def test_commit_reports_on_turn_usage():
    tm = _commit_tm()
    reported = []

    async def on_turn_usage(input_tokens, output_tokens, cached_tokens):
        reported.append((input_tokens, output_tokens, cached_tokens))

    tm.on_turn_usage = on_turn_usage

    with patch("bolna.agent_manager.task_manager.convert_to_request_log"):
        _log_commit(tm)("telugu reply", _capture())
        for task in list(tm._usage_tasks):
            await task

    assert reported == [(120, 30, 100)]


def test_commit_latency_entry_gets_real_seq_and_origin():
    latency = MagicMock()
    latency.model_dump.return_value = {"sequence_id": -1, "first_token_latency_ms": 90.0}
    tm = _commit_tm()
    tm.on_turn_usage = None
    tm.llm_latencies.turn_latencies = []

    with patch("bolna.agent_manager.task_manager.convert_to_request_log"):
        _log_commit(tm)("telugu reply", _capture(latency=latency))

    tm._stamp_llm_latency_dict.assert_called_once()
    assert len(tm.llm_latencies.turn_latencies) == 1
    entry = tm.llm_latencies.turn_latencies[0]
    assert entry["sequence_id"] == 4
    assert entry["origin"] == "language_switch_speculation"


def test_commit_without_capture_is_noop():
    tm = _commit_tm()

    with patch("bolna.agent_manager.task_manager.convert_to_request_log") as log_mock:
        _log_commit(tm)("anything", None)

    log_mock.assert_not_called()


def test_discard_logs_under_language_switch_component():
    tm = _commit_tm()

    with patch("bolna.agent_manager.task_manager.convert_to_request_log") as log_mock:
        _log_discard(tm)("unheard reply", _capture())

    assert log_mock.call_count == 2
    request_call, response_call = log_mock.call_args_list
    assert request_call.kwargs["component"].value == "llm_language_switch"
    assert response_call.kwargs["component"].value == "llm_language_switch"
    assert response_call.kwargs["message"] == "unheard reply"
    assert response_call.kwargs["input_tokens"] == 120
    assert response_call.kwargs["output_tokens"] == 30


def test_discard_without_capture_is_noop():
    tm = _commit_tm()

    with patch("bolna.agent_manager.task_manager.convert_to_request_log") as log_mock:
        _log_discard(tm)("", None)

    log_mock.assert_not_called()
