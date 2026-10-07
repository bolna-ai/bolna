"""Jev veto on eager-end-of-turn speculation: rules, config, fail-open, and the TaskManager flow."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from pydantic import ValidationError

from bolna.agent_manager.interruption_manager import InterruptionManager
from bolna.agent_manager.task_manager import TaskManager
from bolna.helpers import speculation_gate as sg
from bolna.helpers.speculation_gate import GateVerdict, SpeculationGate, evaluate_rules
from bolna.models import ConversationConfig


# --- rules ---------------------------------------------------------------------------------


def test_a_noul_below_its_min_fails():
    assert evaluate_rules({"complete": {"min": 0.5}}, {"complete": {"type": "noul", "noul": 0.2}}) == ["complete"]
    assert evaluate_rules({"complete": {"min": 0.5}}, {"complete": {"type": "noul", "noul": 0.8}}) == []


def test_a_choice_checks_its_labels_and_confidence():
    answer = {"type": "choice", "choice": "undetermined", "confidence": 0.9}
    assert evaluate_rules({"intent": {"deny": ["undetermined"]}}, {"intent": answer}) == ["intent"]
    answer = {"type": "choice", "choice": "book", "confidence": 0.6}
    assert evaluate_rules({"intent": {"allow": ["book"]}}, {"intent": answer}) == []
    assert evaluate_rules({"intent": {"allow": ["book"], "min": 0.7}}, {"intent": answer}) == ["intent"]


def test_a_score_checks_its_range():
    answer = {"type": "score", "score": 1.4, "confidence": 0.5}
    assert evaluate_rules({"s": {"min": 1.0, "max": 2.0}}, {"s": answer}) == []
    assert evaluate_rules({"s": {"min": 1.5}}, {"s": answer}) == ["s"]


def test_a_missing_answer_fails_and_unset_bounds_are_ignored():
    assert evaluate_rules({"complete": {"min": 0.5}}, {}) == ["complete"]
    rule = {"min": 0.5, "max": None, "allow": None, "deny": None}
    assert evaluate_rules({"complete": rule}, {"complete": {"type": "noul", "noul": 0.9}}) == []


# --- config --------------------------------------------------------------------------------


def test_the_default_gate_needs_only_an_empty_block():
    config = ConversationConfig(speculation_gate={})
    assert config.speculation_gate.provider == "typesafe"


def test_custom_questions_need_rules():
    with pytest.raises(ValidationError):
        ConversationConfig(speculation_gate={"questions": {"done": {"type": "noul", "instructions": "Done?"}}})


def test_rules_must_name_an_asked_question():
    with pytest.raises(ValidationError):
        ConversationConfig(speculation_gate={"rules": {"intent": {"min": 0.5}}})


def test_a_choice_question_needs_labels():
    with pytest.raises(ValidationError):
        ConversationConfig(
            speculation_gate={"questions": {"intent": {"type": "choice"}}, "rules": {"intent": {"min": 0.5}}}
        )


def test_custom_questions_and_rules_validate():
    config = ConversationConfig(
        speculation_gate={
            "questions": {
                "intent": {
                    "type": "choice",
                    "instructions": "What does the caller want?",
                    "criteria": {"book": None, "cancel": None, "undetermined": "Not said yet"},
                }
            },
            "rules": {"intent": {"deny": ["undetermined"], "min": 0.7}},
        }
    )
    assert config.speculation_gate.rules["intent"].deny == ["undetermined"]


# --- gate ----------------------------------------------------------------------------------


def _client_returning(response=None, exc=None):
    client = MagicMock()
    client.post = AsyncMock(return_value=response, side_effect=exc)
    return client


def test_the_call_key_wins_over_the_env(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "env-key")
    assert SpeculationGate({}, provider_api_keys={"typesafe": "call-key"}).api_key == "call-key"
    assert SpeculationGate({}).api_key == "env-key"


def test_no_key_disables_the_gate(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    assert SpeculationGate({}).enabled is False


async def test_a_vetoing_answer_disallows(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "k")
    body = {"answers": {"complete": {"type": "noul", "noul": 0.1}}, "usage": {"input_tokens": 300, "output_tokens": 60}}
    client = _client_returning(httpx.Response(200, json=body))
    monkeypatch.setattr(sg, "get_shared_http_client", lambda: client)
    gate = SpeculationGate({})
    verdict = await gate.evaluate("I want to book a table for", [{"role": "assistant", "content": "Hi!"}])
    assert verdict.allow is False and verdict.failed == ["complete"]
    sent = client.post.call_args.kwargs["json"]
    assert sent["model"] == "jev-latest"
    assert sent["state"] == {
        "conversation": [{"role": "assistant", "content": "Hi!"}],
        "caller_so_far": "I want to book a table for",
    }
    assert gate.usage_totals["vetoes"] == 1 and gate.usage_totals["input_tokens"] == 300


@pytest.mark.parametrize(
    "client",
    [
        _client_returning(httpx.Response(500, text="boom")),
        _client_returning(exc=httpx.ReadTimeout("slow")),
    ],
)
async def test_an_error_or_timeout_fails_open(monkeypatch, client):
    monkeypatch.setenv("TYPESAFE_API_KEY", "k")
    monkeypatch.setattr(sg, "get_shared_http_client", lambda: client)
    verdict = await SpeculationGate({}).evaluate("hello", [])
    assert verdict.allow is True and verdict.error


def test_only_the_last_turns_and_the_agent_context_are_sent(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "k")
    gate = SpeculationGate({"history_turns": 2, "context": "Restaurant booking line"})
    history = [{"role": "system", "content": "prompt"}] + [{"role": "user", "content": str(i)} for i in range(5)]
    state = gate.build_state("so", history)
    assert state["conversation"] == [{"role": "user", "content": "3"}, {"role": "user", "content": "4"}]
    assert state["context"] == "Restaurant booking line"


# --- TaskManager flow ----------------------------------------------------------------------

_EAGER = {
    "data": {"type": "eager_end_of_turn", "content": "I want to book for", "confidence": None},
    "meta_info": {"io": "plivo", "sequence_id": 2, "request_id": "req-1"},
}


def _make_tm(verdict):
    tm = MagicMock()
    tm.hangup_triggered = False
    tm._end_call_in_progress = False
    tm.has_transfer = False
    tm.stream = True
    tm.history = [{"role": "assistant", "content": "How can I help?"}]
    tm.response_in_pipeline = False
    tm.function_call_in_flight = False
    tm.output_task = MagicMock()
    tm.eager_llm_task = None
    tm.eager_gate_task = None
    tm.eager_history_snapshot = None
    tm.transcriber_output_queue = asyncio.Queue()
    tm.process_transcriber_request = AsyncMock(return_value=0)
    tm._set_call_details = MagicMock()
    tm._get_next_step = MagicMock(return_value="llm")
    transcriber = SimpleNamespace(eager_end_of_turn=True, eager_eot_threshold=None, current_turn_id=1)
    tm.tools = {"input": MagicMock(), "transcriber": transcriber}
    tm.tools["input"].welcome_message_played = MagicMock(return_value=True)
    tm.tools["input"].is_audio_being_played_to_user = MagicMock(return_value=False)
    tm.interruption_manager = InterruptionManager(number_of_words_for_interruption=2)
    tm.regen_settle_armed = MagicMock(return_value=False)
    llm_started = asyncio.Event()

    async def slow_llm(_packet):
        llm_started.set()
        await asyncio.sleep(10)

    tm._run_llm_task = slow_llm
    tm._TaskManager__get_updated_meta_info = MagicMock(side_effect=lambda m: dict(m))
    tm.task_config = {"tools_config": {"transcriber": {"provider": "deepgram"}}}
    tm.speculation_gate = SimpleNamespace(enabled=True, evaluate=AsyncMock(return_value=verdict))
    for name in ("_should_ignore_transcriber_input", "_listen_transcriber", "_veto_eager_speculation"):
        setattr(tm, name, getattr(TaskManager, name).__get__(tm, TaskManager))
    tm._cancel_eager_gate = TaskManager._cancel_eager_gate.__get__(tm, TaskManager)
    return tm, llm_started


async def _drive(tm):
    await tm.transcriber_output_queue.put(_EAGER)
    try:
        await asyncio.wait_for(tm._listen_transcriber(), timeout=0.3)
    except asyncio.TimeoutError:
        pass


async def test_a_veto_drops_the_speculation_and_its_user_row():
    tm, llm_started = _make_tm(GateVerdict(allow=False, failed=["complete"]))
    await _drive(tm)
    assert llm_started.is_set()  # the gate never holds the speculation back
    assert tm.eager_llm_task is None
    assert tm.history == [{"role": "assistant", "content": "How can I help?"}]
    sent_history = tm.speculation_gate.evaluate.call_args.args[1]
    assert sent_history == [{"role": "assistant", "content": "How can I help?"}]


async def test_an_allow_keeps_the_speculation():
    tm, _ = _make_tm(GateVerdict(allow=True))
    await _drive(tm)
    assert tm.eager_llm_task is not None and not tm.eager_llm_task.done()
    assert tm.history[-1]["content"] == "I want to book for"
    tm.eager_llm_task.cancel()


async def test_a_late_veto_leaves_an_adopted_speculation_alone():
    tm, _ = _make_tm(GateVerdict(allow=False, failed=["complete"]))
    eager_task = asyncio.create_task(asyncio.sleep(10))
    adopted = asyncio.create_task(asyncio.sleep(10))
    tm.eager_llm_task = None  # EndOfTurn already moved the speculation to llm_task
    tm.llm_task = adopted
    await tm._veto_eager_speculation(eager_task, "I want to book for", [])
    assert not eager_task.cancelled() and not adopted.cancelled()
    eager_task.cancel()
    adopted.cancel()
