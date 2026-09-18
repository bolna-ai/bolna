"""The routing decision must be complete before the trailing fields are decoded."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from bolna.agent_manager.task_manager import TaskManager
from bolna.llms.routing_stream import RoutingStreamReader, read_routing_stream


def _tool(name, required):
    return {"function": {"name": name, "parameters": {"required": required, "properties": {}}}}


# Schema order the router emits: the edge's own parameters, then reasoning, then confidence.
TOOLS = [
    _tool("transition_to_billing", ["account_id", "reasoning", "confidence"]),
    _tool("stay_on_current_node", ["reasoning", "confidence"]),
]


def _delta(name=None, arguments=None, usage=None):
    tool_calls = None
    if name is not None or arguments is not None:
        tool_calls = [SimpleNamespace(index=0, id="c1", function=SimpleNamespace(name=name, arguments=arguments))]
    choices = [SimpleNamespace(delta=SimpleNamespace(tool_calls=tool_calls))]
    return SimpleNamespace(choices=choices, usage=usage, service_tier=None)


def _usage_chunk(prompt=1800, completion=40):
    usage = SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=0),
        prompt_tokens_details=SimpleNamespace(cached_tokens=1024),
    )
    return SimpleNamespace(choices=[], usage=usage, service_tier=None)


class _Stream:
    """Async iterable that records how far the consumer actually read."""

    def __init__(self, chunks):
        self._chunks = chunks
        self.consumed = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.consumed >= len(self._chunks):
            raise StopAsyncIteration
        chunk = self._chunks[self.consumed]
        self.consumed += 1
        return chunk


def _fragments(*parts):
    return [_delta(arguments=p) for p in parts]


@pytest.mark.asyncio
async def test_decides_before_the_trailing_fields_are_decoded():
    stream = _Stream(
        [_delta(name="transition_to_billing", arguments='{"account_id"')]
        + _fragments(': "A-1", ', '"reasoning": "caller asked about ', 'an invoice", "confidence": 0.9}')
        + [_usage_chunk()]
    )
    reader = RoutingStreamReader(stream, TOOLS)
    args = await reader.decide()

    assert reader.function_name == "transition_to_billing"
    assert args == {"account_id": "A-1"}
    assert reader.early is True
    # Stopped at the chunk that opened `reasoning`, not at the end of the stream.
    assert stream.consumed < 5

    tail = await reader.finish()
    assert tail["reasoning"] == "caller asked about an invoice"
    assert tail["confidence"] == 0.9
    assert tail["usage"]["input_tokens"] == 1800
    assert tail["usage"]["cached_tokens"] == 1024


@pytest.mark.asyncio
async def test_waits_for_edge_params_when_the_rationale_comes_first():
    stream = _Stream(
        [_delta(name="transition_to_billing", arguments='{"reasoning": "wants billing"')]
        + _fragments(', "account_id": "A-2", "confidence": 0.8}')
        + [_usage_chunk()]
    )
    reader = RoutingStreamReader(stream, TOOLS)
    args = await reader.decide()

    assert reader.early is False
    assert args == {"reasoning": "wants billing", "account_id": "A-2", "confidence": 0.8}


async def test_a_paramless_transition_decides_on_the_function_name_alone():
    stream = _Stream(
        [_delta(name="stay_on_current_node", arguments='{"reasoning": "still ')]
        + _fragments('gathering details", "confidence": 0.4}')
        + [_usage_chunk()]
    )
    reader = RoutingStreamReader(stream, TOOLS)

    assert await reader.decide() == {}
    assert reader.early is True
    assert stream.consumed == 1  # nothing beyond the chunk that opened `reasoning`


@pytest.mark.asyncio
async def test_resolves_at_end_of_stream_when_no_trailing_field_is_emitted():
    stream = _Stream([_delta(name="stay_on_current_node", arguments="{}")] + [_usage_chunk(completion=8)])
    reader = RoutingStreamReader(stream, TOOLS)

    assert await reader.decide() == {}
    assert reader.early is False
    assert (await reader.finish())["usage"]["output_tokens"] == 8


@pytest.mark.asyncio
async def test_no_tool_call_yields_no_decision():
    assert await read_routing_stream(_Stream([_delta(), _usage_chunk()]), TOOLS) is None


@pytest.mark.asyncio
async def test_malformed_arguments_yield_no_decision():
    stream = _Stream([_delta(name="transition_to_billing", arguments='{"account_id": ')])
    assert await RoutingStreamReader(stream, TOOLS).decide() is None


@pytest.mark.asyncio
async def test_early_decision_hands_back_a_tail_task():
    stream = _Stream(
        [_delta(name="stay_on_current_node", arguments="{")]
        + _fragments('"reasoning": "needs clarification", "confidence": 0.6}')
        + [_usage_chunk()]
    )
    result = await read_routing_stream(stream, TOOLS, overflowed=True)

    assert result["function_name"] == "stay_on_current_node"
    assert result["arguments"] == {}
    assert result["overflowed"] is True
    assert isinstance(result["routing_tail"], asyncio.Task)
    tail = await result["routing_tail"]
    assert tail["reasoning"] == "needs clarification"
    assert tail["confidence"] == 0.6


@pytest.mark.asyncio
async def test_exhausted_stream_needs_no_tail():
    stream = _Stream([_delta(name="stay_on_current_node", arguments="{}"), _usage_chunk()])
    result = await read_routing_stream(stream, TOOLS)

    assert result["routing_tail"] is None
    assert result["usage"]["input_tokens"] == 1800


class _StubManager:
    """Only what `_apply_routing_tail` and `_settle_routing_tails` touch on a TaskManager."""

    _apply_routing_tail = TaskManager._apply_routing_tail
    _settle_routing_tails = TaskManager._settle_routing_tails
    _log_routing_response = TaskManager._log_routing_response
    _routing_response_line = staticmethod(TaskManager._routing_response_line)

    def __init__(self):
        self.run_id = "run-1"
        self.on_turn_usage = AsyncMock()
        self.on_overflow = AsyncMock()
        self._routing_tail_tasks = set()

    def spawn(self, tail, entry, routing_info=None):
        task = asyncio.create_task(
            self._apply_routing_tail(
                tail, routing_info or {"previous_node": "dispatch", "routing_type": "llm"}, entry, {}
            )
        )
        self._routing_tail_tasks.add(task)
        task.add_done_callback(self._routing_tail_tasks.discard)
        return task


@pytest.mark.asyncio
async def test_a_rationale_still_lands_when_the_call_ends_right_after_the_hop():
    async def tail():
        await asyncio.sleep(0.05)
        return {"reasoning": "caller wants billing", "confidence": 0.9, "usage": {"input_tokens": 1800}}

    entry = {}
    manager = _StubManager()
    with patch("bolna.agent_manager.task_manager.convert_to_request_log"):
        manager.spawn(tail(), entry)
        await manager._settle_routing_tails()

    assert entry["reasoning"] == "caller wants billing"
    assert entry["confidence"] == 0.9
    assert entry["input_tokens"] == 1800


@pytest.mark.asyncio
async def test_a_hung_rationale_does_not_hold_up_the_hangup(monkeypatch):
    monkeypatch.setattr("bolna.agent_manager.task_manager.ROUTING_TAIL_SETTLE_TIMEOUT", 0.05)

    async def never():
        await asyncio.sleep(30)

    entry = {}
    manager = _StubManager()
    with patch("bolna.agent_manager.task_manager.convert_to_request_log") as log:
        task = manager.spawn(never(), entry)
        await manager._settle_routing_tails()  # returns on the timeout rather than after 30s
        # Asserted without awaiting the task: cancel() only schedules, so the row has to be
        # written by the time settling returns or it lands after the latency snapshot.
        assert log.call_count == 1
        assert "Reasoning" not in log.call_args.kwargs["message"]

    assert task.cancelled()
    assert entry == {}


@pytest.mark.asyncio
async def test_settling_is_a_no_op_on_a_call_that_never_routed():
    # Every call reaches this on teardown, not just graph agents, and asyncio.wait([]) raises.
    await _StubManager()._settle_routing_tails()


@pytest.mark.asyncio
async def test_the_response_row_carries_the_rationale_and_the_token_counts():
    # The row is deferred to the tail precisely so it is not written while both are still
    # in flight, which would leave the trace without a rationale and without usage.
    async def tail():
        return {
            "reasoning": "caller asked about an invoice",
            "confidence": 0.9,
            "usage": {"input_tokens": 1800, "output_tokens": 44, "cached_tokens": 1024},
        }

    routing_info = {
        "previous_node": "dispatch",
        "current_node": "billing",
        "transitioned": True,
        "routing_type": "llm",
        "routing_model": "gpt-4.1-mini",
        "routing_latency_ms": 612.0,
        "routing_started_at": 1_000_000.0,
    }
    manager = _StubManager()
    with patch("bolna.agent_manager.task_manager.convert_to_request_log") as log:
        manager.spawn(tail(), {}, routing_info)
        await manager._settle_routing_tails()

    row = log.call_args.kwargs
    assert row["message"] == "Node: dispatch → billing | Confidence: 0.9 | Reasoning: caller asked about an invoice"
    assert (row["input_tokens"], row["output_tokens"], row["cached_tokens"]) == (1800, 44, 1024)
    assert row["latency"] == 0.612
    # Stamped at the hop, not when the tail happened to land.
    assert row["ts"] == 1_000_000.612


@pytest.mark.asyncio
async def test_a_second_tool_call_does_not_corrupt_the_first():
    # parallel_tool_calls is off, but some backends ignore it. Appending the second call's
    # arguments onto the first would leave neither parseable and drop the hop to its catch-all.
    second = SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(
                    tool_calls=[
                        SimpleNamespace(
                            index=1,
                            id="c2",
                            function=SimpleNamespace(name="stay_on_current_node", arguments='{"reasoning": "no"}'),
                        )
                    ]
                )
            )
        ],
        usage=None,
        service_tier=None,
    )
    stream = _Stream(
        [_delta(name="transition_to_billing", arguments='{"account_id": "A-9", ')]
        + [second]
        + _fragments('"reasoning": "invoice", "confidence": 0.8}')
        + [_usage_chunk()]
    )
    reader = RoutingStreamReader(stream, TOOLS)

    assert await reader.decide() == {"account_id": "A-9"}
    assert reader.function_name == "transition_to_billing"


@pytest.mark.asyncio
async def test_an_overflowed_hop_meters_against_the_overflow_backend():
    # The streamed decision has no usage yet, so which backend served the hop has to survive
    # on routing_usage or the PTU pool is billed for tokens the overflow backend served.
    async def tail():
        return {"reasoning": "r", "confidence": 1.0, "usage": {"input_tokens": 1800, "output_tokens": 40}}

    routing_info = {
        "previous_node": "dispatch",
        "current_node": "billing",
        "transitioned": True,
        "routing_type": "llm",
        "routing_provider": "azure",
        "routing_usage": {"overflowed": True, "service_tier": "priority"},
    }
    manager = _StubManager()
    with patch("bolna.agent_manager.task_manager.convert_to_request_log"):
        manager.spawn(tail(), {}, routing_info)
        await manager._settle_routing_tails()

    manager.on_overflow.assert_awaited_once_with(1800, 40, None)
    manager.on_turn_usage.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_deterministic_hop_keeps_its_marker_but_takes_the_spent_tokens():
    # An intent call that declined leaves its telemetry on the catch-all hop. The tokens
    # belong there; the free-text rationale does not, or the hop stops classifying as
    # deterministic downstream.
    async def tail():
        return {"reasoning": "free text", "confidence": 0.4, "usage": {"input_tokens": 1800}}

    entry = {"reasoning": "deterministic:router:unconditional"}
    routing_info = {
        "previous_node": "dispatch",
        "current_node": "general",
        "transitioned": True,
        "routing_type": "deterministic",
    }
    manager = _StubManager()
    with patch("bolna.agent_manager.task_manager.convert_to_request_log"):
        manager.spawn(tail(), entry, routing_info)
        await manager._settle_routing_tails()

    assert entry["reasoning"] == "deterministic:router:unconditional"
    assert entry["input_tokens"] == 1800


@pytest.mark.asyncio
async def test_a_tail_without_usage_does_not_blank_the_recorded_counts():
    # Some providers never send a usage chunk; the hop's own counts must survive.
    async def tail():
        return {"reasoning": "r", "usage": {}}

    entry = {"input_tokens": 1800, "output_tokens": 44}
    manager = _StubManager()
    with patch("bolna.agent_manager.task_manager.convert_to_request_log"):
        manager.spawn(tail(), entry)
        await manager._settle_routing_tails()

    assert (entry["input_tokens"], entry["output_tokens"]) == (1800, 44)
