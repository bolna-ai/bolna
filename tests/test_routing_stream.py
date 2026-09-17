"""The routing decision must be complete before the observability fields are decoded."""

import asyncio
from types import SimpleNamespace

import pytest

from bolna.llms.routing_stream import RoutingStreamReader, read_routing_stream


def _tool(name, required):
    return {"function": {"name": name, "parameters": {"required": required, "properties": {}}}}


# Production order: the edge's own parameters, then reasoning, then confidence.
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
async def test_decides_before_the_observability_fields_are_decoded():
    stream = _Stream(
        [_delta(name="transition_to_billing", arguments='{"account_id"')]
        + _fragments(': "A-1", ', '"reasoning": "caller asked about ', 'an invoice", "confidence": 0.9}')
        + [_usage_chunk()]
    )
    reader = RoutingStreamReader(stream, TOOLS)
    args = await reader.decide()

    assert reader.function_name == "transition_to_billing"
    assert args == {"account_id": "A-1"}  # the edge parameter, and nothing after it
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
async def test_resolves_at_end_of_stream_when_nothing_trailing_is_emitted():
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
    assert result["decided_early"] is True
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
