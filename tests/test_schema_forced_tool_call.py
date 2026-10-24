"""A forced tool call on the bolna provider goes out as a JSON schema and comes back as a tool call.

The provider's server does not enforce tool_choice, so the forced call is requested as constrained JSON and
re-streamed as tool-call deltas. The turn and the routing hop must read it exactly like a real call.
"""

import json

import httpx
import pytest
from openai import AsyncOpenAI

from bolna.llms import OpenAiLLM
from bolna.llms.types import FunctionCallPayload

FILLER = "One moment while I pull that up."
FETCH = "custom_task_fetch_report"
FETCH_TOOL = {
    "type": "function",
    "function": {
        "name": FETCH,
        "description": "Fetch the caller's report.",
        "parameters": {
            "type": "object",
            "properties": {"user_id": {"type": "string"}},
            "required": ["user_id"],
            "additionalProperties": False,
        },
    },
}
API_TOOLS = {
    "tools": [FETCH_TOOL],
    "tools_params": {FETCH: {"url": "https://tool.test/report", "method": "POST", "pre_call_message": FILLER}},
}
TRAILING = {"reasoning": {"type": "string"}, "confidence": {"type": "number"}}
ROUTING_TOOLS = [
    {
        "type": "function",
        "function": {"name": "transition_to_no_data", "parameters": {"type": "object", "properties": TRAILING}},
    },
    {
        "type": "function",
        "function": {
            "name": "transition_to_book",
            "parameters": {"type": "object", "properties": {"slot": {"type": "object"}, **TRAILING}},
        },
    },
]
MSGS = [{"role": "user", "content": "hi"}]


def sse(delta=None, finish_reason=None):
    choice = {"index": 0, "delta": delta or {}, "finish_reason": finish_reason}
    chunk = {"id": "chatcmpl-x", "object": "chat.completion.chunk", "created": 0, "model": "m", "choices": [choice]}
    return f"data: {json.dumps(chunk)}\n\n".encode()


def constrained_reply(text, piece=3):
    """A schema-constrained stream: vLLM's empty opening chunk, the JSON in small pieces, then stop."""
    pieces = [text[i : i + piece] for i in range(0, len(text), piece)]
    return (
        [sse({"role": "assistant", "content": ""})]
        + [sse({"content": p}) for p in pieces]
        + [sse(finish_reason="stop")]
    )


class Backend:
    def __init__(self, events):
        self.events = events
        self.requests = []

    def client(self):
        async def body():
            for event in self.events + [b"data: [DONE]\n\n"]:
                yield event

        def handle(request):
            self.requests.append(json.loads(request.content))
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body())

        transport = httpx.MockTransport(handle)
        return AsyncOpenAI(
            api_key="k",
            base_url="https://llm.test/v1",
            max_retries=0,
            http_client=httpx.AsyncClient(transport=transport),
        )


def bolna_llm(backend, provider="bolna", **kwargs):
    llm = OpenAiLLM(model="test-model", llm_key="k", base_url="https://llm.test/v1", provider=provider, **kwargs)
    llm.async_client = backend.client()
    return llm


async def forced_turn(llm):
    tool_choice = {"type": "function", "function": {"name": FETCH}}
    return [
        c
        async for c in llm.generate_stream(
            MSGS, meta_info={"sequence_id": 1}, tool_choice=tool_choice, tools=[FETCH_TOOL]
        )
    ]


async def test_a_forced_function_is_requested_as_its_arguments_schema_with_the_tools_still_in_the_prompt():
    backend = Backend(constrained_reply('{"user_id": "u-42"}'))

    await forced_turn(bolna_llm(backend, api_tools=API_TOOLS))

    sent = backend.requests[0]
    # The schema only constrains decoding; the model reads what the function is for from the tools.
    assert [t["function"]["name"] for t in sent["tools"]] == [FETCH]
    assert sent["tool_choice"] == "auto"
    assert sent["response_format"]["json_schema"]["schema"] == FETCH_TOOL["function"]["parameters"]
    assert "stop" not in sent


async def test_a_reply_cut_at_the_token_cap_keeps_its_length_finish():
    events = constrained_reply('{"user_id": "u-')[:-1] + [sse(finish_reason="length")]
    meta_info = {"sequence_id": 1}
    llm = bolna_llm(Backend(events), api_tools=API_TOOLS)

    [
        c
        async for c in llm.generate_stream(
            MSGS, meta_info=meta_info, tool_choice={"type": "function", "function": {"name": FETCH}}, tools=[FETCH_TOOL]
        )
    ]

    assert meta_info["llm_finish_reason"] == "length"


async def test_a_forced_function_comes_back_as_a_tool_call_with_its_filler():
    chunks = await forced_turn(bolna_llm(Backend(constrained_reply('{"user_id": "u-42"}')), api_tools=API_TOOLS))

    calls = [c.data for c in chunks if isinstance(c.data, FunctionCallPayload)]
    assert [(p.called_fun, p.user_id) for p in calls] == [(FETCH, "u-42")]
    assert [c.data for c in chunks if c.function_name == FETCH and isinstance(c.data, str)] == [FILLER]
    assert not "".join(c.data for c in chunks if isinstance(c.data, str) and not c.function_name).strip()


async def test_other_providers_still_force_through_tool_choice():
    backend = Backend(
        [
            sse(
                {
                    "tool_calls": [
                        {"index": 0, "id": "c1", "type": "function", "function": {"name": FETCH, "arguments": "{}"}}
                    ]
                }
            ),
            sse(finish_reason="tool_calls"),
        ]
    )

    await forced_turn(bolna_llm(backend, provider="openai", api_tools=API_TOOLS))

    assert backend.requests[0]["tool_choice"] == {"type": "function", "function": {"name": FETCH}}
    assert backend.requests[0]["response_format"] == {"type": "text"}


async def test_a_routing_hop_picks_among_transitions_by_schema():
    backend = Backend(
        constrained_reply(
            '{"name": "transition_to_no_data", "arguments": {"reasoning": "nothing came back", "confidence": 0.8}}'
        )
    )

    decision = await bolna_llm(backend).route(MSGS, ROUTING_TOOLS)

    sent = backend.requests[0]
    options = sent["response_format"]["json_schema"]["schema"]["anyOf"]
    assert [o["properties"]["name"]["const"] for o in options] == ["transition_to_no_data", "transition_to_book"]
    # Each edge's condition lives only in its function's description, so the tools must stay in the prompt.
    assert [t["function"]["name"] for t in sent["tools"]] == ["transition_to_no_data", "transition_to_book"]
    assert sent["tool_choice"] == "auto"
    assert decision["function_name"] == "transition_to_no_data"
    tail = await decision["routing_tail"]
    assert (tail["reasoning"], tail["confidence"]) == ("nothing came back", 0.8)


@pytest.mark.parametrize("piece", [1, 2, 7, 1000])
async def test_edge_parameters_survive_any_chunking_of_the_reply(piece):
    reply = '{"name": "transition_to_book", "arguments": {"slot": {"day": "mon", "at": {"h": 9}}, "reasoning": "asked for monday", "confidence": 1}}'

    decision = await bolna_llm(Backend(constrained_reply(reply, piece))).route(MSGS, ROUTING_TOOLS)

    assert decision["function_name"] == "transition_to_book"
    assert decision["arguments"] == {"slot": {"day": "mon", "at": {"h": 9}}}
    assert (await decision["routing_tail"])["reasoning"] == "asked for monday"
