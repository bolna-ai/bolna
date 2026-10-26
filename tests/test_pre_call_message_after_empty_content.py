"""The pre-call filler plays when the stream before the call says nothing, like vLLM's empty opening chunk."""

import json

import httpx
import pytest
from openai import AsyncOpenAI

from bolna.llms import AzureLLM, OpenAiLLM

FILLER = "One moment while I check that."
TOOL = "custom_task_fetch_report"
API_TOOLS = {
    "tools": [
        {
            "type": "function",
            "function": {
                "name": TOOL,
                "description": "Fetch the caller's report.",
                "parameters": {"type": "object", "properties": {"user_id": {"type": "string"}}, "required": []},
            },
        }
    ],
    "tools_params": {TOOL: {"url": "https://tool.test/report", "method": "POST", "pre_call_message": FILLER}},
}


def sse(delta=None, finish_reason=None):
    choice = {"index": 0, "delta": delta or {}, "finish_reason": finish_reason}
    chunk = {"id": "chatcmpl-x", "object": "chat.completion.chunk", "created": 0, "model": "m", "choices": [choice]}
    return f"data: {json.dumps(chunk)}\n\n".encode()


def tool_call_delta():
    call = {
        "index": 0,
        "id": "call-1",
        "type": "function",
        "function": {"name": TOOL, "arguments": '{"user_id": "u1"}'},
    }
    return {"tool_calls": [call]}


class Body(httpx.AsyncByteStream):
    def __init__(self, events):
        self.events = events

    async def __aiter__(self):
        for event in self.events:
            yield event


def client(events):
    def handle(request):
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=Body(events))

    return AsyncOpenAI(
        api_key="k",
        base_url="https://backend.test/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
    )


def openai_llm():
    return OpenAiLLM(model="test-model", llm_key="k", api_tools=API_TOOLS)


def azure_llm():
    return AzureLLM(model="gpt-4.1-mini", llm_key="k", base_url="https://x.openai.azure.com", api_tools=API_TOOLS)


async def fillers(llm, events):
    llm.async_client = client(events + [sse(finish_reason="tool_calls"), b"data: [DONE]\n\n"])
    chunks = [c async for c in llm.generate_stream([{"role": "user", "content": "hi"}], meta_info={"sequence_id": 1})]
    return [c.data for c in chunks if c.function_name == TOOL]


@pytest.mark.parametrize("make_llm", [openai_llm, azure_llm], ids=["openai", "azure"])
@pytest.mark.parametrize("silent", ["", "\n\n"], ids=["empty", "whitespace"])
async def test_a_chunk_that_says_nothing_still_lets_the_filler_play(make_llm, silent):
    events = [sse({"role": "assistant", "content": silent}), sse(tool_call_delta())]

    assert await fillers(make_llm(), events) == [FILLER]


@pytest.mark.parametrize("make_llm", [openai_llm, azure_llm], ids=["openai", "azure"])
async def test_spoken_text_before_the_call_still_replaces_the_filler(make_llm):
    events = [sse({"role": "assistant", "content": ""}), sse({"content": "Checking that now."}), sse(tool_call_delta())]

    assert await fillers(make_llm(), events) == []
