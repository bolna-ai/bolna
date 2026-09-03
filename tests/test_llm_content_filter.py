"""A content-filter rejection ends the call, like every other LLM error.

No provider swallows it, and no agent turns it into a spoken chunk. See INIT.md (run a66bfe9a).
"""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from litellm import ContentPolicyViolationError
from openai import BadRequestError

from bolna.agent_types.graph_agent import GraphAgent
from bolna.agent_types.knowledgebase_agent import KnowledgeBaseAgent
from bolna.enums import NodeType
from bolna.helpers.utils import format_error_message
from bolna.llms.azure_llm import AzureLLM
from bolna.llms.litellm import LiteLLM
from bolna.llms.openai_llm import OpenAiLLM

# Verbatim from execution a66bfe9a-e1b8-4d5a-9ee4-635362306936 (Meritto, 2026-08-24).
AZURE_FILTER_MESSAGE = (
    "Error code: 400 - {'error': {'message': \"The response was filtered due to the prompt "
    "triggering Azure OpenAI's content management policy. Please modify your prompt and retry.\", "
    "'type': null, 'param': 'prompt', 'code': 'content_filter', 'status': 400}}"
)


def _bad_request(code, message=AZURE_FILTER_MESSAGE):
    error_obj = {"message": message, "param": "prompt", "code": code, "status": 400}
    request = httpx.Request("POST", "https://example.invalid/chat/completions")
    response = httpx.Response(400, request=request, json={"error": error_obj})
    return BadRequestError(message, response=response, body=error_obj)


async def _drain(agen):
    return [chunk async for chunk in agen]


def _stub_llm(error):
    llm = MagicMock()
    llm.model_args = {}
    llm.model_family = "gpt-4.1-mini"
    llm.trigger_function_call = False
    llm.request_log_model = "azure/ptu-gpt-4-1-mini"
    llm.run_id = "a66bfe9a-e1b8-4d5a-9ee4-635362306936"
    llm._create_completion = AsyncMock(side_effect=error)
    llm.async_client.chat.completions.create = AsyncMock(side_effect=error)
    return llm


# ------------------------------------------------------------------ no provider swallows it


async def test_azure_lets_a_content_filter_400_end_the_call():
    llm = _stub_llm(_bad_request("content_filter"))
    with pytest.raises(BadRequestError):
        await _drain(AzureLLM._generate_stream_chat(llm, [{"role": "user", "content": "hi"}], meta_info={}))


async def test_azure_still_raises_other_bad_requests():
    llm = _stub_llm(_bad_request("invalid_request_error", message="tools array is empty"))
    with pytest.raises(BadRequestError):
        await _drain(AzureLLM._generate_stream_chat(llm, [{"role": "user", "content": "hi"}], meta_info={}))


async def test_openai_lets_a_content_filter_400_end_the_call():
    llm = _stub_llm(_bad_request("content_filter"))
    llm.use_responses_api = False
    with pytest.raises(BadRequestError):
        await _drain(OpenAiLLM._generate_stream_chat(llm, [{"role": "user", "content": "hi"}], meta_info={}))


async def test_litellm_lets_a_content_policy_violation_end_the_call():
    llm = MagicMock()
    llm.model = "azure/gpt-4.1-mini"
    llm.model_args = {}
    llm.trigger_function_call = False
    llm.run_id = "r"
    error = ContentPolicyViolationError(message="blocked", model="m", llm_provider="azure")
    with patch("bolna.llms.litellm.acompletion", AsyncMock(side_effect=error)):
        with pytest.raises(ContentPolicyViolationError):
            await _drain(LiteLLM.generate_stream(llm, [{"role": "user", "content": "hi"}], meta_info={}))


# ------------------------------------------------------------------ nothing reaches the TTS


def _graph_stub(error):
    agent = MagicMock()
    agent._event_triggered_generation = False
    agent._node_type_of.return_value = NodeType.LLM
    agent._should_hold_for_first_delivery.return_value = False
    agent.decide_next_node_with_functions = AsyncMock(side_effect=error)
    return agent


async def test_graph_agent_propagates_instead_of_speaking_the_error():
    with pytest.raises(BadRequestError):
        await _drain(GraphAgent.generate(_graph_stub(_bad_request("content_filter")), [], meta_info={}))


async def test_graph_agent_yields_no_chunk_before_it_raises():
    chunks = []
    agen = GraphAgent.generate(_graph_stub(RuntimeError("boom")), [{"role": "user", "content": "hi"}], meta_info={})
    with pytest.raises(RuntimeError):
        async for chunk in agen:
            chunks.append(chunk)
    assert chunks == []


async def test_knowledgebase_agent_propagates_instead_of_speaking_the_error():
    agent = MagicMock()
    agent._add_rag_context = AsyncMock(side_effect=_bad_request("content_filter"))
    with pytest.raises(BadRequestError):
        await _drain(KnowledgeBaseAgent.generate(agent, [], meta_info={"sequence_id": 1}))


async def test_the_filtered_turn_speaks_nothing_end_to_end():
    """The Meritto call: nothing is synthesized, and the error still ends the call."""
    azure = _stub_llm(_bad_request("content_filter"))
    agent = MagicMock()
    agent._event_triggered_generation = False
    agent._node_type_of.return_value = NodeType.LLM
    agent._should_hold_for_first_delivery.return_value = False
    agent.node_history = []
    agent.decide_next_node_with_functions = AsyncMock(return_value=(None, None, 0.0, None, None, None, None, None))
    agent._build_messages = AsyncMock(return_value=[{"role": "user", "content": "attempt it again"}])
    agent._get_tool_choice_for_node.return_value = None
    agent._tools_for_node.return_value = []
    agent.llm.generate_stream = lambda *a, **k: AzureLLM._generate_stream_chat(azure, *a, **k)

    chunks = []
    with pytest.raises(BadRequestError):
        async for chunk in GraphAgent.generate(agent, [{"role": "user", "content": "hi"}], meta_info={}):
            chunks.append(chunk)

    # Only routing/messages bookkeeping dicts; task_manager `continue`s past those.
    assert all(isinstance(c, dict) for c in chunks)
    assert not any("content management policy" in str(c) for c in chunks)


# ------------------------------------------------------------------ customer-facing trace row


def test_azure_wording_maps_to_the_friendly_content_policy_message():
    # Azure says "content management policy"; the old mapper matched neither spelling.
    assert format_error_message("llm", "azure", AZURE_FILTER_MESSAGE) == (
        "Content policy violation - response blocked by safety filter"
    )


def test_openai_wording_still_maps():
    assert format_error_message("llm", "openai", "Request blocked by content policy") == (
        "Content policy violation - response blocked by safety filter"
    )
