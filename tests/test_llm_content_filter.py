"""A prompt Azure's content filter rejects must never reach the caller's ear.

Two independent guarantees, one per layer:

1. LLM layer — a 400 carrying code `content_filter` is not a call-breaking fault. The turn is
   dropped and stamped on `_non_fatal_errors`, the same treatment litellm already gives
   ContentPolicyViolationError. `__do_llm_generation` clears `response_in_pipeline` for an
   empty turn, so silence-recovery still fires.
2. Agent layer — graph/KB agents must not turn an exception into a text chunk. A yielded chunk
   is indistinguishable from model output, so it gets synthesized AND written to history.
   Real faults propagate, and the task manager ends the call via LLMError.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from openai import BadRequestError

from bolna.agent_types.graph_agent import GraphAgent
from bolna.agent_types.knowledgebase_agent import KnowledgeBaseAgent
from bolna.enums import LogComponent, NodeType
from bolna.helpers.utils import format_error_message
from bolna.llms.azure_llm import AzureLLM
from bolna.llms.openai_base import is_content_filter_error

# Verbatim from execution a66bfe9a-e1b8-4d5a-9ee4-635362306936 (Meritto, 2026-08-24).
AZURE_FILTER_MESSAGE = (
    "Error code: 400 - {'error': {'message': \"The response was filtered due to the prompt "
    "triggering Azure OpenAI's content management policy. Please modify your prompt and retry.\", "
    "'type': null, 'param': 'prompt', 'code': 'content_filter', 'status': 400}}"
)


def _bad_request(code, *, enveloped=False, message=AZURE_FILTER_MESSAGE):
    """`enveloped` mimics the shape where only the outer body carries the code, so `.code` is None."""
    error_obj = {"message": message, "param": "prompt", "code": code, "status": 400}
    request = httpx.Request("POST", "https://example.invalid/chat/completions")
    response = httpx.Response(400, request=request, json={"error": error_obj})
    return BadRequestError(message, response=response, body={"error": error_obj} if enveloped else error_obj)


# --------------------------------------------------------------------------- detection


def test_detects_the_azure_content_filter_400():
    assert is_content_filter_error(_bad_request("content_filter")) is True


def test_detects_it_when_only_the_outer_body_carries_the_code():
    err = _bad_request("content_filter", enveloped=True)
    assert err.code is None  # the attribute route misses this shape
    assert is_content_filter_error(err) is True


def test_other_400s_are_not_content_filter():
    assert is_content_filter_error(_bad_request("invalid_request_error")) is False


def test_non_api_errors_are_not_content_filter():
    assert is_content_filter_error(ValueError("boom")) is False


# --------------------------------------------------------------------------- LLM layer


def _azure_stub(error):
    llm = MagicMock()
    llm.model_args = {}
    llm.model_family = "gpt-4.1-mini"
    llm.trigger_function_call = False
    llm.request_log_model = "azure/ptu-gpt-4-1-mini"
    llm.run_id = "a66bfe9a-e1b8-4d5a-9ee4-635362306936"
    llm._create_completion = AsyncMock(side_effect=error)
    return llm


async def _drain(agen):
    return [chunk async for chunk in agen]


async def test_a_filtered_prompt_yields_nothing_and_keeps_the_call_alive():
    llm = _azure_stub(_bad_request("content_filter"))
    meta = {"sequence_id": 3, "request_id": "r"}

    with patch("bolna.llms.openai_base.convert_to_request_log") as trace:
        chunks = await _drain(
            AzureLLM._generate_stream_chat(llm, [{"role": "user", "content": "attempt it again"}], meta_info=meta)
        )

    assert chunks == []  # nothing reaches the synthesizer
    assert meta["_non_fatal_errors"] == [
        {
            "error_type": "content_policy_violation",
            "error": AZURE_FILTER_MESSAGE,
            "model": "azure/ptu-gpt-4-1-mini",
        }
    ]
    assert trace.call_args.kwargs["component"] is LogComponent.WARNING


async def test_no_chunk_ever_carries_the_raw_error_text():
    llm = _azure_stub(_bad_request("content_filter"))
    with patch("bolna.llms.openai_base.convert_to_request_log"):
        chunks = await _drain(AzureLLM._generate_stream_chat(llm, [{"role": "user", "content": "hi"}], meta_info={}))
    assert not any("content management policy" in str(chunk) for chunk in chunks)


async def test_a_real_bad_request_still_breaks_the_call():
    # Only content_filter is recoverable. A malformed request must still surface as an
    # LLMError upstream rather than leaving a silent agent on a live line.
    llm = _azure_stub(_bad_request("invalid_request_error", message="tools array is empty"))
    with pytest.raises(BadRequestError):
        await _drain(AzureLLM._generate_stream_chat(llm, [{"role": "user", "content": "hi"}], meta_info={}))


async def test_a_filtered_prompt_without_a_run_id_skips_the_trace_row_but_still_stamps_meta():
    llm = _azure_stub(_bad_request("content_filter"))
    llm.run_id = None
    meta = {"sequence_id": 1}
    with patch("bolna.llms.openai_base.convert_to_request_log") as trace:
        assert (
            await _drain(AzureLLM._generate_stream_chat(llm, [{"role": "user", "content": "hi"}], meta_info=meta)) == []
        )
    trace.assert_not_called()
    assert meta["_non_fatal_errors"][0]["error_type"] == "content_policy_violation"


# --------------------------------------------------------------------------- agent layer


def _graph_stub(error):
    agent = MagicMock()
    agent._event_triggered_generation = False
    agent._node_type_of.return_value = NodeType.LLM
    agent._should_hold_for_first_delivery.return_value = False
    agent.decide_next_node_with_functions = AsyncMock(side_effect=error)
    return agent


async def test_graph_agent_propagates_instead_of_speaking_the_error():
    error = _bad_request("content_filter")
    with pytest.raises(BadRequestError):
        await _drain(GraphAgent.generate(_graph_stub(error), [{"role": "user", "content": "hi"}], meta_info={}))


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
        await _drain(
            KnowledgeBaseAgent.generate(agent, [{"role": "user", "content": "hi"}], meta_info={"sequence_id": 1})
        )


# --------------------------------------------------------------------------- customer-facing trace


def test_azure_wording_maps_to_the_friendly_content_policy_message():
    # Azure says "content management policy" with code content_filter; the old mapper matched
    # neither and printed the raw 400 into the customer's trace.
    assert format_error_message("llm", "azure", AZURE_FILTER_MESSAGE) == (
        "Content policy violation - response blocked by safety filter"
    )


def test_openai_wording_still_maps():
    assert format_error_message("llm", "openai", "Request blocked by content policy") == (
        "Content policy violation - response blocked by safety filter"
    )


async def test_graph_agent_with_a_filtered_prompt_speaks_nothing():
    """Both layers together, i.e. the Meritto call: Azure rejects the prompt, the graph agent
    yields only bookkeeping dicts, and not one chunk of text reaches the synthesizer."""
    azure = _azure_stub(_bad_request("content_filter"))
    meta = {"sequence_id": 7}

    agent = MagicMock()
    agent._event_triggered_generation = False
    agent._node_type_of.return_value = NodeType.LLM
    agent._should_hold_for_first_delivery.return_value = False
    agent.node_history = []
    agent.decide_next_node_with_functions = AsyncMock(return_value=(None, None, 0.0, None, None, None, None, None))
    agent._build_messages = AsyncMock(return_value=[{"role": "user", "content": "planning to attempt it again"}])
    agent._get_tool_choice_for_node.return_value = None
    agent._tools_for_node.return_value = []
    agent.llm.generate_stream = lambda *args, **kwargs: AzureLLM._generate_stream_chat(azure, *args, **kwargs)

    with patch("bolna.llms.openai_base.convert_to_request_log"):
        chunks = await _drain(
            GraphAgent.generate(agent, [{"role": "user", "content": "my school was not good"}], meta_info=meta)
        )

    # Only routing/messages bookkeeping dicts — task_manager `continue`s past those.
    assert chunks and all(isinstance(chunk, dict) for chunk in chunks)
    assert not any("An error occurred" in str(chunk) for chunk in chunks)
    assert meta["_non_fatal_errors"][0]["error_type"] == "content_policy_violation"
