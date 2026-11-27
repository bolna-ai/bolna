import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from bolna.llms.decision_router import DecisionRouter
from bolna.models import GraphAgentConfig

TRAILING = {"reasoning": {"type": "string"}, "confidence": {"type": "number"}}
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "stay_on_current_node",
            "description": "Call this function when: nothing else matches",
            "parameters": {"type": "object", "properties": TRAILING, "required": ["reasoning", "confidence"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "transition_to_book",
            "description": "Call this function when: the caller picks a slot",
            "parameters": {
                "type": "object",
                "properties": {"slot": {"type": "string"}, **TRAILING},
                "required": ["slot", "reasoning", "confidence"],
            },
        },
    },
]
MESSAGES = [
    {"role": "system", "content": "Book a demo."},
    {"role": "assistant", "content": "Which day works?"},
    {"role": "user", "content": "Monday at nine"},
]
KEYS = {"TYPESAFE_API_KEY": "ts-key", "OPENAI_API_KEY": "oa-key", "GOOGLE_API_KEY": "g-key"}


def router(provider, reply, status=200, fallback=None):
    sent = []

    def handler(request):
        sent.append(request)
        return httpx.Response(status, json=reply)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with (
        patch.dict(os.environ, KEYS),
        patch("bolna.llms.decision_router.get_shared_http_client", return_value=client),
    ):
        return DecisionRouter(provider, fallback=fallback or MagicMock()), sent


def keyed(choice):
    return {"answers": {"next_node": {"choice": choice, "confidence": 0.9}}, "usage": {"input_tokens": 120}}


async def test_jev_scores_every_edge_against_the_routing_context():
    r, sent = router("typesafe", keyed("stay_on_current_node"))

    decision = await r.route(MESSAGES, TOOLS)

    body = json.loads(sent[0].content)
    assert str(sent[0].url) == "https://api.typesafe.ai/v1/systemone"
    assert sent[0].headers["authorization"] == "Bearer ts-key"
    assert body["state"].startswith("Book a demo.")
    assert body["state"].endswith("Agent: Which day works?\nCaller: Monday at nine")
    assert body["questions"]["next_node"]["criteria"] == {
        "stay_on_current_node": "Call this function when: nothing else matches",
        "transition_to_book": "Call this function when: the caller picks a slot",
    }
    assert decision["function_name"] == "stay_on_current_node"
    assert decision["arguments"] == {"confidence": 0.9}
    assert decision["usage"]["input_tokens"] == 120


async def test_gemini_takes_the_same_question_under_its_own_key_and_field():
    r, sent = router("gemini-decisions", keyed("stay_on_current_node"))

    await r.route(MESSAGES, TOOLS)

    body = json.loads(sent[0].content)
    assert sent[0].headers["x-goog-api-key"] == "g-key"
    assert body["model"] == "gemini-decision-1" and "input" in body and "state" not in body


async def test_openai_decisions_lists_its_questions_and_answers():
    reply = {"answers": [{"name": "next_node", "type": "choice", "choice": "stay_on_current_node", "confidence": 0.7}]}
    r, sent = router("openai-decisions", reply)

    decision = await r.route(MESSAGES, TOOLS)

    question = json.loads(sent[0].content)["questions"][0]
    assert [c["value"] for c in question["choices"]] == ["stay_on_current_node", "transition_to_book"]
    assert decision["arguments"] == {"confidence": 0.7}


async def test_the_chosen_edges_parameters_come_from_the_llm_router_forced_onto_that_edge():
    fallback = MagicMock()
    filled = {"function_name": "transition_to_book", "arguments": {"slot": "mon 9", "reasoning": "r", "confidence": 1}}
    fallback.route = AsyncMock(return_value=filled)
    r, _ = router("typesafe", keyed("transition_to_book"), fallback=fallback)

    decision = await r.route(MESSAGES, TOOLS)

    messages, tools, tool_choice, _ = fallback.route.call_args.args
    assert (messages, tools) == (MESSAGES, TOOLS)
    assert tool_choice == {"type": "function", "function": {"name": "transition_to_book"}}
    assert decision["arguments"] == {"slot": "mon 9", "confidence": 0.9}


@pytest.mark.parametrize("reply, status", [(keyed("transition_to_nowhere"), 200), ({"error": "overloaded"}, 503)])
async def test_a_hop_the_decision_api_cannot_answer_routes_on_the_llm_router(reply, status):
    fallback = MagicMock()
    fallback.route = AsyncMock(return_value={"function_name": "stay_on_current_node", "arguments": {}})
    r, _ = router("typesafe", reply, status=status, fallback=fallback)

    decision = await r.route(MESSAGES, TOOLS)

    assert fallback.route.call_args.args[:3] == (MESSAGES, TOOLS, "required")
    assert decision["function_name"] == "stay_on_current_node"


async def test_without_its_key_every_hop_routes_on_the_llm_router():
    fallback = MagicMock()
    fallback.route = AsyncMock(return_value={"function_name": "stay_on_current_node", "arguments": {}})
    with patch.dict(os.environ, {}, clear=True):
        r = DecisionRouter("typesafe", fallback=fallback)

    decision = await r.route(MESSAGES, TOOLS)

    assert fallback.route.call_args.args[:3] == (MESSAGES, TOOLS, "required")
    assert decision["function_name"] == "stay_on_current_node"


def test_a_decision_routed_graph_rejects_per_node_routing_overrides():
    config = {
        "agent_information": "a",
        "model": "gpt-4.1-mini",
        "provider": "openai",
        "routing_provider": "typesafe",
        "current_node_id": "start",
        "nodes": [{"id": "start", "prompt": "hi", "routing_model": "gpt-4.1", "edges": []}],
    }

    with pytest.raises(ValueError, match="routing model or effort"):
        GraphAgentConfig(**config)


def test_a_graph_routed_on_a_decision_model_keeps_its_llm_router_beneath_it():
    from bolna.agent_types.graph_agent import GraphAgent

    built = []

    def llm_router(**kwargs):
        built.append(kwargs)
        return MagicMock()

    config = {
        "agent_information": "a",
        "model": "gpt-4.1-mini",
        "provider": "openai",
        "llm_key": "conv-key",
        "routing_provider": "openai-decisions",
        "current_node_id": "start",
        "nodes": [{"id": "start", "prompt": "hi", "edges": []}],
    }
    with (
        patch.dict(os.environ, KEYS),
        patch("bolna.agent_types.graph_agent.SUPPORTED_LLM_PROVIDERS", {"openai": llm_router}),
        patch("bolna.agent_types.graph_agent.OpenAiLLM", side_effect=lambda **kw: MagicMock()),
    ):
        agent = GraphAgent(config)

    assert isinstance(agent.routing_llm, DecisionRouter)
    # Beneath it, the router the agent would have had without one: the platform's default on its own provider.
    assert (built[0]["provider"], built[0]["model"]) == ("openai", "gpt-4.1-mini")
    assert (agent.routing_provider, agent.routing_model) == ("openai-decisions", "gpt-6-luna")
