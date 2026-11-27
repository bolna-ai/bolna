import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bolna.agent_types.graph_agent import GraphAgent
from bolna.llms.decision_model import Decision, DecisionError
from bolna.llms.decision_router import DecisionRouter, edge_choices, routing_context
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
LLM_ROUTER = ("openai", "gpt-4.1-mini", None)
STAY = {"function_name": "stay_on_current_node", "arguments": {}}


def router(choose, llm_route=None, available=True):
    decision_model = MagicMock(available=available)
    decision_model.choose = AsyncMock(side_effect=choose) if callable(choose) else AsyncMock(return_value=choose)
    llm_router = MagicMock()
    llm_router.route = AsyncMock(return_value=llm_route)
    return DecisionRouter(decision_model, llm_router, LLM_ROUTER), decision_model, llm_router


def decided(choice):
    return Decision(choice, 0.9, {"input_tokens": 120, "output_tokens": 1})


def fails(error):
    def choose(*_):
        raise error

    return choose


async def test_the_edges_are_one_choice_question_over_the_routing_prompt():
    r, decision_model, llm_router = router(decided("stay_on_current_node"))

    hop = await r.route(MESSAGES, TOOLS)

    context, _, choices = decision_model.choose.call_args.args
    assert (
        context
        == routing_context(MESSAGES)
        == ("Book a demo.\n\n# Conversation so far\nAgent: Which day works?\nCaller: Monday at nine")
    )
    assert (
        choices
        == edge_choices(TOOLS)
        == {
            "stay_on_current_node": "Call this function when: nothing else matches",
            "transition_to_book": "Call this function when: the caller picks a slot",
        }
    )
    assert (hop["function_name"], hop["arguments"]) == ("stay_on_current_node", {"confidence": 0.9})
    assert hop["usage"]["input_tokens"] == 120 and hop["routing_tail"] is None
    llm_router.route.assert_not_awaited()


async def test_the_chosen_edges_parameters_come_from_the_llm_router_forced_onto_that_edge():
    fill_tail = asyncio.get_running_loop().create_future()
    fill_tail.set_result({"usage": {"input_tokens": 900, "output_tokens": 30}, "reasoning": "r"})
    filled = {
        "function_name": "transition_to_book",
        "arguments": {"slot": "mon 9", "reasoning": "r", "confidence": 1},
        "routing_tail": fill_tail,
    }
    r, _, llm_router = router(decided("transition_to_book"), llm_route=filled)

    hop = await r.route(MESSAGES, TOOLS)

    messages, tools, tool_choice, _ = llm_router.route.call_args.args
    assert (messages, tools) == (MESSAGES, TOOLS)
    assert tool_choice == {"type": "function", "function": {"name": "transition_to_book"}}
    assert hop["arguments"] == {"slot": "mon 9", "confidence": 0.9}
    # The hop keeps the decision's usage; the LLM call that filled the parameters is reported beside it.
    assert await hop["routing_tail"] == {
        "usage": {"input_tokens": 120, "output_tokens": 1},
        "parameter_fill": {
            "routing_provider": "openai",
            "routing_model": "gpt-4.1-mini",
            "usage": {"input_tokens": 900, "output_tokens": 30},
        },
    }


async def test_a_hop_the_decision_model_cannot_answer_is_routed_and_reported_by_the_llm_router():
    r, decision_model, llm_router = router(fails(DecisionError("not an edge")), llm_route=STAY)

    first, second = await r.route(MESSAGES, TOOLS), await r.route(MESSAGES, TOOLS)

    assert llm_router.route.call_args.args[:3] == (MESSAGES, TOOLS, "required")
    assert first == second == {**STAY, "routed_by": LLM_ROUTER}
    assert decision_model.choose.await_count == 2


async def test_a_failing_decision_api_is_not_asked_again_in_the_call():
    r, decision_model, _ = router(fails(TimeoutError()), llm_route=STAY)

    hops = [await r.route(MESSAGES, TOOLS) for _ in range(3)]

    assert all(hop["routed_by"] == LLM_ROUTER for hop in hops)
    assert decision_model.choose.await_count == 1


async def test_without_its_key_every_hop_routes_on_the_llm_router():
    r, decision_model, _ = router(decided("stay_on_current_node"), llm_route=STAY, available=False)

    assert (await r.route(MESSAGES, TOOLS))["routed_by"] == LLM_ROUTER
    decision_model.choose.assert_not_awaited()


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


def decision_routed_agent(built):
    def llm_router(**kwargs):
        built.append(kwargs)
        return MagicMock()

    edge = {"to_node_id": "done", "condition": "the caller is finished"}
    config = {
        "agent_information": "a",
        "model": "gpt-4.1-mini",
        "provider": "openai",
        "llm_key": "conv-key",
        "routing_provider": "openai-decisions",
        "current_node_id": "start",
        "nodes": [{"id": "start", "prompt": "hi", "edges": [edge]}, {"id": "done", "prompt": "bye", "edges": []}],
    }
    with (
        patch.dict("os.environ", {"OPENAI_API_KEY": "oa-key"}),
        patch("bolna.agent_types.graph_agent.SUPPORTED_LLM_PROVIDERS", {"openai": llm_router}),
        patch("bolna.agent_types.graph_agent.OpenAiLLM", side_effect=lambda **kw: MagicMock()),
    ):
        return GraphAgent(config)


def test_a_graph_routed_on_a_decision_model_keeps_its_llm_router_behind_it():
    built = []
    agent = decision_routed_agent(built)

    assert isinstance(agent.routing_llm, DecisionRouter)
    # Behind it, the router the agent would have had without one: the platform's default on its own provider.
    assert (built[0]["provider"], built[0]["model"]) == ("openai", "gpt-4.1-mini")
    assert agent.routing_llm.llm_router_identity == ("openai", "gpt-4.1-mini", None)
    assert (agent.routing_provider, agent.routing_model) == ("openai-decisions", "gpt-6-luna")


async def test_a_hop_the_llm_router_answered_is_reported_under_the_llm_router():
    agent = decision_routed_agent([])
    node = agent.get_node_by_id("start")
    history = [{"role": "user", "content": "that is all"}]

    async def hop(result):
        agent._reset_routing_identity()
        agent.routing_llm.route = AsyncMock(return_value=result)
        await agent._decide_next_node_llm(node, node["edges"], history, 0.0)
        return agent._last_routing_provider, agent._last_routing_model

    assert await hop({**STAY, "routed_by": LLM_ROUTER}) == ("openai", "gpt-4.1-mini")
    assert await hop(dict(STAY)) == ("openai-decisions", "gpt-6-luna")
