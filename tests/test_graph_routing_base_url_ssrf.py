"""The routing hop's base_url goes through the same SSRF guard as the conversation's.

generate() validates the conversation base_url before any outbound call. Routing carries
a separate customer-supplied URL, set either as routing_base_url or copied from the
conversation config, and that one reached the wire unchecked, so a config naming the
cloud metadata endpoint for routing got the request the conversation LLM is refused.
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bolna.agent_types.graph_agent import GraphAgent
from bolna.helpers.function_calling_helpers import SSRFError

PLATFORM_ENV = {"OPENAI_API_KEY": "platform-openai-key"}

NODES = [
    {
        "id": "start",
        "prompt": "hi",
        "edges": [{"to_node_id": "next", "condition": "caller says yes", "function_description": "go next"}],
    },
    {"id": "next", "prompt": "bye", "edges": []},
]


def _build(**config_overrides):
    config = {
        "agent_information": "Test agent",
        "model": "gpt-4o-mini",
        "provider": "openai",
        "llm_key": "conv-key",
        "current_node_id": "start",
        "nodes": NODES,
    }
    config.update(config_overrides)

    def make(**kwargs):
        client = MagicMock()
        client.captured_kwargs = kwargs
        client.route = AsyncMock(return_value=None)
        return client

    with (
        patch.dict(os.environ, PLATFORM_ENV, clear=True),
        patch("bolna.agent_types.graph_agent.SUPPORTED_LLM_PROVIDERS", {"openai": make, "custom": make}),
        patch("bolna.agent_types.graph_agent.OpenAiLLM", side_effect=make),
    ):
        return GraphAgent(config)


async def _route(agent):
    node = agent.get_node_by_id("start")
    return await agent._decide_next_node_llm(node, node["edges"], [{"role": "user", "content": "yes"}], 0.0)


@pytest.mark.asyncio
async def test_a_routing_base_url_is_validated_before_the_routing_call():
    agent = _build(routing_provider="custom", routing_base_url="http://169.254.169.254/latest/meta-data/")

    with patch(
        "bolna.agent_types.graph_agent.guard_llm_base_url",
        AsyncMock(side_effect=SSRFError("Blocked outbound request to a non-public LLM endpoint")),
    ) as guard:
        with pytest.raises(SSRFError):
            await _route(agent)

    guard.assert_awaited_once_with("http://169.254.169.254/latest/meta-data/")
    agent.routing_llm.route.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_allowed_routing_base_url_reaches_the_routing_call():
    agent = _build(routing_provider="custom", routing_base_url="https://router.example.com/v1")

    with patch("bolna.agent_types.graph_agent.guard_llm_base_url", AsyncMock()) as guard:
        await _route(agent)
        await _route(agent)

    guard.assert_awaited_once_with("https://router.example.com/v1")  # validated once, then cached
    assert agent.routing_llm.route.await_count == 2


@pytest.mark.asyncio
async def test_a_base_url_inherited_from_the_conversation_is_validated_too():
    """Routing copies the conversation credentials when it shares its provider, and that
    copy is what the routing client dials."""
    agent = _build(provider="custom", base_url="http://10.0.0.5:8000/v1", routing_provider="custom")

    with patch(
        "bolna.agent_types.graph_agent.guard_llm_base_url", AsyncMock(side_effect=SSRFError("blocked"))
    ) as guard:
        with pytest.raises(SSRFError):
            await _route(agent)

    guard.assert_awaited_once_with("http://10.0.0.5:8000/v1")


@pytest.mark.asyncio
async def test_routing_without_a_base_url_calls_no_guard():
    agent = _build()

    with patch("bolna.agent_types.graph_agent.guard_llm_base_url", AsyncMock()) as guard:
        await _route(agent)

    guard.assert_not_awaited()
    agent.routing_llm.route.assert_awaited_once()
