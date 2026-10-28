"""LlmAgent.llm_config accepts a config object or a dict. An object must match agent_type."""

import pytest

from bolna.models import KnowledgeAgentConfig, LlmAgent, SimpleLlmAgent


def test_llm_agent_accepts_simple_llm_agent_instance():
    agent = LlmAgent(
        agent_type="simple_llm_agent",
        agent_flow_type="streaming",
        llm_config=SimpleLlmAgent(provider="openai", model="gpt-4o-mini"),
    )
    assert isinstance(agent.llm_config, SimpleLlmAgent)
    assert agent.llm_config.model == "gpt-4o-mini"


def test_llm_agent_still_accepts_dict():
    agent = LlmAgent(
        agent_type="simple_llm_agent",
        agent_flow_type="streaming",
        llm_config={"provider": "openai", "model": "gpt-4o-mini"},
    )
    assert isinstance(agent.llm_config, SimpleLlmAgent)


def test_llm_agent_rejects_mismatched_instance():
    # Built outside the raises block so a construction error fails the test instead of passing it.
    wrong_config = KnowledgeAgentConfig()
    with pytest.raises(ValueError):
        LlmAgent(
            agent_type="simple_llm_agent",
            agent_flow_type="streaming",
            llm_config=wrong_config,
        )
