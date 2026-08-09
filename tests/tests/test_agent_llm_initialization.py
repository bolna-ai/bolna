from unittest.mock import Mock

import openai
import pytest

from bolna.agent_types import graph_agent, knowledgebase_agent
from bolna.agent_types.graph_agent import GraphAgent
from bolna.agent_types.knowledgebase_agent import KnowledgeBaseAgent
from bolna.exceptions import LLMError


class FailingLLM:
    def __init__(self, **kwargs):
        raise RuntimeError("provider constructor failed")


def _uninitialized_agent(agent_class, provider="azure", model="configured-model"):
    agent = agent_class.__new__(agent_class)
    agent.config = {"provider": provider}
    agent.llm_model = model
    if agent_class is GraphAgent:
        agent.llm_key = None
    return agent


@pytest.mark.parametrize(
    ("agent_class", "agent_module"),
    [
        (GraphAgent, graph_agent),
        (KnowledgeBaseAgent, knowledgebase_agent),
    ],
)
def test_llm_initialization_failure_is_attributed_and_preserves_cause(monkeypatch, agent_class, agent_module):
    agent = _uninitialized_agent(agent_class)
    monkeypatch.setitem(agent_module.SUPPORTED_LLM_PROVIDERS, "azure", FailingLLM)

    with pytest.raises(LLMError) as exc_info:
        agent._initialize_llm()

    assert exc_info.value.provider == "azure"
    assert exc_info.value.model == "configured-model"
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert str(exc_info.value.__cause__) == "provider constructor failed"


def test_graph_agent_does_not_fall_back_to_openai(monkeypatch):
    agent = _uninitialized_agent(GraphAgent)
    fallback = Mock()
    monkeypatch.setitem(graph_agent.SUPPORTED_LLM_PROVIDERS, "azure", FailingLLM)
    monkeypatch.setattr(graph_agent, "OpenAiLLM", fallback)

    with pytest.raises(LLMError):
        agent._initialize_llm()

    fallback.assert_not_called()


def test_knowledge_agent_does_not_fall_back_to_openai(monkeypatch):
    agent = _uninitialized_agent(KnowledgeBaseAgent)
    fallback = Mock()
    monkeypatch.setitem(knowledgebase_agent.SUPPORTED_LLM_PROVIDERS, "azure", FailingLLM)
    monkeypatch.setattr(openai, "OpenAI", fallback)

    with pytest.raises(LLMError):
        agent._initialize_llm()

    fallback.assert_not_called()


@pytest.mark.parametrize("agent_class", [GraphAgent, KnowledgeBaseAgent])
def test_unknown_provider_is_rejected_without_being_rewritten(agent_class):
    agent = _uninitialized_agent(agent_class, provider="unknown-provider")

    with pytest.raises(LLMError) as exc_info:
        agent._initialize_llm()

    assert exc_info.value.provider == "unknown-provider"
    assert isinstance(exc_info.value.__cause__, ValueError)
    assert str(exc_info.value.__cause__) == "Unsupported LLM provider: unknown-provider"
