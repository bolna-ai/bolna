import json

import pytest
from pydantic import ValidationError

from bolna.agent_manager.task_manager import TaskManager
from bolna.models import (
    GraphAgentConfig,
    KnowledgeAgentConfig,
    LlmAgent,
    MultiAgent,
    SimpleLlmAgent,
)


@pytest.mark.parametrize(
    ("agent_type", "llm_config", "expected_config_type"),
    [
        (
            "simple_llm_agent",
            {"provider": "openai", "model": "gpt-4o-mini"},
            SimpleLlmAgent,
        ),
        (
            "knowledgebase_agent",
            {"provider": "openai", "model": "gpt-4o-mini"},
            KnowledgeAgentConfig,
        ),
        (
            "graph_agent",
            {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "agent_information": "Test graph",
                "nodes": [{"id": "start"}],
                "current_node_id": "start",
            },
            GraphAgentConfig,
        ),
        (
            "multiagent",
            {
                "agent_map": {"default": {"provider": "openai", "model": "gpt-4o-mini"}},
                "agent_routing_config": {},
                "default_agent": "default",
            },
            MultiAgent,
        ),
    ],
)
def test_supported_agent_types_still_validate(agent_type, llm_config, expected_config_type):
    agent = LlmAgent(
        agent_flow_type="streaming",
        agent_type=agent_type,
        llm_config=llm_config,
    )

    assert isinstance(agent.llm_config, expected_config_type)


def test_legacy_llm_agent_graph_fails_with_migration_message():
    with pytest.raises(ValidationError, match="migrate this configuration to agent_type 'graph_agent'"):
        LlmAgent(
            agent_flow_type="streaming",
            agent_type="llm_agent_graph",
            llm_config={},
        )


def test_unknown_agent_type_fails_during_validation():
    with pytest.raises(ValidationError, match="Unsupported agent_type: made_up_agent"):
        LlmAgent(
            agent_flow_type="streaming",
            agent_type="made_up_agent",
            llm_config={},
        )


def test_public_schema_does_not_advertise_legacy_graph_config():
    schema = json.dumps(LlmAgent.model_json_schema())

    assert "LlmAgentGraph" not in schema


def test_runtime_dispatch_raises_value_error_for_unsupported_type():
    task_manager = TaskManager.__new__(TaskManager)

    with pytest.raises(ValueError, match="Unsupported runtime agent_type: made_up_agent"):
        TaskManager._TaskManager__get_agent_object(
            task_manager,
            llm=object(),
            agent_type="made_up_agent",
        )
