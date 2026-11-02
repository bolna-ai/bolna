"""Graph and knowledge-base agents build their own conversation LLM, so TaskManager builds none for them."""

from unittest.mock import MagicMock, patch

import pytest

from bolna.agent_manager.task_manager import TaskManager
from tests.test_llm_verbosity_passthrough import _llm_agent_config, _task_config


@pytest.mark.parametrize(
    "agent_type, builds", [("simple_llm_agent", True), ("graph_agent", False), ("knowledgebase_agent", False)]
)
async def test_task_level_llm_is_built_only_for_agents_that_use_it(agent_type, builds):
    config = _task_config(_llm_agent_config())
    config["tools_config"]["llm_agent"]["agent_type"] = agent_type
    llm_class = MagicMock()
    with (
        patch("bolna.agent_manager.task_manager.SUPPORTED_LLM_PROVIDERS", {"openai": llm_class}),
        patch("bolna.agent_manager.task_manager.GraphAgent", MagicMock()),
        patch("bolna.agent_manager.task_manager.KnowledgeBaseAgent", MagicMock()),
    ):
        TaskManager("agent", 0, config, MagicMock())
    assert llm_class.called is builds
