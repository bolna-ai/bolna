"""Unit tests for TaskType enum (#265)."""

import pytest
from pydantic import ValidationError

from bolna.enums import TaskType
from bolna.models import Task, ToolsConfig, ToolsChainModel, Transcriber, LlmAgent


def _minimal_task(task_type: str | None = None) -> Task:
    kwargs = {
        "tools_config": ToolsConfig(
            llm_agent=LlmAgent(
                agent_type="simple_llm_agent",
                agent_flow_type="streaming",
                llm_config={"provider": "openai", "model": "gpt-4o-mini"},
            ),
            transcriber=Transcriber(provider="deepgram", stream=True),
        ),
        "toolchain": ToolsChainModel(execution="parallel", pipelines=[["llm"]]),
    }
    if task_type is not None:
        kwargs["task_type"] = task_type
    return Task(**kwargs)


def test_task_type_all_values():
    assert TaskType.all_values() == ["conversation", "extraction", "summarization", "webhook"]


def test_task_type_str_equality():
    assert TaskType.CONVERSATION == "conversation"
    assert "extraction" == TaskType.EXTRACTION


@pytest.mark.parametrize("task_type", TaskType.all_values())
def test_task_model_accepts_known_task_types(task_type):
    task = _minimal_task(task_type)
    assert task.task_type == task_type


def test_task_model_defaults_to_conversation():
    task = _minimal_task()
    assert task.task_type == TaskType.CONVERSATION.value


def test_task_model_rejects_unknown_task_type():
    with pytest.raises(ValidationError):
        _minimal_task("not_a_real_task")
