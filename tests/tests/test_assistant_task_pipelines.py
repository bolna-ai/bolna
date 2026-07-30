import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from bolna.agent_manager.task_manager import TaskManager
from bolna.assistant import Assistant
from bolna.helpers.utils import get_required_input_types
from bolna.input_handlers.default import DefaultInputHandler
from bolna.models import ElevenLabsConfig, LlmAgent, SimpleLlmAgent, Synthesizer, Transcriber


def _llm_agent():
    # Keep the README's public construction style covered: llm_config is a
    # SimpleLlmAgent instance rather than its serialized dict.
    return LlmAgent(
        agent_type="simple_llm_agent",
        agent_flow_type="streaming",
        llm_config=SimpleLlmAgent(provider="openai", model="gpt-4o-mini"),
    )


def _transcriber():
    return Transcriber(provider="deepgram", model="nova-2", stream=True, language="en")


def _synthesizer():
    return Synthesizer(
        provider="elevenlabs",
        provider_config=ElevenLabsConfig(
            voice="George",
            voice_id="JBFqnCBsd6RMkjVDRZzb",
            model="eleven_turbo_v2_5",
        ),
        stream=True,
    )


@pytest.mark.parametrize(
    ("transcriber", "synthesizer", "enable_textual_input", "expected_pipelines"),
    [
        (_transcriber(), _synthesizer(), False, [["transcriber", "llm", "synthesizer"]]),
        (_transcriber(), None, False, [["transcriber", "llm"]]),
        (None, None, True, [["llm"]]),
        (
            _transcriber(),
            _synthesizer(),
            True,
            [["transcriber", "llm", "synthesizer"], ["llm"]],
        ),
        (None, None, False, [["llm"]]),
    ],
)
def test_add_task_builds_pipelines_from_available_components(
    transcriber,
    synthesizer,
    enable_textual_input,
    expected_pipelines,
):
    assistant = Assistant()

    assistant.add_task(
        task_type="conversation",
        llm_agent=_llm_agent(),
        transcriber=transcriber,
        synthesizer=synthesizer,
        enable_textual_input=enable_textual_input,
    )

    task = assistant.tasks[0]
    assert task["toolchain"]["pipelines"] == expected_pipelines
    assert (task["tools_config"]["transcriber"] is not None) is (transcriber is not None)
    assert (task["tools_config"]["synthesizer"] is not None) is (synthesizer is not None)


@pytest.mark.asyncio
async def test_text_input_uses_text_pipeline_sequence():
    queues = {"llm": asyncio.Queue()}
    handler = DefaultInputHandler(
        queues=queues,
        input_types={"audio": 0, "text": 1},
        turn_based_conversation=True,
    )

    await handler.process_message({"type": "text", "data": "hello"})

    packet = queues["llm"].get_nowait()
    assert packet["data"] == "hello"
    assert packet["meta_info"]["sequence"] == 1
    assert packet["meta_info"]["bypass_synth"] is True


@pytest.mark.asyncio
async def test_text_only_readme_task_initializes_without_audio_tools():
    assistant = Assistant(name="text_only_agent")
    assistant.add_task(
        task_type="conversation",
        llm_agent=_llm_agent(),
        enable_textual_input=True,
    )
    task = assistant.tasks[0]

    def setup_input(task_manager, *_args):
        task_manager.tools["input"] = SimpleNamespace(is_dtmf_active=False)

    async def no_initial_message(_task_manager):
        return None

    with (
        patch.object(TaskManager, "_TaskManager__setup_input_handlers", setup_input),
        patch.object(TaskManager, "_TaskManager__setup_llm", return_value=object()),
        patch.object(TaskManager, "_TaskManager__setup_tasks"),
        patch.object(TaskManager, "message_task_new", no_initial_message),
    ):
        task_manager = TaskManager(
            assistant_name=assistant.name,
            task_id=0,
            task=task,
            ws=None,
            turn_based_conversation=True,
        )
        await task_manager.first_message_task_new

    assert get_required_input_types(task) == {"text": 0}
    assert task_manager.pipelines == [["llm"]]
    assert task_manager.synthesizer_voice is None
    assert task_manager.minimum_wait_duration == 0
    assert task_manager.stream is False
    assert "transcriber" not in task_manager.tools
    assert "synthesizer" not in task_manager.tools
    assert task_manager.output_handler_set is True
