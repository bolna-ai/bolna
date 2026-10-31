"""The README quickstart must build a task that TaskManager can construct.

Text-only tasks have no transcriber. Speech-in/text-out tasks have no synthesizer.
A pipeline whose own LLM is not OpenAI must not require OPENAI_API_KEY just to be created.
"""

import asyncio
import os
from unittest.mock import MagicMock

from bolna.agent_types.contextual_conversational_agent import StreamingContextualAgent, completion_llm_for
from bolna.assistant import Assistant
from bolna.models import LlmAgent, SimpleLlmAgent, Transcriber


def _llm_agent():
    return LlmAgent(
        agent_type="simple_llm_agent",
        agent_flow_type="streaming",
        llm_config=SimpleLlmAgent(provider="openai", model="gpt-4o-mini", temperature=0.2),
    )


def test_text_only_task_keeps_a_null_transcriber_and_an_llm_pipeline():
    assistant = Assistant(name="text_only_agent")
    assistant.add_task(task_type="conversation", llm_agent=_llm_agent(), enable_textual_input=True)
    task = assistant.tasks[0]
    assert task["tools_config"]["transcriber"] is None
    assert task["tools_config"]["synthesizer"] is None
    assert all("transcriber" not in pipeline for pipeline in task["toolchain"]["pipelines"])
    assert ["llm"] in task["toolchain"]["pipelines"]


def test_configured_transcriber_is_stored_on_the_task():
    assistant = Assistant(name="demo_agent")
    transcriber = Transcriber(provider="deepgram", model="nova-2", stream=True, language="en")
    assistant.add_task(
        task_type="conversation",
        llm_agent=_llm_agent(),
        transcriber=transcriber,
        enable_textual_input=False,
    )
    task = assistant.tasks[0]
    assert task["tools_config"]["transcriber"]["provider"] == "deepgram"
    assert task["tools_config"]["synthesizer"] is None
    assert task["toolchain"]["pipelines"] == [["transcriber", "llm"]]


def test_completion_check_reuses_the_call_llm_without_an_openai_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_BASE_URL", raising=False)
    main_llm = MagicMock()
    main_llm.model = "llama-3.3-70b-versatile"
    agent = StreamingContextualAgent(main_llm)
    assert agent.conversation_completion_llm is main_llm
    assert agent.voicemail_llm is None


def test_completion_check_stays_on_openai_when_the_key_is_set(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("CHECK_FOR_COMPLETION_LLM", raising=False)
    main_llm = MagicMock()
    main_llm.model = "llama-3.3-70b-versatile"
    completion = completion_llm_for(main_llm)
    assert completion is not main_llm
    assert completion.model == "llama-3.3-70b-versatile"


def test_voicemail_check_is_a_no_when_azure_is_not_configured(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_BASE_URL", raising=False)
    main_llm = MagicMock()
    main_llm.model = "llama-3.3-70b-versatile"
    agent = StreamingContextualAgent(main_llm)
    result, metadata = asyncio.run(agent.check_for_voicemail("please leave a message"))
    assert result == {"is_voicemail": "No"}
    assert metadata == {}
    assert "OPENAI_API_KEY" not in os.environ
