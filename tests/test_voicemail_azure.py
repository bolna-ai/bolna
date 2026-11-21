"""Voicemail detection always runs on the platform Azure OpenAI deployment, and an agent without one does not start."""

from types import SimpleNamespace

import pytest

from bolna.agent_types.contextual_conversational_agent import StreamingContextualAgent
from bolna.llms import AzureLLM


def _agent():
    return StreamingContextualAgent(SimpleNamespace(model="gpt-4.1-mini"))


def test_voicemail_runs_on_the_platform_azure_deployment(monkeypatch):
    monkeypatch.delenv("VOICEMAIL_DETECTION_LLM", raising=False)
    monkeypatch.setenv("AZURE_OPENAI_BASE_URL", "https://platform.azure.example")
    llm = _agent().voicemail_llm
    assert isinstance(llm, AzureLLM)
    assert llm.model == "gpt-4.1-mini"
    assert llm.llm_host == "platform.azure.example"


def test_the_endpoint_falls_back_to_azure_openai_endpoint(monkeypatch):
    monkeypatch.delenv("AZURE_OPENAI_BASE_URL")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://endpoint.azure.example")
    assert _agent().voicemail_llm.llm_host == "endpoint.azure.example"


def test_without_an_azure_endpoint_the_agent_does_not_start(monkeypatch):
    monkeypatch.delenv("AZURE_OPENAI_BASE_URL")
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    with pytest.raises(Exception):
        _agent()
