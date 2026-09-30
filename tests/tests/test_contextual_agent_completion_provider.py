"""Tests for provider selection of the check_for_completion / hangup-detection LLM.

Regression for bolna-ai/bolna#224 and #742: when an agent is configured with a
non-OpenAI provider routed through LiteLLM (e.g. Groq), StreamingContextualAgent
must build its conversation_completion_llm with the SAME provider as the main LLM.
Previously it hard-coded OpenAiLLM, so a Groq key (gsk_...) was sent to the OpenAI
API and check_for_completion failed with a 401, and the agent never replied.
"""

from unittest.mock import patch

from bolna.agent_types.contextual_conversational_agent import StreamingContextualAgent
from bolna.llms import OpenAiLLM, LiteLLM


def _stub_init(record):
    """Return an __init__ that records its kwargs instead of doing real work."""

    def _init(self, **kwargs):
        record.append(kwargs)

    return _init


def _make_litellm_stub(model="groq/llama-3.3-70b-versatile"):
    """A lightweight LiteLLM instance (Groq routed via litellm) without network/keys."""
    with patch.object(LiteLLM, "__init__", lambda self, **kw: None):
        llm = LiteLLM.__new__(LiteLLM)
    llm.model = model
    llm.api_key = "gsk_test_groq_key"
    llm.api_base = None
    llm.api_version = None
    return llm


def _build_agent(main_llm, litellm_calls, openai_calls):
    """Construct the agent with both provider __init__s stubbed out (no network/keys)."""
    with (
        patch.object(LiteLLM, "__init__", _stub_init(litellm_calls)),
        patch.object(OpenAiLLM, "__init__", _stub_init(openai_calls)),
    ):
        return StreamingContextualAgent(main_llm)


def test_completion_llm_uses_same_provider_and_credentials_as_main_llm():
    main_llm = _make_litellm_stub()

    litellm_calls, openai_calls = [], []
    agent = _build_agent(main_llm, litellm_calls, openai_calls)

    # The completion LLM must be built with the main LLM's provider class (LiteLLM),
    # NOT a hard-coded OpenAiLLM, so the Groq key is not sent to the OpenAI API.
    assert isinstance(agent.conversation_completion_llm, LiteLLM)
    completion_calls = [c for c in litellm_calls if c.get("model") == "groq/llama-3.3-70b-versatile"]
    assert completion_calls, "completion LLM should be built via the LiteLLM (Groq) provider"
    # And it must carry over the provider key that the main LLM resolved.
    assert completion_calls[0]["llm_key"] == "gsk_test_groq_key"


def test_completion_llm_stays_openai_for_openai_agent():
    with patch.object(OpenAiLLM, "__init__", lambda self, **kw: None):
        main_llm = OpenAiLLM.__new__(OpenAiLLM)
    main_llm.model = "gpt-4o-mini"

    litellm_calls, openai_calls = [], []
    agent = _build_agent(main_llm, litellm_calls, openai_calls)

    assert isinstance(agent.conversation_completion_llm, OpenAiLLM)
    assert any(c.get("model") == "gpt-4o-mini" for c in openai_calls)


def test_completion_model_env_override_is_respected(monkeypatch):
    monkeypatch.setenv("CHECK_FOR_COMPLETION_LLM", "groq/llama-3.1-8b-instant")
    main_llm = _make_litellm_stub()

    litellm_calls, openai_calls = [], []
    _build_agent(main_llm, litellm_calls, openai_calls)

    assert any(c.get("model") == "groq/llama-3.1-8b-instant" for c in litellm_calls)
