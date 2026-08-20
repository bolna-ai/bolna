"""verbosity is declared on the Llm model, so the API accepts and stores it.

These cover the three paths that build LLM kwargs from a stored agent config, each of
which used to drop the field and leave the model on its own default of "low".
"""

from unittest.mock import MagicMock, patch

from bolna.agent_manager.task_manager import TaskManager
from bolna.agent_types.graph_agent import GraphAgent
from bolna.agent_types.knowledgebase_agent import KnowledgeBaseAgent


def _llm_agent_config(**overrides):
    cfg = {
        "model": "gpt-5.4-mini",
        "max_tokens": 150,
        "provider": "openai",
        "temperature": 1,
        "verbosity": "high",
    }
    cfg.update(overrides)
    return cfg


def _task_config(llm_config):
    return {
        "task_type": "conversation",
        "toolchain": {"execution": "sequential", "pipelines": [["llm"]]},
        "tools_config": {
            "llm_agent": {
                "agent_type": "simple_llm_agent",
                "agent_flow_type": "streaming",
                "llm_config": llm_config,
            },
            "synthesizer": {
                "provider": "elevenlabs",
                "provider_config": {"voice": "Nila", "voice_id": "test", "model": "eleven_turbo_v2_5"},
                "stream": True,
                "buffer_size": 100,
            },
            "transcriber": {
                "provider": "deepgram",
                "model": "nova-3",
                "language": "en",
                "stream": True,
                "encoding": "linear16",
                "sampling_rate": 16000,
                "endpointing": 250,
            },
            "input": {"provider": "default"},
            "output": {"provider": "default"},
        },
        "task_config": {},
    }


def _build_llm_config(llm_agent_config):
    """The llm_config TaskManager hands to the LLM class for a simple_llm_agent."""
    tm = TaskManager("agent", 0, _task_config(llm_agent_config), MagicMock())
    return tm.llm_config


def _captured_llm_kwargs(agent_cls, config):
    """Build the agent's LLM through its own factory and return the kwargs it passed."""
    captured = {}

    def _capture(**kwargs):
        captured.update(kwargs)
        return MagicMock()

    with (
        patch("bolna.agent_types.graph_agent.OpenAI", return_value=MagicMock()),
        patch("bolna.agent_types.graph_agent.OpenAiLLM", return_value=MagicMock()),
        patch(f"{agent_cls.__module__}.SUPPORTED_LLM_PROVIDERS", {"openai": _capture}),
    ):
        agent_cls(config)
    return captured


class TestTaskManagerForwarding:
    async def test_verbosity_is_forwarded(self):
        assert _build_llm_config(_llm_agent_config())["verbosity"] == "high"

    async def test_absent_verbosity_is_not_invented(self):
        cfg = _llm_agent_config()
        cfg.pop("verbosity")
        assert "verbosity" not in _build_llm_config(cfg)

    async def test_existing_gpt5_keys_still_forwarded(self):
        cfg = _llm_agent_config(reasoning_effort="low", reasoning_summary="auto", thinking_budget=512)
        llm_config = _build_llm_config(cfg)
        assert llm_config["reasoning_effort"] == "low"
        assert llm_config["reasoning_summary"] == "auto"
        assert llm_config["thinking_budget"] == 512


class TestOpenAiLLMReceivesVerbosity:
    async def test_verbosity_reaches_the_responses_payload(self):
        from bolna.llms.openai_llm import OpenAiLLM

        llm = OpenAiLLM(
            **_build_llm_config(_llm_agent_config()), llm_key="test-key", base_url="https://api.openai.com/v1"
        )
        create_kwargs, _ = llm._build_responses_create_kwargs(
            [{"role": "user", "content": "hi"}], None, False, None, stream=True
        )
        assert create_kwargs["text"]["verbosity"] == "high"

    async def test_default_is_low_when_unset(self):
        from bolna.llms.openai_llm import OpenAiLLM

        cfg = _llm_agent_config()
        cfg.pop("verbosity")
        llm = OpenAiLLM(**_build_llm_config(cfg), llm_key="test-key", base_url="https://api.openai.com/v1")
        create_kwargs, _ = llm._build_responses_create_kwargs(
            [{"role": "user", "content": "hi"}], None, False, None, stream=True
        )
        assert create_kwargs["text"]["verbosity"] == "low"


class TestAgentPassthrough:
    def test_graph_agent_forwards_verbosity(self):
        config = {
            "agent_information": "Test agent",
            "model": "gpt-5.4-mini",
            "provider": "openai",
            "temperature": 1,
            "max_tokens": 150,
            "verbosity": "high",
            "current_node_id": "start",
            "nodes": [{"id": "start", "prompt": "hi", "edges": []}],
        }
        assert _captured_llm_kwargs(GraphAgent, config).get("verbosity") == "high"

    def test_knowledgebase_agent_forwards_verbosity(self):
        config = {
            "model": "gpt-5.4-mini",
            "provider": "openai",
            "temperature": 1,
            "max_tokens": 150,
            "verbosity": "high",
            "vector_store": {"provider": "lancedb", "vector_id": "test"},
        }
        assert _captured_llm_kwargs(KnowledgeBaseAgent, config).get("verbosity") == "high"


class TestNonOpenAiProvidersUnaffected:
    """Providers with no notion of verbosity must not break now that it is forwarded."""

    def test_litellm_absorbs_verbosity_without_sending_it(self):
        from bolna.llms.litellm import LiteLLM

        llm = LiteLLM(model="claude-sonnet-5", max_tokens=150, temperature=0.1, verbosity="high", llm_key="k")
        assert "verbosity" not in llm.model_args

    def test_gemini_construction_succeeds(self):
        from bolna.llms.gemini_llm import GeminiLLM

        GeminiLLM(model="gemini-2.5-flash", max_tokens=150, temperature=0.1, verbosity="high", llm_key="k")
