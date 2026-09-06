"""prompt_cache_key is the routing hint that keeps a call's turns on one server.

It rides in kwargs rather than the stored agent config, so it crosses four boundaries that
each used to drop unknown keys silently: the two TaskManager injected_cfg blocks, the graph
and knowledgebase agents' own kwarg allowlists, and model_args, which every chat-completions
call spreads. A drop at any of them is invisible at runtime and only shows up as a cold cache.
"""

from unittest.mock import MagicMock, patch

import pytest

from bolna.agent_manager.task_manager import TaskManager
from bolna.agent_types.graph_agent import GraphAgent
from bolna.agent_types.knowledgebase_agent import KnowledgeBaseAgent

KEY = "agent-4f2b1c90"


def _task_config(agent_type="simple_llm_agent"):
    return {
        "task_type": "conversation",
        "toolchain": {"execution": "sequential", "pipelines": [["llm"]]},
        "tools_config": {
            "llm_agent": {
                "agent_type": agent_type,
                "agent_flow_type": "streaming",
                "llm_config": {
                    "model": "gpt-4.1-mini",
                    "max_tokens": 150,
                    "provider": "openai",
                    "temperature": 0.2,
                },
            },
            "synthesizer": {
                "provider": "elevenlabs",
                "provider_config": {
                    "voice": "Nila",
                    "voice_id": "test",
                    "model": "eleven_turbo_v2_5",
                    "synthesizer_key": "test-key",
                },
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


def _openai_llm(**kwargs):
    from bolna.llms.openai_llm import OpenAiLLM

    return OpenAiLLM(
        model="gpt-4.1-mini",
        max_tokens=150,
        temperature=0.2,
        llm_key="test-key",
        base_url="https://api.openai.com/v1",
        **kwargs,
    )


def _azure_llm(**kwargs):
    from bolna.llms.azure_llm import AzureLLM

    return AzureLLM(
        model="ptu-gpt-4-1-mini",
        max_tokens=150,
        temperature=0.2,
        llm_key="test-key",
        base_url="https://example.openai.azure.com",
        **kwargs,
    )


def _captured_llm_kwargs(agent_cls, config):
    """Build the agent's LLM through its own factory and return the kwargs it passed."""
    captured = {}

    def _capture(**kwargs):
        captured.update(kwargs)
        return MagicMock()

    with (
        patch("bolna.agent_types.graph_agent.OpenAiLLM", return_value=MagicMock()),
        patch(f"{agent_cls.__module__}.SUPPORTED_LLM_PROVIDERS", {"openai": _capture}),
    ):
        agent_cls(config)
    return captured


class TestModelArgs:
    """model_args is spread into every chat-completions call, including route and PTU overflow."""

    @pytest.mark.parametrize("build", [_openai_llm, _azure_llm], ids=["openai", "azure"])
    def test_key_reaches_model_args(self, build):
        assert build(prompt_cache_key=KEY).model_args["prompt_cache_key"] == KEY

    @pytest.mark.parametrize("build", [_openai_llm, _azure_llm], ids=["openai", "azure"])
    def test_absent_key_sends_nothing(self, build):
        assert "prompt_cache_key" not in build().model_args

    @pytest.mark.parametrize("build", [_openai_llm, _azure_llm], ids=["openai", "azure"])
    def test_empty_key_sends_nothing(self, build):
        """An empty string is a valid dict value but not a valid routing key."""
        assert "prompt_cache_key" not in build(prompt_cache_key="").model_args


class TestResponsesPayload:
    def test_key_reaches_the_responses_payload(self):
        llm = _openai_llm(prompt_cache_key=KEY, use_responses_api=True)
        create_kwargs, _ = llm._build_responses_create_kwargs(
            [{"role": "user", "content": "hi"}], None, False, None, stream=True
        )
        assert create_kwargs["prompt_cache_key"] == KEY

    def test_absent_key_is_not_invented(self):
        llm = _openai_llm(use_responses_api=True)
        create_kwargs, _ = llm._build_responses_create_kwargs(
            [{"role": "user", "content": "hi"}], None, False, None, stream=True
        )
        assert "prompt_cache_key" not in create_kwargs


class TestTaskManagerForwarding:
    """Graph and knowledgebase agents get their config through injected_cfg, not **kwargs."""

    @pytest.mark.parametrize("agent_type", ["graph_agent", "knowledgebase_agent"])
    async def test_injected_cfg_carries_the_key(self, agent_type):
        tm = TaskManager("agent", 0, _task_config(agent_type), MagicMock(), prompt_cache_key=KEY)
        captured = {}

        def _capture(config):
            captured.update(config)
            return MagicMock()

        with (
            patch("bolna.agent_manager.task_manager.GraphAgent", _capture),
            patch("bolna.agent_manager.task_manager.KnowledgeBaseAgent", _capture),
        ):
            tm._TaskManager__get_agent_object(MagicMock(), agent_type)
        assert captured["prompt_cache_key"] == KEY

    @pytest.mark.parametrize("agent_type", ["graph_agent", "knowledgebase_agent"])
    async def test_absent_key_is_not_invented(self, agent_type):
        tm = TaskManager("agent", 0, _task_config(agent_type), MagicMock())
        captured = {}

        def _capture(config):
            captured.update(config)
            return MagicMock()

        with (
            patch("bolna.agent_manager.task_manager.GraphAgent", _capture),
            patch("bolna.agent_manager.task_manager.KnowledgeBaseAgent", _capture),
        ):
            tm._TaskManager__get_agent_object(MagicMock(), agent_type)
        assert "prompt_cache_key" not in captured


class TestAgentPassthrough:
    """Each agent filters its config through a second allowlist before reaching the LLM."""

    def test_graph_agent_forwards_the_key(self):
        config = {
            "agent_information": "Test agent",
            "model": "gpt-4.1-mini",
            "provider": "openai",
            "temperature": 0.2,
            "max_tokens": 150,
            "prompt_cache_key": KEY,
            "current_node_id": "start",
            "nodes": [{"id": "start", "prompt": "hi", "edges": []}],
        }
        assert _captured_llm_kwargs(GraphAgent, config).get("prompt_cache_key") == KEY

    def test_knowledgebase_agent_forwards_the_key(self):
        config = {
            "model": "gpt-4.1-mini",
            "provider": "openai",
            "temperature": 0.2,
            "max_tokens": 150,
            "prompt_cache_key": KEY,
            "vector_store": {"provider": "lancedb", "vector_id": "test"},
        }
        assert _captured_llm_kwargs(KnowledgeBaseAgent, config).get("prompt_cache_key") == KEY
