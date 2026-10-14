"""A custom LLM's extra_body reaches every request sent to that endpoint, and nowhere else."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from bolna.agent_types.graph_agent import GraphAgent
from bolna.agent_types.knowledgebase_agent import KnowledgeBaseAgent
from bolna.llms.openai_llm import OpenAiLLM
from bolna.models import Llm

from tests.test_llm_verbosity_passthrough import _build_llm_config

EXTRA_BODY = {"chat_template_kwargs": {"enable_thinking": False}}
MESSAGES = [{"role": "user", "content": "hi"}]


def _custom_llm(**kwargs):
    llm = OpenAiLLM(
        model="qwen3-8b",
        provider="custom",
        base_url="https://llm.example.com/v1",
        llm_key="test-key",
        extra_body=EXTRA_BODY,
        **kwargs,
    )
    llm._base_url_validated = True
    llm.async_client = MagicMock()
    llm.async_client.chat.completions.create = AsyncMock(side_effect=RuntimeError("stop"))
    return llm


class TestValidation:
    def test_custom_provider_accepts_extra_body(self):
        assert Llm(provider="custom", model="qwen3-8b", extra_body=EXTRA_BODY).extra_body == EXTRA_BODY

    def test_platform_provider_rejects_extra_body(self):
        with pytest.raises(ValidationError, match="only supported for custom"):
            Llm(provider="openai", model="gpt-4.1-mini", extra_body=EXTRA_BODY)

    @pytest.mark.parametrize("key", ["model", "messages", "stream", "tools"])
    def test_reserved_request_fields_are_rejected(self, key):
        with pytest.raises(ValidationError, match=key):
            Llm(provider="custom", model="qwen3-8b", extra_body={key: "x"})


class TestRequestShape:
    async def test_streaming_turn_carries_extra_body(self):
        llm = _custom_llm()
        with pytest.raises(RuntimeError):
            async for _ in llm._generate_stream_chat(MESSAGES):
                pass
        sent = llm.async_client.chat.completions.create.call_args.kwargs
        assert sent["extra_body"] == EXTRA_BODY
        assert "service_tier" not in sent

    async def test_non_streaming_call_carries_extra_body(self):
        llm = _custom_llm()
        with pytest.raises(RuntimeError):
            await llm.generate(MESSAGES)
        assert llm.async_client.chat.completions.create.call_args.kwargs["extra_body"] == EXTRA_BODY

    async def test_routing_call_carries_extra_body(self):
        llm = _custom_llm()
        with pytest.raises(RuntimeError):
            await llm.route(MESSAGES, tools=[])
        assert llm.async_client.chat.completions.create.call_args.kwargs["extra_body"] == EXTRA_BODY

    def test_platform_provider_never_sends_extra_body(self):
        llm = OpenAiLLM(model="gpt-4.1-mini", llm_key="test-key", extra_body=EXTRA_BODY)
        assert "extra_body" not in llm.model_args
        assert llm.model_args["service_tier"] == "default"


class TestPassthrough:
    async def test_task_manager_forwards_extra_body(self):
        cfg = {"model": "qwen3-8b", "max_tokens": 150, "provider": "custom", "temperature": 0.2}
        assert _build_llm_config({**cfg, "extra_body": EXTRA_BODY})["extra_body"] == EXTRA_BODY

    def test_graph_agent_sends_it_on_conversation_routing_and_aux_llms(self):
        conversation = MagicMock()
        openai_llm = MagicMock()
        config = {
            "agent_information": "Test agent",
            "model": "qwen3-8b",
            "provider": "custom",
            "base_url": "https://llm.example.com/v1",
            "llm_key": "test-key",
            "extra_body": EXTRA_BODY,
            "current_node_id": "start",
            "nodes": [{"id": "start", "prompt": "hi", "edges": []}],
        }
        with (
            patch("bolna.agent_types.graph_agent.OpenAiLLM", openai_llm),
            patch("bolna.agent_types.graph_agent.SUPPORTED_LLM_PROVIDERS", {"custom": conversation}),
        ):
            GraphAgent(config)

        built = [call.kwargs for call in conversation.call_args_list + openai_llm.call_args_list]
        # conversation, routing, completion check, voicemail
        assert len(built) == 4
        assert all(kwargs["extra_body"] == EXTRA_BODY for kwargs in built)
        assert all(kwargs["provider"] == "custom" for kwargs in built)

    def test_knowledgebase_agent_forwards_extra_body(self):
        captured = {}
        config = {
            "model": "qwen3-8b",
            "provider": "custom",
            "extra_body": EXTRA_BODY,
            "vector_store": {"provider": "lancedb", "vector_id": "test"},
        }
        with patch(
            "bolna.agent_types.knowledgebase_agent.SUPPORTED_LLM_PROVIDERS",
            {"custom": lambda **kwargs: captured.update(kwargs) or MagicMock()},
        ):
            KnowledgeBaseAgent(config)
        assert captured["extra_body"] == EXTRA_BODY
