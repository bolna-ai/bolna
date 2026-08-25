"""Conversation history carries bookkeeping keys (asr_turn_id, response_uid, turn_id,
message_category) that must never reach a provider.

Only the anthropic-family transformers and the Responses adapter rebuild the payload from
known keys; the nine OpenAI-compatible providers forward the message dict verbatim, so the
LLM adapters strip before sending. These tests pin both halves — what litellm does on its
own, and what bolna actually puts on the wire.
"""

import json

import pytest

from bolna.enums import ChatRole
from bolna.helpers.conversation_history import ConversationHistory
from bolna.llms.message_models import (
    INTERNAL_MESSAGE_KEYS,
    ChatMessage,
    MessageFormatAdapter,
    strip_internal_keys,
)

EXTRA = {"asr_turn_id": 7, "response_uid": "abc"}


def _transform(provider: str, model: str, messages: list[dict] | None = None):
    """Run the provider's real request builder and return the body litellm would send."""
    pytest.importorskip("litellm")
    from litellm.types.utils import LlmProviders
    from litellm.utils import ProviderConfigManager

    config = ProviderConfigManager.get_provider_chat_config(model=model, provider=LlmProviders(provider))
    if config is None:
        pytest.skip(f"{provider} has no chat config in this litellm build")
    return config.transform_request(
        model=model,
        messages=messages if messages is not None else [{"role": "user", "content": "hello", **EXTRA}],
        optional_params={},
        litellm_params={},
        headers={},
    )


def test_append_user_carries_extra_keys():
    history = ConversationHistory()
    history.append_user("hello", **EXTRA)
    assert history.messages[-1] == {"role": ChatRole.USER, "content": "hello", **EXTRA}


def test_append_user_without_kwargs_stays_bare():
    """Synthetic turns (idle-flush, injected prompts) have no ASR turn and must not gain a key."""
    history = ConversationHistory()
    history.append_user("hello")
    assert history.messages[-1] == {"role": ChatRole.USER, "content": "hello"}


def test_chat_message_model_ignores_unknown_keys():
    """The Responses adapter parses through ChatMessage; extra="forbid" would raise here."""
    parsed = ChatMessage(**{"role": "user", "content": "hello", **EXTRA})
    assert parsed.model_dump(exclude_none=True) == {"role": "user", "content": "hello"}


def test_responses_api_input_drops_extra_keys():
    _, items = MessageFormatAdapter.chat_to_responses_input([{"role": "user", "content": "hello", **EXTRA}])
    assert len(items) == 1
    assert set(items[0]) == {"type", "role", "content"}
    assert "asr_turn_id" not in json.dumps(items, default=str)


@pytest.mark.parametrize(
    "provider,model",
    [
        ("anthropic", "claude-3-5-sonnet-20241022"),
        ("bedrock", "anthropic.claude-3-5-sonnet-20240620-v1:0"),
    ],
)
def test_litellm_rebuilding_providers_drop_extra_keys(provider, model):
    """These transformers construct the payload from known keys, so nothing extra survives."""
    assert "asr_turn_id" not in json.dumps(_transform(provider, model), default=str)


@pytest.mark.parametrize(
    "provider,model",
    [
        ("cohere", "command-r"),
        ("groq", "llama-3.1-8b-instant"),
        ("deepseek", "deepseek-chat"),
        ("openrouter", "openai/gpt-4o-mini"),
        ("together_ai", "meta-llama/Llama-3-8b-chat-hf"),
        ("perplexity", "sonar"),
        ("fireworks_ai", "accounts/fireworks/models/llama-v3-8b-instruct"),
        ("deepinfra", "meta-llama/Llama-2-70b-chat-hf"),
        ("vllm", "facebook/opt-125m"),
    ],
)
def test_openai_compatible_providers_would_forward_extra_keys(provider, model):
    """These pass the dict through untouched, which is why bolna strips before calling them.
    Pinned so the day litellm starts sanitising, we notice the guard is redundant."""
    assert "asr_turn_id" in json.dumps(_transform(provider, model), default=str)


@pytest.mark.parametrize(
    "provider,model",
    [
        ("anthropic", "claude-3-5-sonnet-20241022"),
        ("bedrock", "anthropic.claude-3-5-sonnet-20240620-v1:0"),
        ("cohere", "command-r"),
        ("groq", "llama-3.1-8b-instant"),
        ("deepseek", "deepseek-chat"),
        ("openrouter", "openai/gpt-4o-mini"),
        ("together_ai", "meta-llama/Llama-3-8b-chat-hf"),
        ("perplexity", "sonar"),
        ("fireworks_ai", "accounts/fireworks/models/llama-v3-8b-instruct"),
        ("deepinfra", "meta-llama/Llama-2-70b-chat-hf"),
        ("vllm", "facebook/opt-125m"),
    ],
)
def test_nothing_leaks_once_bolna_strips(provider, model):
    """The guarantee that matters: what bolna actually sends carries no bookkeeping keys,
    for every provider, without relying on the server to ignore them."""
    sent = strip_internal_keys([{"role": "user", "content": "hello", **EXTRA}])
    body = json.dumps(_transform(provider, model, messages=sent), default=str)
    assert "asr_turn_id" not in body and "response_uid" not in body


def test_strip_preserves_real_chat_keys():
    """A tool-call turn must keep content=None and tool_calls, or the API rejects the thread."""
    assert strip_internal_keys([{"role": "assistant", "content": None, "tool_calls": [{"id": "x"}], "turn_id": 3}]) == [
        {"role": "assistant", "content": None, "tool_calls": [{"id": "x"}]}
    ]


def test_strip_only_removes_our_own_keys():
    """Denylist, not allowlist — a provider key we don't know about must survive untouched."""
    message = {"role": "user", "content": "hi", "name": "caller", "cache_control": {"type": "ephemeral"}}
    assert strip_internal_keys([{**message, **EXTRA, "message_category": "handoff"}]) == [message]


def test_strip_removes_every_internal_key():
    noisy = {k: "x" for k in INTERNAL_MESSAGE_KEYS}
    assert strip_internal_keys([{"role": "user", "content": "hi", **noisy}]) == [{"role": "user", "content": "hi"}]
