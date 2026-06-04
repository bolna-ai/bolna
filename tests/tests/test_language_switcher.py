"""Unit tests for LanguageSwitcher — the dedicated LLM that decides language
switches from the unbiased Saaras v3 detector transcript.

LiteLLM is mocked so no network/credentials are needed.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

MOD = "bolna.helpers.language_switcher"


def _make_switcher(generate_return, labels=("en", "hi")):
    """Build a LanguageSwitcher whose underlying LiteLLM.generate is mocked."""
    from bolna.helpers.language_switcher import LanguageSwitcher

    fake_llm = MagicMock()
    fake_llm.generate = AsyncMock(return_value=generate_return)
    with patch(f"{MOD}.LiteLLM", return_value=fake_llm):
        switcher = LanguageSwitcher(available_labels=list(labels))
    return switcher, fake_llm


@pytest.mark.asyncio
async def test_decide_returns_target_when_llm_picks_supported_language():
    switcher, fake_llm = _make_switcher(json.dumps({"target_language": "hi", "reasoning": "caller spoke Hindi"}))
    result = await switcher.decide("aap kaise hain", active_label="en")
    assert result == {"target_language": "hi", "reasoning": "caller spoke Hindi"}
    fake_llm.generate.assert_awaited_once()
    # The decision prompt must carry the transcript + active language.
    sent_messages = fake_llm.generate.await_args.args[0]
    prompt_text = sent_messages[0]["content"]
    assert "aap kaise hain" in prompt_text
    assert "en" in prompt_text


@pytest.mark.asyncio
async def test_decide_returns_null_target_to_stay():
    switcher, _ = _make_switcher(json.dumps({"target_language": None, "reasoning": "still English"}))
    result = await switcher.decide("hello there", active_label="en")
    assert result["target_language"] is None


@pytest.mark.asyncio
async def test_decide_empty_transcript_skips_llm():
    switcher, fake_llm = _make_switcher(json.dumps({"target_language": "hi"}))
    result = await switcher.decide("   ", active_label="en")
    assert result is None
    fake_llm.generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_decide_bad_json_returns_none():
    switcher, _ = _make_switcher("not-json")
    result = await switcher.decide("something", active_label="en")
    assert result is None


@pytest.mark.asyncio
async def test_decide_tolerates_markdown_fenced_json():
    # Claude often wraps JSON in ```json ... ``` fences; we must still parse it.
    fenced = '```json\n{"target_language": "hi", "reasoning": "switched"}\n```'
    switcher, _ = _make_switcher(fenced)
    result = await switcher.decide("kuch baat", active_label="en")
    assert result == {"target_language": "hi", "reasoning": "switched"}


@pytest.mark.asyncio
async def test_decide_sends_user_role_message():
    # Anthropic requires a user message; a system-only list errors. Guard the role.
    switcher, fake_llm = _make_switcher(json.dumps({"target_language": None}))
    await switcher.decide("hello", active_label="en")
    sent_messages = fake_llm.generate.await_args.args[0]
    assert sent_messages[0]["role"] == "user"


@pytest.mark.asyncio
async def test_default_model_is_sonnet():
    switcher, _ = _make_switcher(json.dumps({"target_language": None}))
    assert switcher.model == "claude-sonnet-4-6"
