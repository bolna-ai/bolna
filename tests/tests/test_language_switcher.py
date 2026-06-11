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
async def test_decide_returns_target_and_confidence_list():
    payload = json.dumps(
        {
            "languages": [{"language": "hi", "confidence": 0.9}, {"language": "en", "confidence": 0.1}],
            "target_language": "hi",
            "reasoning": "caller spoke Hindi",
        }
    )
    switcher, fake_llm = _make_switcher(payload)
    result = await switcher.decide("aap kaise hain", "up cause an", active_label="en")
    assert result["target_language"] == "hi"
    assert result["languages"][0] == {"language": "hi", "confidence": 0.9}
    fake_llm.generate.assert_awaited_once()
    # Both transcripts + active language must reach the prompt.
    prompt_text = fake_llm.generate.await_args.args[0][0]["content"]
    assert "aap kaise hain" in prompt_text  # unbiased detector transcript
    assert "up cause an" in prompt_text  # live language-locked transcript
    assert "en" in prompt_text


@pytest.mark.asyncio
async def test_decide_returns_null_target_to_stay():
    switcher, _ = _make_switcher(json.dumps({"languages": [], "target_language": None, "reasoning": "still English"}))
    result = await switcher.decide("hello there", "hello there", active_label="en")
    assert result["target_language"] is None


@pytest.mark.asyncio
async def test_decide_empty_detector_transcript_skips_llm():
    switcher, fake_llm = _make_switcher(json.dumps({"target_language": "hi"}))
    # Empty unbiased transcript → no decision even if the live transcript has content.
    result = await switcher.decide("   ", "some live text", active_label="en")
    assert result is None
    fake_llm.generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_decide_bad_json_returns_none():
    switcher, _ = _make_switcher("not-json")
    result = await switcher.decide("something", "something", active_label="en")
    assert result is None


@pytest.mark.asyncio
async def test_decide_tolerates_markdown_fenced_json():
    # Claude often wraps JSON in ```json ... ``` fences; we must still parse it.
    fenced = '```json\n{"target_language": "hi", "reasoning": "switched"}\n```'
    switcher, _ = _make_switcher(fenced)
    result = await switcher.decide("kuch baat", "", active_label="en")
    assert result == {"target_language": "hi", "reasoning": "switched"}


@pytest.mark.asyncio
async def test_decide_sends_user_role_message():
    # Anthropic requires a user message; a system-only list errors. Guard the role.
    switcher, fake_llm = _make_switcher(json.dumps({"target_language": None}))
    await switcher.decide("hello", "", active_label="en")
    sent_messages = fake_llm.generate.await_args.args[0]
    assert sent_messages[0]["role"] == "user"


@pytest.mark.asyncio
async def test_default_model_is_sonnet():
    switcher, _ = _make_switcher(json.dumps({"target_language": None}))
    assert switcher.model == "claude-sonnet-4-6"


@pytest.mark.asyncio
async def test_prewarm_fires_one_llm_call_and_swallows_errors():
    import asyncio

    switcher, fake_llm = _make_switcher("ok")
    switcher.prewarm()
    await asyncio.sleep(0)  # let the fire-and-forget task run
    fake_llm.generate.assert_awaited_once()

    # A failing prewarm must never propagate.
    switcher2, fake_llm2 = _make_switcher("ok")
    fake_llm2.generate.side_effect = RuntimeError("boom")
    switcher2.prewarm()
    await asyncio.sleep(0)
