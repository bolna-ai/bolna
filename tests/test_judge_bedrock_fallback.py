"""Bedrock judge: IAM-role auth (no API key) + runtime fallback to the API-key judge.

Bedrock permission/throttle errors surface only at INVOKE time, so the construction-time
credential fallback cannot catch them — without a runtime swap every decide fails and the
call loses switching entirely.
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch


from bolna.helpers.language_switcher import (
    DEFAULT_LANGUAGE_SWITCH_LLM,
    LanguageSwitcher,
    resolve_switch_llm_credentials,
)

BEDROCK = "bedrock/global.anthropic.claude-haiku-4-5-20251001-v1:0"
DEFAULT = f"anthropic/{DEFAULT_LANGUAGE_SWITCH_LLM}"


def test_bedrock_resolves_no_api_key():
    key, _, _ = resolve_switch_llm_credentials(BEDROCK)
    assert key == ""


def test_bedrock_judge_is_not_treated_as_dead(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    sw = LanguageSwitcher(available_labels=["en", "hi"], model=BEDROCK)
    assert sw.model == BEDROCK  # NOT swapped at construction
    assert sw.has_credentials is True  # IAM role — the tool must not be re-injected


def test_bedrock_survives_when_no_anthropic_key_exists(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("LANGUAGE_SWITCH_LLM_API_KEY", raising=False)
    sw = LanguageSwitcher(available_labels=["en", "hi"], model=BEDROCK)
    assert sw.model == BEDROCK


async def test_runtime_fallback_after_consecutive_failures(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("LANGUAGE_SWITCH_HEDGE_AFTER_S", "0")
    sw = LanguageSwitcher(available_labels=["en", "hi"], model=BEDROCK)
    sw._log_decision = MagicMock()
    sw._llm = MagicMock()
    sw._llm.generate = AsyncMock(side_effect=Exception("AccessDeniedException"))

    with patch("bolna.helpers.language_switcher.LiteLLM"):
        await sw.decide("hello", "", "hi")
        assert sw.model == BEDROCK  # 1 failure: still on bedrock
        await sw.decide("hello", "", "hi")
        assert sw.model == DEFAULT  # 2nd failure: swapped to the API-key judge
    assert sw._runtime_fallback_done is True


async def test_success_resets_the_failure_counter(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("LANGUAGE_SWITCH_HEDGE_AFTER_S", "0")
    sw = LanguageSwitcher(available_labels=["en", "hi"], model=BEDROCK)
    sw._log_decision = MagicMock()
    sw._llm = MagicMock()
    sw._llm.generate = AsyncMock(side_effect=[Exception("throttled"), '{"target_language": null}'])
    await sw.decide("hello", "", "hi")
    await sw.decide("hello", "", "hi")
    assert sw._consecutive_failures == 0
    assert sw.model == BEDROCK  # an isolated blip must not cost the faster judge


async def test_no_fallback_without_an_anthropic_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("LANGUAGE_SWITCH_LLM_API_KEY", raising=False)
    monkeypatch.setenv("LANGUAGE_SWITCH_HEDGE_AFTER_S", "0")
    sw = LanguageSwitcher(available_labels=["en", "hi"], model=BEDROCK)
    sw._log_decision = MagicMock()
    sw._llm = MagicMock()
    sw._llm.generate = AsyncMock(side_effect=Exception("AccessDeniedException"))
    for _ in range(4):
        await sw.decide("hello", "", "hi")
    assert sw.model == BEDROCK  # nowhere to fall back to; stay put


def test_anthropic_default_is_unchanged(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    sw = LanguageSwitcher(available_labels=["en", "hi"])
    assert sw.model == DEFAULT
    assert sw.has_credentials is True
    assert os.getenv("ANTHROPIC_API_KEY") == "k"


def test_bedrock_claude_system_block_is_cacheable(monkeypatch):
    # Without cache_control the full rules block reprocesses on every decide and the
    # hedge doubles the miss; litellm translates it to Bedrock cachePoint for claude ids.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    sw = LanguageSwitcher(available_labels=["en", "hi"], model=BEDROCK)
    block = sw._system_message()["content"][0]
    assert block.get("cache_control") == {"type": "ephemeral"}


def test_non_claude_bedrock_model_gets_no_cache_block(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    sw = LanguageSwitcher(available_labels=["en", "hi"], model="bedrock/meta.llama3-70b")
    block = sw._system_message()["content"][0]
    assert "cache_control" not in block


async def test_parsed_null_is_not_a_judge_failure(monkeypatch):
    # json.loads("null") → None with no exception: the model validly declining,
    # not a dead judge — it must never count toward the runtime fallback.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("LANGUAGE_SWITCH_HEDGE_AFTER_S", "0")
    sw = LanguageSwitcher(available_labels=["en", "hi"], model=BEDROCK)
    sw._log_decision = MagicMock()
    sw._llm = MagicMock()
    sw._llm.generate = AsyncMock(return_value="null")
    for _ in range(4):
        await sw.decide("hello", "", "hi")
    assert sw.model == BEDROCK
    assert sw._consecutive_failures == 0


async def test_errored_attempts_still_trigger_the_fallback_when_hedged(monkeypatch):
    # With hedging on, a dead judge returns None instead of raising (per-attempt exceptions
    # are swallowed) — the errored-None path must still count as failure.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("LANGUAGE_SWITCH_HEDGE_AFTER_S", "0.01")
    sw = LanguageSwitcher(available_labels=["en", "hi"], model=BEDROCK)
    sw._log_decision = MagicMock()
    sw._llm = MagicMock()
    sw._llm.generate = AsyncMock(side_effect=Exception("AccessDeniedException"))
    with patch("bolna.helpers.language_switcher.LiteLLM"):
        await sw.decide("hello", "", "hi")
        await sw.decide("hello", "", "hi")
        assert sw.model == DEFAULT


async def test_null_resets_an_error_streak(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("LANGUAGE_SWITCH_HEDGE_AFTER_S", "0.01")
    sw = LanguageSwitcher(available_labels=["en", "hi"], model=BEDROCK)
    sw._log_decision = MagicMock()
    sw._llm = MagicMock()
    sw._llm.generate = AsyncMock(side_effect=[Exception("throttled"), "null", "null"])
    await sw.decide("hello", "", "hi")  # errored first attempt; hedge parses null → valid decide
    await sw.decide("hello", "", "hi")
    assert sw._consecutive_failures == 0
    assert sw.model == BEDROCK


def test_bedrock_region_defaults_to_ap_south_1(monkeypatch):
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    sw = LanguageSwitcher(available_labels=["en", "hi"], model=BEDROCK)
    assert sw._llm.model_args.get("aws_region_name") == "ap-south-1"


def test_bedrock_region_env_override_wins(monkeypatch):
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    sw = LanguageSwitcher(available_labels=["en", "hi"], model=BEDROCK)
    assert sw._llm.model_args.get("aws_region_name") == "us-east-1"


def test_non_bedrock_judge_carries_no_region(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    sw = LanguageSwitcher(available_labels=["en", "hi"])
    assert "aws_region_name" not in sw._llm.model_args
