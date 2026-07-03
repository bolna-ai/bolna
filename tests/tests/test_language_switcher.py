"""Unit tests for LanguageSwitcher (LiteLLM mocked — no network)."""

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
    # Both transcripts + active language must reach the per-turn user message.
    prompt_text = fake_llm.generate.await_args.args[0][-1]["content"]
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
    # Anthropic requires a user message; a system-only list errors. Guard the shape.
    switcher, fake_llm = _make_switcher(json.dumps({"target_language": None}))
    await switcher.decide("hello", "", active_label="en")
    sent_messages = fake_llm.generate.await_args.args[0]
    assert [m["role"] for m in sent_messages] == ["system", "user"]


@pytest.mark.asyncio
async def test_default_model_is_haiku_with_provider_prefix():
    # Bare claude-* names break on older litellm; the switcher must namespace them.
    switcher, _ = _make_switcher(json.dumps({"target_language": None}))
    assert switcher.model == "anthropic/claude-haiku-4-5-20251001"


@pytest.mark.asyncio
async def test_bare_claude_model_gets_anthropic_prefix():
    from bolna.helpers.language_switcher import LanguageSwitcher

    with patch(f"{MOD}.LiteLLM", return_value=MagicMock()):
        switcher = LanguageSwitcher(available_labels=["en"], model="claude-haiku-4-5-20251001")
    assert switcher.model == "anthropic/claude-haiku-4-5-20251001"


@pytest.mark.asyncio
async def test_provider_prefixed_model_left_untouched():
    from bolna.helpers.language_switcher import LanguageSwitcher

    with patch(f"{MOD}.LiteLLM", return_value=MagicMock()):
        prefixed = LanguageSwitcher(available_labels=["en"], model="anthropic/claude-haiku-4-5-20251001")
        non_claude = LanguageSwitcher(available_labels=["en"], model="azure/gpt-4.1-mini")
    assert prefixed.model == "anthropic/claude-haiku-4-5-20251001"
    assert non_claude.model == "azure/gpt-4.1-mini"


CRED_ENVS = [
    "LANGUAGE_SWITCH_LLM_API_KEY",
    "LANGUAGE_SWITCH_LLM_API_BASE",
    "LANGUAGE_SWITCH_LLM_API_VERSION",
    "ANTHROPIC_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_VERSION",
    "OPENAI_API_KEY",
]


def _clear_cred_envs(monkeypatch):
    for name in CRED_ENVS:
        monkeypatch.delenv(name, raising=False)


def test_credentials_claude_falls_back_to_anthropic(monkeypatch):
    from bolna.helpers.language_switcher import resolve_switch_llm_credentials

    _clear_cred_envs(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-key")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "az-key")  # must not leak into the claude path
    assert resolve_switch_llm_credentials("anthropic/claude-haiku-4-5-20251001") == ("ant-key", "", "")


def test_credentials_azure_falls_back_to_azure_openai_envs(monkeypatch):
    # azure/* picks up AZURE_OPENAI_* (not the Anthropic key, which would 401).
    from bolna.helpers.language_switcher import resolve_switch_llm_credentials

    _clear_cred_envs(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-key")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "az-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://ptu.example.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2024-06-01")
    assert resolve_switch_llm_credentials("azure/ptu-gpt-4-1-mini") == (
        "az-key",
        "https://ptu.example.azure.com",
        "2024-06-01",
    )


def test_credentials_openai_style_falls_back_to_openai(monkeypatch):
    from bolna.helpers.language_switcher import resolve_switch_llm_credentials

    _clear_cred_envs(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "oai-key")
    assert resolve_switch_llm_credentials("gpt-4.1-nano") == ("oai-key", "", "")


def test_credentials_azure_version_defaults_like_azure_llm(monkeypatch):
    # AZURE_OPENAI_API_VERSION unset → default (not ""), matching azure_llm.py, else every decide fails.
    from bolna.helpers.language_switcher import resolve_switch_llm_credentials

    _clear_cred_envs(monkeypatch)
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "az-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://ptu.example.azure.com")
    _, _, version = resolve_switch_llm_credentials("azure/ptu-gpt-4-1-mini")
    assert version == "2024-12-01-preview"


def test_misconfigured_azure_judge_falls_back_to_default(monkeypatch):
    # azure flag granted but AZURE_OPENAI_* missing → don't ship a dead judge; use the default.
    from bolna.helpers.language_switcher import LanguageSwitcher

    _clear_cred_envs(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-key")  # default judge's key is present
    with patch(f"{MOD}.LiteLLM", return_value=MagicMock()) as fake_cls:
        sw = LanguageSwitcher(available_labels=["en", "te"], model="azure/ptu-gpt-4-1-mini")
    assert sw.model == "anthropic/claude-haiku-4-5-20251001"
    assert fake_cls.call_args.kwargs["llm_key"] == "ant-key"


def test_credentials_dedicated_envs_always_win(monkeypatch):
    from bolna.helpers.language_switcher import resolve_switch_llm_credentials

    _clear_cred_envs(monkeypatch)
    monkeypatch.setenv("LANGUAGE_SWITCH_LLM_API_KEY", "dedicated-key")
    monkeypatch.setenv("LANGUAGE_SWITCH_LLM_API_BASE", "https://dedicated.example.com")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "az-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://global.azure.com")
    key, base, version = resolve_switch_llm_credentials("azure/ptu-gpt-4-1-mini")
    assert (key, base) == ("dedicated-key", "https://dedicated.example.com")


@pytest.mark.asyncio
async def test_decide_splits_static_system_and_dynamic_user():
    switcher, fake_llm = _make_switcher(json.dumps({"target_language": None}), labels=("hi", "te"))
    await switcher.decide("హలో", "garbled", active_label="te")

    messages = fake_llm.generate.await_args.args[0]
    assert [m["role"] for m in messages] == ["system", "user"]
    system_block = messages[0]["content"][0]
    # Claude default → Anthropic cache annotation on the static prefix.
    assert system_block["cache_control"] == {"type": "ephemeral"}
    # Static block carries the rules and NO per-turn data (else the prefix never caches).
    assert "CODE-MIXING IS NOT A SWITCH" in system_block["text"]
    assert "హలో" not in system_block["text"]
    # Dynamic user message carries the per-turn data.
    assert "హలో" in messages[1]["content"] and "garbled" in messages[1]["content"]
    assert "te" in messages[1]["content"]


@pytest.mark.asyncio
async def test_decide_azure_model_gets_no_anthropic_annotation():
    from bolna.helpers.language_switcher import LanguageSwitcher

    fake_llm = MagicMock()
    fake_llm.generate = AsyncMock(return_value=json.dumps({"target_language": None}))
    with patch(f"{MOD}.LiteLLM", return_value=fake_llm):
        switcher = LanguageSwitcher(available_labels=["hi"], model="azure/ptu-gpt-4-1-mini")
    await switcher.decide("hello", "", active_label="hi")
    system_block = fake_llm.generate.await_args.args[0][0]["content"][0]
    # Azure/OpenAI cache prefixes automatically; cache_control would be a foreign key.
    assert "cache_control" not in system_block


def test_prompt_variants_rules_stay_in_sync():
    # The rules are duplicated across both prompt variants; pin each rule's marker in both.
    from bolna.prompts import LANGUAGE_SWITCH_PROMPT, LANGUAGE_SWITCH_SYSTEM_PROMPT

    markers = [
        "INTENT ABOUT A NAMED LANGUAGE",
        "judge the MATRIX language",
        "NUMBERS, CODES, AND IDENTIFIERS ARE NOT LANGUAGE EVIDENCE",
        "CLOSELY RELATED OR ACOUSTICALLY CONFUSABLE LANGUAGES",
        "the majority frame wins",
        "flip SCRIPT mid-turn",
        "Judge the language by the words, not the script",
        "CODE-MIXING IS NOT A SWITCH",
        "one function word does not create a matrix",
        "explicit_request",
        "target_confidence",
        "detection_confidence",
    ]
    for marker in markers:
        assert marker in LANGUAGE_SWITCH_PROMPT, f"missing in base prompt: {marker}"
        assert marker in LANGUAGE_SWITCH_SYSTEM_PROMPT, f"missing in cache variant: {marker}"


def test_azure_switcher_wires_credentials_into_litellm(monkeypatch):
    from bolna.helpers.language_switcher import LanguageSwitcher

    _clear_cred_envs(monkeypatch)
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "az-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://ptu.example.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2024-06-01")
    with patch(f"{MOD}.LiteLLM", return_value=MagicMock()) as fake_cls:
        LanguageSwitcher(available_labels=["en", "te"], model="azure/ptu-gpt-4-1-mini")
    kwargs = fake_cls.call_args.kwargs
    assert kwargs["llm_key"] == "az-key"
    assert kwargs["base_url"] == "https://ptu.example.azure.com"
    assert kwargs["api_version"] == "2024-06-01"


@pytest.mark.asyncio
async def test_prewarm_fires_one_llm_call_and_swallows_errors():
    switcher, fake_llm = _make_switcher("ok")
    await switcher.prewarm()  # await the returned task — deterministic, no yield-counting
    fake_llm.generate.assert_awaited_once()

    # A failing prewarm must never propagate.
    switcher2, fake_llm2 = _make_switcher("ok")
    fake_llm2.generate.side_effect = RuntimeError("boom")
    await switcher2.prewarm()  # _warm swallows the error internally, so this won't raise
