"""san0808 round-3 review fixes.

1. The simple-agent standing directive carries the verbatim carve-out (parity with graph).
2. The one-shot "restate your previous line" lives ONLY in the switch-time context note —
   the standing directive is also the call-start pin, where no previous line exists.
3. Bedrock-hosted Claude gets cache_control (litellm translates it to cachePoint); without
   it every decide reprocesses the full rules block and the hedge doubles the miss.
4. Sarvam non-data frames: counted always, logged once per frame type.
5. A parsed `null` reply is a valid decision, not a judge failure — only errored attempts
   count toward the runtime fallback.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bolna.agent_manager.task_manager import TaskManager
from bolna.helpers.language_switcher import DEFAULT_LANGUAGE_SWITCH_LLM, LanguageSwitcher

BEDROCK = "bedrock/global.anthropic.claude-haiku-4-5-20251001-v1:0"

DIRECTIVE = TaskManager._TaskManager__language_directive
CONTEXT_NOTE = TaskManager._TaskManager__switch_context_note


def _tm():
    tm = MagicMock()
    return tm


def test_standing_directive_has_the_verbatim_carve_out():
    text = DIRECTIVE(_tm(), "hi")
    assert "Never translate or alter proper nouns" in text
    assert "alphanumeric identifiers" in text


def test_standing_directive_has_no_repeat_instruction():
    # It is also the call-start pin: there is no "previous line" to repeat there.
    text = DIRECTIVE(_tm(), "hi")
    assert "previous line" not in text
    assert "last line" not in text


def test_switch_note_carries_the_one_shot_restate():
    note = CONTEXT_NOTE(_tm(), "en", "can you speak english")
    assert "restate your previous line" in note
    assert "NEXT reply only" in note


def test_bedrock_claude_system_block_is_cacheable(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    sw = LanguageSwitcher(available_labels=["en", "hi"], model=BEDROCK)
    block = sw._system_message()["content"][0]
    assert block.get("cache_control") == {"type": "ephemeral"}


def test_non_claude_bedrock_model_gets_no_cache_block(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    sw = LanguageSwitcher(available_labels=["en", "hi"], model="bedrock/meta.llama3-70b")
    block = sw._system_message()["content"][0]
    assert "cache_control" not in block


class _FakeWS:
    def __init__(self, frames):
        self.frames = list(frames)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.frames:
            raise StopAsyncIteration
        return self.frames.pop(0)


@pytest.mark.asyncio
async def test_sarvam_non_data_frames_log_once_per_type():
    from bolna.lid.sarvam import SarvamLID

    lid = SarvamLID(on_language=None, config={"sarvam_api_key": "k"})
    lid._ws = _FakeWS(
        [json.dumps({"type": "events", "data": {}})] * 4 + [json.dumps({"type": "error", "message": "quota"})]
    )
    lid._schedule_reconnect = MagicMock()
    with patch("bolna.lid.sarvam.logger") as log:
        await lid._receiver_loop()
    assert lid.unknown_frames == 5
    warnings = [c for c in log.warning.call_args_list if "non-data frame" in str(c)]
    assert len(warnings) == 2  # once for 'events', once for 'error' — not 5


@pytest.mark.asyncio
async def test_parsed_null_is_not_a_judge_failure(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("LANGUAGE_SWITCH_HEDGE_AFTER_S", "0")
    sw = LanguageSwitcher(available_labels=["en", "hi"], model=BEDROCK)
    sw._log_decision = MagicMock()
    sw._llm = MagicMock()
    sw._llm.generate = AsyncMock(return_value="null")  # model validly declining
    for _ in range(4):
        await sw.decide("hello", "", "hi")
    assert sw.model == BEDROCK  # never swapped
    assert sw._consecutive_failures == 0


@pytest.mark.asyncio
async def test_errored_attempts_still_trigger_the_fallback(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("LANGUAGE_SWITCH_HEDGE_AFTER_S", "0.01")
    sw = LanguageSwitcher(available_labels=["en", "hi"], model=BEDROCK)
    sw._log_decision = MagicMock()
    sw._llm = MagicMock()
    sw._llm.generate = AsyncMock(side_effect=Exception("AccessDeniedException"))
    with patch("bolna.helpers.language_switcher.LiteLLM"):
        await sw.decide("hello", "", "hi")
        await sw.decide("hello", "", "hi")
        assert sw.model == f"anthropic/{DEFAULT_LANGUAGE_SWITCH_LLM}"


@pytest.mark.asyncio
async def test_null_resets_an_error_streak(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("LANGUAGE_SWITCH_HEDGE_AFTER_S", "0.01")
    sw = LanguageSwitcher(available_labels=["en", "hi"], model=BEDROCK)
    sw._log_decision = MagicMock()
    sw._llm = MagicMock()
    sw._llm.generate = AsyncMock(side_effect=[Exception("throttled"), Exception("throttled"), "null"])
    # decide 1: both hedge attempts consume the two exceptions? keep it simple — one error decide
    sw._llm.generate = AsyncMock(side_effect=[Exception("throttled"), "null", "null"])
    await sw.decide("hello", "", "hi")  # errored first attempt, hedge parses null → valid decide
    await sw.decide("hello", "", "hi")
    assert sw._consecutive_failures == 0
    assert sw.model == BEDROCK
