"""Hedged decide: a second identical request cuts the tail without raising the decide timeout.

The judge's slowness is a per-request tail rather than a slow model, so a fresh request usually
beats the straggler. The common turn must never pay for two.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock


from bolna.helpers.language_switcher import DEFAULT_HEDGE_AFTER_S, LanguageSwitcher

REPLY = json.dumps({"target_language": "mr", "target_confidence": 0.9, "reasoning": "Marathi"})


def _switcher(monkeypatch, generate, hedge_after=0.05):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("LANGUAGE_SWITCH_HEDGE_AFTER_S", str(hedge_after))
    sw = LanguageSwitcher(available_labels=["hi", "mr"], run_id="r1")
    sw._llm = MagicMock()
    sw._llm.generate = generate
    sw._log_decision = MagicMock()
    return sw


async def _decide(sw):
    return await sw.decide("mala samajla nahi", "garbled", "hi")


async def test_fast_decide_fires_only_one_request(monkeypatch):
    gen = AsyncMock(return_value=REPLY)
    sw = _switcher(monkeypatch, gen, hedge_after=0.5)
    assert (await _decide(sw))["target_language"] == "mr"
    assert gen.await_count == 1  # the common turn must not pay for two
    assert sw.hedge_won is False


async def test_slow_first_request_is_hedged_and_second_wins(monkeypatch):
    calls = {"n": 0}

    async def generate(_messages):
        calls["n"] += 1
        if calls["n"] == 1:
            await asyncio.sleep(5)  # the straggler
        return REPLY

    sw = _switcher(monkeypatch, generate, hedge_after=0.05)
    result = await asyncio.wait_for(_decide(sw), timeout=2.0)
    assert result["target_language"] == "mr"
    assert calls["n"] == 2
    assert sw.hedge_won is True
    # Reported latency is the caller's wait, not the straggler's.
    assert sw.latency_ms < 2000


async def test_hedge_disabled_by_zero(monkeypatch):
    gen = AsyncMock(return_value=REPLY)
    sw = _switcher(monkeypatch, gen, hedge_after=0)
    assert (await _decide(sw))["target_language"] == "mr"
    assert gen.await_count == 1


async def test_failing_straggler_does_not_lose_the_hedged_answer(monkeypatch):
    calls = {"n": 0}

    async def generate(_messages):
        calls["n"] += 1
        if calls["n"] == 1:
            await asyncio.sleep(0.1)
            raise RuntimeError("provider 500")
        await asyncio.sleep(0.3)
        return REPLY

    sw = _switcher(monkeypatch, generate, hedge_after=0.05)
    # First attempt raises before the second answers — the reply must still be returned.
    assert (await asyncio.wait_for(_decide(sw), timeout=2.0))["target_language"] == "mr"


async def test_both_failing_returns_none(monkeypatch):
    async def generate(_messages):
        await asyncio.sleep(0.05)
        raise RuntimeError("provider down")

    assert await _decide(_switcher(monkeypatch, generate, hedge_after=0.02)) is None


def test_hedge_default_sits_between_p50_and_the_tail():
    # Above the ~1.4s p50 (so common turns fire once) and below the 5.9s observed tail.
    assert 1.4 < DEFAULT_HEDGE_AFTER_S < 5.8


async def test_hedge_won_resets_between_decides(monkeypatch):
    # Without a per-decide reset the flag stays True for the rest of the call, so every later
    # fast decide would be misreported as hedged in the logs.
    calls = {"n": 0}

    async def generate(_messages):
        calls["n"] += 1
        if calls["n"] == 1:
            await asyncio.sleep(5)  # force a hedge on the first decide
        return REPLY

    sw = _switcher(monkeypatch, generate, hedge_after=0.05)
    await asyncio.wait_for(_decide(sw), timeout=2.0)
    assert sw.hedge_won is True

    sw._llm.generate = AsyncMock(return_value=REPLY)  # second decide answers immediately
    await _decide(sw)
    assert sw.hedge_won is False
