"""The LID trace rows are stamped when each event happened, not both at reply time.

_log_decision() used to write the request and response rows together after the model replied, so
both carried the completion timestamp: in the trace the pair landed after the main LLM's response
and showed ~0ms apart: the two rows landed 37 microseconds apart, which no real model call takes.
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from bolna.helpers.language_switcher import LanguageSwitcher

REPLY = json.dumps({"target_language": "mr", "target_confidence": 0.9, "reasoning": "Marathi"})
MOD = "bolna.helpers.language_switcher.convert_to_request_log"


def _switcher(monkeypatch, generate):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    sw = LanguageSwitcher(available_labels=["hi", "mr"], run_id="r1")
    sw._llm = MagicMock()
    sw._llm.generate = generate
    return sw


def _rows(calls):
    return {c.kwargs["direction"].value: c.kwargs for c in calls}


async def test_the_two_rows_are_stamped_around_the_call_not_together(monkeypatch):
    async def slow_generate(*a, **k):
        await asyncio.sleep(0.2)
        return REPLY, {}

    sw = _switcher(monkeypatch, slow_generate)
    with patch(MOD) as log:
        await sw.decide("mala samajla nahi", "garbled", "hi")

    rows = _rows(log.call_args_list)
    request_ts, response_ts = rows["request"]["ts"], rows["response"]["ts"]
    # The gap must reflect the real round trip, not two log calls running back to back.
    assert response_ts - request_ts >= 0.2
    assert rows["response"]["latency"] >= 0.2
    # Both rows still belong to one leg.
    assert rows["request"]["meta_info"]["request_id"] == rows["response"]["meta_info"]["request_id"]


async def test_a_failed_judge_is_distinguishable_from_a_decline(monkeypatch):
    # _hedged_generate swallows the error and returns None, so without the flag a throttled
    # judge and a model that validly declined to switch look identical in the trace.
    sw = _switcher(monkeypatch, AsyncMock(side_effect=RuntimeError("bedrock throttled")))
    with patch(MOD) as log:
        assert await sw.decide("mala samajla nahi", "garbled", "hi") is None

    rows = _rows(log.call_args_list)
    assert set(rows) == {"request", "response"}
    assert rows["response"]["message"] == {"error": "generate_failed"}


async def test_a_declined_decision_still_leaves_both_rows(monkeypatch):
    sw = _switcher(monkeypatch, AsyncMock(return_value=("null", {})))
    with patch(MOD) as log:
        assert await sw.decide("mala samajla nahi", "garbled", "hi") is None

    rows = _rows(log.call_args_list)
    assert set(rows) == {"request", "response"}
    assert rows["response"]["message"] == {"target_language": None}  # declined, not failed


async def test_a_cancelled_decide_still_leaves_a_reply_row(monkeypatch):
    # task_manager caps decide() with wait_for(6.0s) and the observed tail is 5.9s, so this fires.
    # CancelledError is a BaseException, so `except Exception` would have missed exactly the
    # slowest decides — the ones the request row is most likely to be left orphaned on.
    async def never_answers(*a, **k):
        await asyncio.sleep(10)

    sw = _switcher(monkeypatch, never_answers)
    with patch(MOD) as log:
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(sw.decide("mala samajla nahi", "garbled", "hi"), timeout=0.05)

    rows = _rows(log.call_args_list)
    assert set(rows) == {"request", "response"}
    assert rows["response"]["message"] == {"error": "cancelled"}


async def test_a_throwing_response_log_neither_duplicates_nor_discards(monkeypatch):
    # _log_response used to sit inside the try, so a throw from it was caught by `except Exception`,
    # which logged a SECOND response row and returned None — discarding a decision the model gave.
    sw = _switcher(monkeypatch, AsyncMock(return_value=(REPLY, {})))
    sw.last_usage = "not-a-dict"  # usage.get(...) would have raised
    with patch(MOD) as log:
        result = await sw.decide("mala samajla nahi", "garbled", "hi")

    assert result["target_language"] == "mr"  # the decision survives
    directions = [c.kwargs["direction"].value for c in log.call_args_list]
    assert directions.count("response") == 1  # exactly one reply row


async def test_a_decline_is_self_describing_not_a_blank_cell(monkeypatch):
    # write_request_logs normalizes None to "", so message=None reached the trace as an empty
    # Data cell and read like a truncated row rather than "the model declined to switch".
    sw = _switcher(monkeypatch, AsyncMock(return_value=("null", {})))
    with patch(MOD) as log:
        assert await sw.decide("mala samajla nahi", "garbled", "hi") is None

    assert _rows(log.call_args_list)["response"]["message"] == {"target_language": None}
