"""The LID trace rows are stamped when each event happened, not both at reply time.

_log_decision() used to write the request and response rows together after the model replied, so
both carried the completion timestamp: in the trace the pair landed after the main LLM's response
and showed ~0ms apart (run 7a07ae20: request 26.609920, response 26.609957 — 37 microseconds).
"""

import asyncio
import json
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
    assert rows["response"]["message"] is None  # declined, not failed
