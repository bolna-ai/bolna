"""Per-call LID spend: judge tokens (Haiku) + detector audio seconds (Sarvam/Soniox),
persisted as one `lid_usage` record inside lid_detection_events (JSONB — no new columns)."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

from bolna.agent_manager.task_manager import TaskManager
from bolna.transcriber.transcriber_pool import TranscriberPool

MOD = "bolna.helpers.language_switcher"


def make_switcher(payload, usage=None):
    from bolna.helpers.language_switcher import LanguageSwitcher

    fake_llm = MagicMock()
    fake_llm.generate = AsyncMock(return_value=(payload, usage or {}))
    with patch(f"{MOD}.LiteLLM", return_value=fake_llm):
        switcher = LanguageSwitcher(available_labels=["en", "hi"])
    return switcher, fake_llm


# ── judge token capture ───────────────────────────────────────────────────────────


async def test_decide_tallies_judge_usage():
    payload = json.dumps({"target_language": None, "reasoning": "stay"})
    usage = {"input_tokens": 5500, "output_tokens": 60, "cached_tokens": 5000}
    switcher, _ = make_switcher(payload, usage)
    await switcher.decide("hello", "hello", "en")
    await switcher.decide("namaste", "namaste", "en")
    assert switcher.usage_totals == {
        "input_tokens": 11000,
        "output_tokens": 120,
        "cached_tokens": 10000,
        "requests": 2,
    }


async def test_csv_response_row_carries_tokens():
    payload = json.dumps({"target_language": None, "reasoning": "stay"})
    usage = {"input_tokens": 5500, "output_tokens": 60, "cached_tokens": 5000}
    switcher, _ = make_switcher(payload, usage)
    with patch(f"{MOD}.convert_to_request_log") as log:
        await switcher.decide("hello", "hello", "en")
    response_kwargs = log.call_args_list[-1].kwargs
    assert response_kwargs["input_tokens"] == 5500
    assert response_kwargs["output_tokens"] == 60
    assert response_kwargs["cached_tokens"] == 5000


async def test_missing_usage_logs_none_not_crash():
    payload = json.dumps({"target_language": None})
    switcher, _ = make_switcher(payload, usage={})
    with patch(f"{MOD}.convert_to_request_log") as log:
        result = await switcher.decide("hello", "hello", "en")
    assert result is not None
    assert log.call_args_list[-1].kwargs["input_tokens"] is None
    # requests counts completed responses, even when the provider omitted usage.
    assert switcher.usage_totals["requests"] == 1
    assert switcher.usage_totals["input_tokens"] == 0


async def test_models_used_tracks_the_answering_model():
    payload = json.dumps({"target_language": None})
    usage = {"input_tokens": 10, "output_tokens": 1}
    switcher, _ = make_switcher(payload, usage)
    await switcher.decide("hello", "hello", "en")
    assert switcher.models_used == [switcher.model]
    # a runtime fallback swap mid-call must show BOTH models
    switcher.model = "anthropic/other-judge"
    await switcher.decide("namaste", "namaste", "en")
    assert len(switcher.models_used) == 2


# ── detector audio seconds ────────────────────────────────────────────────────────


def make_lid(bytes_per_second):
    from bolna.lid.base import LIDBackend

    lid = object.__new__(LIDBackend)
    lid.bytes_fed = 0
    lid.input_bytes_per_second = bytes_per_second
    return lid


def test_audio_seconds_mulaw_8k():
    lid = make_lid(8000)
    lid.bytes_fed = 8000 * 63  # 63s of mulaw@8k
    assert lid.audio_seconds_fed() == 63.0


def test_audio_seconds_linear16_16k():
    lid = make_lid(32000)
    lid.bytes_fed = 32000 * 10
    assert lid.audio_seconds_fed() == 10.0


def test_audio_seconds_zero_rate_is_safe():
    assert make_lid(0).audio_seconds_fed() == 0.0


# ── the lid_usage record ──────────────────────────────────────────────────────────


def make_tm(events=None, seconds=63.0, switcher=True):
    tm = MagicMock()
    pool = MagicMock(spec=TranscriberPool)
    pool.lid_detection_events = events if events is not None else []
    pool.lid_audio_seconds.return_value = seconds
    pool._record_detector_health = MagicMock()
    tm.tools = {"transcriber": pool}
    if switcher:
        tm.language_switcher = MagicMock()
        tm.language_switcher.model = "bedrock/haiku"
        tm.language_switcher.models_used = ["bedrock/haiku"]
        tm.language_switcher.usage_totals = {
            "input_tokens": 11000,
            "output_tokens": 120,
            "cached_tokens": 10000,
            "requests": 2,
        }
    else:
        tm.language_switcher = None
    tm._TaskManager__record_lid_usage = TaskManager._TaskManager__record_lid_usage.__get__(tm, TaskManager)
    tm._TaskManager__snapshot_lid_events = TaskManager._TaskManager__snapshot_lid_events.__get__(tm, TaskManager)
    return tm, pool


def test_snapshot_appends_one_usage_record():
    tm, pool = make_tm()
    events = tm._TaskManager__snapshot_lid_events()
    usage = [e for e in events if e.get("type") == "lid_usage"]
    assert len(usage) == 1
    assert usage[0]["judge_input_tokens"] == 11000
    assert usage[0]["judge_requests"] == 2
    assert usage[0]["detector_audio_seconds"] == 63.0
    assert usage[0]["judge_model"] == "bedrock/haiku"
    assert usage[0]["judge_models"] == ["bedrock/haiku"]


def test_snapshot_is_idempotent():
    tm, pool = make_tm()
    tm._TaskManager__snapshot_lid_events()
    events = tm._TaskManager__snapshot_lid_events()
    assert len([e for e in events if e.get("type") == "lid_usage"]) == 1


def test_no_switcher_no_audio_writes_nothing():
    tm, pool = make_tm(seconds=None, switcher=False)
    events = tm._TaskManager__snapshot_lid_events()
    assert [e for e in events if e.get("type") == "lid_usage"] == []
