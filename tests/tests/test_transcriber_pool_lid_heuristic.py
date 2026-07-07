"""Tests for the LEGACY per-segment LID heuristic (feature-flag fallback path).

Restored from master: when tools_config["llm_language_switch"] is off, the pool
wires the detector's per-segment on_language signal into a debounce/cooldown
heuristic that (in LID_MODE=active) delegates the switch to on_lid_switch.
"""

import asyncio
from unittest.mock import AsyncMock

import pytest

from bolna.transcriber.transcriber_pool import TranscriberPool


class FakeTranscriber:
    def __init__(self):
        self.run = AsyncMock()
        self.input_queue = asyncio.Queue()
        self.transcription_task = None


def _pool(on_lid_switch=None, mode="shadow"):
    pool = TranscriberPool(
        transcribers={"hi": FakeTranscriber(), "en": FakeTranscriber()},
        shared_input_queue=asyncio.Queue(),
        output_queue=asyncio.Queue(),
        active_label="hi",
        multilingual_config={"hi": {"language": "hi"}, "en": {"language": "en"}},
        on_lid_switch=on_lid_switch,
    )
    pool._lid_mode = mode
    return pool


def test_lang_to_label_map_built_from_multilingual_config():
    pool = _pool()
    assert pool._lang_to_label == {"hi": "hi", "en": "en"}


@pytest.mark.asyncio
async def test_shadow_mode_records_event_but_never_switches():
    cb = AsyncMock()
    pool = _pool(on_lid_switch=cb, mode="shadow")
    await pool._handle_lid_signal("en", 0.95)
    cb.assert_not_awaited()
    assert len(pool.lid_detection_events) == 1
    event = pool.lid_detection_events[0]
    assert event["suppressed_reason"] == "shadow_mode"
    assert event["would_switch"] is False
    assert event["target_label"] == "en"


@pytest.mark.asyncio
async def test_active_mode_delegates_switch_with_lid_trigger():
    cb = AsyncMock()
    pool = _pool(on_lid_switch=cb, mode="active")
    await pool._handle_lid_signal("en", 0.95)
    cb.assert_awaited_once_with("en", triggered_by="lid")
    assert pool.lid_detection_events[0]["would_switch"] is True


@pytest.mark.asyncio
async def test_low_confidence_suppressed():
    cb = AsyncMock()
    pool = _pool(on_lid_switch=cb, mode="active")
    await pool._handle_lid_signal("en", 0.40)
    cb.assert_not_awaited()
    assert pool.lid_detection_events[0]["suppressed_reason"] == "low_confidence"


@pytest.mark.asyncio
async def test_active_language_detection_is_noop():
    cb = AsyncMock()
    pool = _pool(on_lid_switch=cb, mode="active")
    await pool._handle_lid_signal("hi", 0.95)  # already the active language
    cb.assert_not_awaited()
    assert pool.lid_detection_events[0]["suppressed_reason"] == "already_active"


@pytest.mark.asyncio
async def test_cooldown_suppresses_rapid_resignal():
    import time

    cb = AsyncMock()
    pool = _pool(on_lid_switch=cb, mode="active")
    pool._lid_last_switch_time = time.monotonic()  # just switched
    await pool._handle_lid_signal("en", 0.95)
    cb.assert_not_awaited()
    assert pool.lid_detection_events[0]["suppressed_reason"] == "cooldown"
