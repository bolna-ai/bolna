"""Rule-3a alphanumeric veto + settle-window regeneration."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from bolna.agent_manager.task_manager import TaskManager, is_alphanumeric_readout
from bolna.constants import LLM_REGEN_SETTLE_S
from bolna.helpers.utils import safe_log_text


# ── rule-3a helper ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text",
    [
        "This B1.",
        "ये B1।",
        "V1",
        "21, 65, 11, 69. Hello.",
        "B1 21 65",
        "9 8 7 6 5",
    ],
)
def test_readouts_are_vetoed(text):
    assert is_alphanumeric_readout(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "I want English",  # bare letters are not code tokens
        "can we talk in Tamil",
        "Is she. Yes.",
        "మీరు తెలుగులో మాట్లాడతారా?",
        "मेरा number 98765 hai na call karo",  # code embedded in a real frame -> judge's call
        "",
        "okay",
    ],
)
def test_speech_is_not_vetoed(text):
    assert is_alphanumeric_readout(text) is False


# ── settle-window regeneration ───────────────────────────────────────────────────


def make_tm(*, activity_values=None):
    tm = MagicMock()
    tm.regen_settle_task = None
    tm.regen_settle_payload = None
    tm.llm_task = None
    tm.tools = {"input": MagicMock()}
    tm.kickoff_calls = []

    def kickoff(text, meta):
        tm.kickoff_calls.append((text, dict(meta)))

    tm.kickoff_llm_generation = kickoff
    tm.regen_settle_armed = TaskManager.regen_settle_armed.__get__(tm, TaskManager)
    tm.arm_regen_settle = TaskManager.arm_regen_settle.__get__(tm, TaskManager)
    tm._TaskManager__regen_after_settle = TaskManager._TaskManager__regen_after_settle.__get__(tm, TaskManager)
    return tm


@pytest.mark.asyncio
async def test_rapid_fragments_regenerate_once_with_last_merge():
    tm = make_tm()
    # Three fragments inside the window — like a barcode readout.
    tm.arm_regen_settle("ये B1", {"sequence_id": 3, "turn_id": 3})
    await asyncio.sleep(LLM_REGEN_SETTLE_S / 3)
    tm.arm_regen_settle("ये B1 21, 65", {"sequence_id": 4, "turn_id": 4})
    await asyncio.sleep(LLM_REGEN_SETTLE_S / 3)
    tm.arm_regen_settle("ये B1 21, 65 11, 69", {"sequence_id": 5, "turn_id": 5})
    await asyncio.sleep(LLM_REGEN_SETTLE_S + 0.15)

    assert len(tm.kickoff_calls) == 1
    text, meta = tm.kickoff_calls[0]
    assert text == "ये B1 21, 65 11, 69"
    assert meta["sequence_id"] == 5
    assert tm.regen_settle_payload is None


@pytest.mark.asyncio
async def test_settle_fires_after_quiet_window():
    tm = make_tm()
    tm.arm_regen_settle("hello there", {"sequence_id": 2, "turn_id": 2})
    assert tm.kickoff_calls == []  # not yet — window open
    await asyncio.sleep(LLM_REGEN_SETTLE_S + 0.15)
    assert len(tm.kickoff_calls) == 1


@pytest.mark.asyncio
async def test_cancelled_settle_never_generates():
    tm = make_tm()
    tm.arm_regen_settle("stale", {"sequence_id": 2, "turn_id": 2})
    tm.regen_settle_task.cancel()  # what __cleanup_downstream_tasks does
    tm.regen_settle_payload = None
    await asyncio.sleep(LLM_REGEN_SETTLE_S + 0.15)
    assert tm.kickoff_calls == []


@pytest.mark.asyncio
async def test_fire_with_cleared_payload_is_noop():
    tm = make_tm()
    tm.arm_regen_settle("x", {"sequence_id": 2})
    tm.regen_settle_payload = None  # cleared by a race with cleanup
    await asyncio.sleep(LLM_REGEN_SETTLE_S + 0.15)
    assert tm.kickoff_calls == []


@pytest.mark.asyncio
async def test_regen_settle_armed_tracks_timer_lifecycle():
    tm = make_tm()
    assert tm.regen_settle_armed() is False  # nothing armed
    tm.arm_regen_settle("hello", {"sequence_id": 2, "turn_id": 2})
    assert tm.regen_settle_armed() is True  # window open: eager EOT and late duplicates must gate on this
    await asyncio.sleep(LLM_REGEN_SETTLE_S + 0.15)
    assert tm.regen_settle_armed() is False  # fired
    tm.arm_regen_settle("again", {"sequence_id": 3, "turn_id": 3})
    tm.regen_settle_task.cancel()
    await asyncio.sleep(0)
    assert tm.regen_settle_armed() is False  # cancelled


# ── log sanitization (CodeQL log-injection) ──────────────────────────────────────


def test_safe_log_text_blocks_forged_entries():
    evil = "hello\n2026-08-18 05:00:00 INFO {task_manager} FAKE granted\r\x1b[31m"
    out = safe_log_text(evil)
    assert "\n" not in out and "\r" not in out and "\x1b" not in out


def test_safe_log_text_preserves_speech_and_truncates():
    assert safe_log_text("ये B1। 21 65") == "ये B1। 21 65"
    assert safe_log_text(None) == ""
    assert len(safe_log_text("x" * 500)) == 120
    assert len(safe_log_text("x" * 500, 80)) == 80


# ── exclusion list: providers whose finals can't land inside the window ──────────


class DeepgramStub:
    pass


class SonioxStub:
    pass


class PoolStub:
    def __init__(self, active_label, transcribers):
        self.active_label = active_label
        self.transcribers = transcribers


def make_can_fire_tm(transcriber):
    tm = MagicMock()
    tm.tools = {"transcriber": transcriber}
    tm.regen_settle_can_fire = TaskManager.regen_settle_can_fire.__get__(tm, TaskManager)
    return tm


def test_excluded_transcriber_skips_the_window():
    assert make_can_fire_tm(DeepgramStub()).regen_settle_can_fire() is False


def test_non_excluded_transcriber_arms():
    assert make_can_fire_tm(SonioxStub()).regen_settle_can_fire() is True


def test_can_fire_follows_the_active_pool_member():
    pool = PoolStub("hi", {"hi": DeepgramStub(), "en": SonioxStub()})
    assert make_can_fire_tm(pool).regen_settle_can_fire() is False
    pool.active_label = "en"
    assert make_can_fire_tm(pool).regen_settle_can_fire() is True
