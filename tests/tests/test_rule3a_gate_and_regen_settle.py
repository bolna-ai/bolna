"""Rule-3a alphanumeric veto + settle-window regeneration.

From prod b381ba0a (VBL barcode readout): (1) the judge switched hi→en at 0.92 on
"This B1" — an alphanumeric readout rule 3a forbids switching on; (2) each fragment
final ("ये B1।", "V1", "21, 65, 11, 69") cancelled the in-flight generation, shattering
one answer across many synth turns. The veto blocks (1) deterministically; the settle
window collapses (2) into a single regeneration of the merged utterance.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from bolna.agent_manager.task_manager import TaskManager, is_alphanumeric_readout
from bolna.constants import LLM_REGEN_SETTLE_S


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
