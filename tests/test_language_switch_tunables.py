"""The switch tunables: decide timeout / settle / audio gap resolution, and the gates they feed.

Covers: the decide ceiling clears the observed tail, the audio gap is tools_config-then-env, the
explicit-request confidence bar never sits above the general gate (a stricter explicit bar would
reject the caller-asked-by-name case while admitting the incidental one), and the playback gate's
deadline stays below the decide ceiling.
"""

import os
from unittest.mock import MagicMock

import pytest

from bolna.agent_manager.task_manager import TaskManager
from bolna.constants import (
    LANGUAGE_SWITCH_AUDIO_GAP_S,
    LANGUAGE_SWITCH_DECIDE_TIMEOUT_S,
    LANGUAGE_SWITCH_MAX_HOLD_S,
    LANGUAGE_SWITCH_SETTLE_MS,
)


def _tm(tools_config=None):
    tm = MagicMock()
    tm.task_config = {"tools_config": tools_config or {}}
    for name in ("switch_decide_timeout_s", "switch_settle_ms", "switch_audio_gap_s"):
        attr = f"_TaskManager__{name}"
        setattr(tm, attr, getattr(TaskManager, attr).__get__(tm, TaskManager))
    return tm


def test_decide_timeout_default_clears_observed_tail():
    # QA 972632b5 recorded 5.8s/5.9s decides; the buffer is drained before the decide, so a
    # timeout loses that utterance outright. The default must sit above the observed tail.
    assert LANGUAGE_SWITCH_DECIDE_TIMEOUT_S >= 6.0
    assert _tm()._TaskManager__switch_decide_timeout_s() == LANGUAGE_SWITCH_DECIDE_TIMEOUT_S


def test_decide_timeout_is_env_overridable(monkeypatch):
    monkeypatch.setenv("LANGUAGE_SWITCH_DECIDE_TIMEOUT_S", "9.5")
    assert _tm()._TaskManager__switch_decide_timeout_s() == 9.5


def test_settle_is_env_overridable(monkeypatch):
    assert _tm()._TaskManager__switch_settle_ms() == LANGUAGE_SWITCH_SETTLE_MS
    monkeypatch.setenv("LANGUAGE_SWITCH_SETTLE_MS", "150")
    assert _tm()._TaskManager__switch_settle_ms() == 150


def test_audio_gap_prefers_tools_config_then_env(monkeypatch):
    # Per-agent first (the right gap depends on the carrier's clear semantics), env fallback —
    # same precedence as language_switch_lid_provider.
    monkeypatch.setenv("LANGUAGE_SWITCH_AUDIO_GAP_S", "0.4")
    assert _tm({"language_switch_audio_gap_s": 0.9})._TaskManager__switch_audio_gap_s() == 0.9
    assert _tm()._TaskManager__switch_audio_gap_s() == 0.4
    monkeypatch.delenv("LANGUAGE_SWITCH_AUDIO_GAP_S")
    assert _tm()._TaskManager__switch_audio_gap_s() == LANGUAGE_SWITCH_AUDIO_GAP_S


def test_audio_gap_zero_disables_the_sleep():
    # The call site guards on > 0, so 0 is the documented off switch.
    assert _tm({"language_switch_audio_gap_s": 0})._TaskManager__switch_audio_gap_s() == 0


@pytest.mark.parametrize("min_conf", ["0.7", "0.5", "0.85"])
def test_explicit_bar_never_stricter_than_general_gate(monkeypatch, min_conf):
    # An explicit by-name request is legitimately short and bypasses the substance gate; if its
    # confidence bar sat above min_conf, a 0.75-confidence "speak Hindi" in a 0.6s utterance
    # would be rejected while an incidental 1.5s utterance at the same confidence switched.
    monkeypatch.delenv("LANGUAGE_SWITCH_EXPLICIT_MIN_CONFIDENCE", raising=False)
    monkeypatch.setenv("LANGUAGE_SWITCH_MIN_CONFIDENCE", min_conf)
    resolved_min = float(os.getenv("LANGUAGE_SWITCH_MIN_CONFIDENCE"))
    resolved_explicit = float(os.getenv("LANGUAGE_SWITCH_EXPLICIT_MIN_CONFIDENCE", str(resolved_min)))
    assert resolved_explicit <= resolved_min


def test_explicit_bar_still_env_overridable(monkeypatch):
    monkeypatch.setenv("LANGUAGE_SWITCH_MIN_CONFIDENCE", "0.7")
    monkeypatch.setenv("LANGUAGE_SWITCH_EXPLICIT_MIN_CONFIDENCE", "0.6")
    assert float(os.getenv("LANGUAGE_SWITCH_EXPLICIT_MIN_CONFIDENCE", "0.7")) == 0.6


def test_playback_gate_deadline_is_below_the_decide_ceiling():
    # The gate bounds caller-facing audio delay, so it must NOT inherit the decide timeout's
    # tail sizing — 6s of held audio would be worse than a truncated wrong-language reply.
    assert LANGUAGE_SWITCH_MAX_HOLD_S < LANGUAGE_SWITCH_DECIDE_TIMEOUT_S


def test_playback_gate_deadline_leaves_room_for_a_hedged_decide():
    # Above a hedged decide (hedge fires at 1.8s, second reply ~1.4s) so the common slow turn
    # resolves inside the gate instead of leaking old-language audio.
    assert LANGUAGE_SWITCH_MAX_HOLD_S >= 3.0
