"""Unit tests for SarvamLID's per-turn transcript buffer and socket liveness.

saaras emits one "data" message per VAD segment (several per spoken turn); the
detector accumulates them and the caller drains once per conversational turn.
"""

import asyncio
from unittest.mock import AsyncMock

import pytest

from bolna.lid.sarvam import SarvamLID


def _detector():
    # on_language=None, empty config → no network; we only exercise the buffer.
    return SarvamLID(on_language=None, config={})


def test_take_turn_transcript_empty_by_default():
    assert _detector().take_turn_transcript() == ("", None)


def test_accumulates_segments_into_one_turn():
    d = _detector()
    d._accumulate("अच्छा मुझे ये बताओ।", "hi")
    d._accumulate("ओके की क्या", "hi")
    d._accumulate("क्या पे रोड?", "hi")
    text, lang = d.take_turn_transcript()
    assert text == "अच्छा मुझे ये बताओ। ओके की क्या क्या पे रोड?"
    assert lang == "hi"


def test_take_clears_buffer():
    d = _detector()
    d._accumulate("I want to", "en")
    d.take_turn_transcript()
    assert d.take_turn_transcript() == ("", None)


def test_latest_language_wins_on_mixed_segments():
    d = _detector()
    d._accumulate("hello", "en")
    d._accumulate("नमस्ते", "hi")
    _, lang = d.take_turn_transcript()
    assert lang == "hi"


def test_blank_segments_ignored():
    d = _detector()
    d._accumulate("", "en")
    d._accumulate("   ", None)
    assert d.take_turn_transcript() == ("", None)


def test_buffer_age_none_when_empty():
    assert _detector().buffer_age_seconds() is None


def test_buffer_age_tracks_last_segment_and_clears_on_drain():
    d = _detector()
    d._accumulate("Can you talk in English?", "en")
    age = d.buffer_age_seconds()
    assert age is not None and 0 <= age < 1.0
    d.take_turn_transcript()
    # Drained → age resets so the idle-flush watcher won't refire on consumed speech.
    assert d.buffer_age_seconds() is None


def test_buffer_age_none_for_blank_only_segments():
    d = _detector()
    d._accumulate("", "en")
    assert d.buffer_age_seconds() is None


def test_buffer_language_peeks_without_drain():
    d = _detector()
    assert d.buffer_language() is None
    d._accumulate("Hello there", "en")
    assert d.buffer_language() == "en"
    # Peek must not drain — the transcript is still claimable afterwards.
    assert d.take_turn_transcript() == ("Hello there", "en")
    assert d.buffer_language() is None


def test_language_streak_counts_consecutive_and_resets():
    d = _detector()
    assert d.buffer_language_streak() == 0
    d._accumulate("hello", "en")
    assert d.buffer_language_streak() == 1
    d._accumulate("are you there", "en")
    assert d.buffer_language_streak() == 2
    # A different language resets the streak to 1 (not 0 — it's one segment of the new lang).
    d._accumulate("नमस्ते", "hi")
    assert d.buffer_language_streak() == 1
    d.take_turn_transcript()
    assert d.buffer_language_streak() == 0


def test_max_segment_tracks_longest_and_resets():
    d = _detector()
    assert d.buffer_max_segment_seconds() == 0.0
    d._accumulate("हाँ", "hi", 0.544)
    d._accumulate("हां, बैठ ला मैं", "hi", 0.960)
    # Two short fragments: longest is still below the 1.0s substance gate.
    assert d.buffer_max_segment_seconds() == 0.960
    d._accumulate("మీరు ఏం మాట్లాడుతున్నారో", "te", 1.92)
    assert d.buffer_max_segment_seconds() == 1.92
    d.take_turn_transcript()
    assert d.buffer_max_segment_seconds() == 0.0


def test_accumulate_backward_compatible_without_duration():
    d = _detector()
    d._accumulate("hello", "en")  # older call shape — duration defaults to 0.0
    assert d.buffer_max_segment_seconds() == 0.0
    assert d.take_turn_transcript() == ("hello", "en")


def test_buffer_event_set_on_segment_cleared_on_drain():
    d = _detector()
    assert not d.buffer_event().is_set()
    d._accumulate("Hello", "en")
    assert d.buffer_event().is_set()
    d.take_turn_transcript()
    # Cleared on drain so the watcher sleeps until NEW speech arrives.
    assert not d.buffer_event().is_set()
    # Blank segments must not wake the watcher.
    d._accumulate("", "en")
    assert not d.buffer_event().is_set()


# ── Socket liveness: graceful server-side close must not leave a mute detector ──


class _GracefullyClosedWS:
    """A websocket whose message iterator ends immediately (server closed 1000)."""

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration


@pytest.mark.asyncio
async def test_graceful_close_marks_dead_and_schedules_reconnect():
    d = _detector()
    d._ws = _GracefullyClosedWS()
    d._reconnect = AsyncMock()
    await d._receiver_loop()
    await asyncio.sleep(0)  # let the scheduled reconnect task run
    # Previously the async-for just ended: no _dead flag, no log, silent mute.
    assert d._dead is True
    d._reconnect.assert_awaited_once()


@pytest.mark.asyncio
async def test_graceful_close_during_stop_does_not_reconnect():
    d = _detector()
    d._ws = _GracefullyClosedWS()
    d._reconnect = AsyncMock()
    d._stopping = True
    await d._receiver_loop()
    await asyncio.sleep(0)
    assert d._dead is False
    d._reconnect.assert_not_awaited()


@pytest.mark.asyncio
async def test_schedule_reconnect_is_idempotent_while_in_flight():
    d = _detector()
    d._reconnect = AsyncMock()
    d._schedule_reconnect("receiver closed")
    d._schedule_reconnect("sender error")  # second trigger while first in flight
    await asyncio.sleep(0)
    d._reconnect.assert_awaited_once()


@pytest.mark.asyncio
async def test_reconnect_restores_liveness_and_resets_state(monkeypatch):
    monkeypatch.setattr(SarvamLID, "_RECONNECT_DELAY_S", 0)
    d = _detector()
    d._dead = True
    d._reconnecting = True
    d._dead_drop_logged = True
    d.start = AsyncMock()
    await d._reconnect()
    d.start.assert_awaited_once()
    assert d._dead is False
    assert d._dead_drop_logged is False
    assert d._reconnecting is False
    assert d._reconnect_attempts == 1


@pytest.mark.asyncio
async def test_reconnect_cap_stops_retrying(monkeypatch):
    monkeypatch.setattr(SarvamLID, "_RECONNECT_DELAY_S", 0)
    d = _detector()
    d._dead = True
    d._reconnect_attempts = SarvamLID._MAX_RECONNECTS
    d.start = AsyncMock()
    await d._reconnect()
    d.start.assert_not_awaited()
    assert d._dead is True


@pytest.mark.asyncio
async def test_reconnect_failure_keeps_detector_dead(monkeypatch):
    monkeypatch.setattr(SarvamLID, "_RECONNECT_DELAY_S", 0)
    d = _detector()
    d._dead = True
    d._reconnecting = True
    d.start = AsyncMock(side_effect=RuntimeError("403"))
    await d._reconnect()
    assert d._dead is True
    assert d._reconnecting is False  # a later trigger may try the remaining budget
