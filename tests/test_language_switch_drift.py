"""Cross-turn drift evidence and the same-language idle-flush skip.

A judge seeing only the current turn rejects a lone "no" every time, so a caller who has changed
language can never accumulate evidence and the agent cannot self-correct unless they name a
language out loud. RECENT TURNS supplies that state, carrying each turn's duration so
acknowledgment mis-tags stay non-evidence.
"""

from unittest.mock import MagicMock


from bolna.agent_manager.task_manager import TaskManager
from bolna.helpers.language_switcher import LanguageSwitcher
from bolna.prompts import LANGUAGE_SWITCH_SYSTEM_PROMPT, LANGUAGE_SWITCH_TURN_PROMPT
from bolna.transcriber.transcriber_pool import TranscriberPool


def _pool(events=None, segments=None):
    pool = MagicMock(spec=TranscriberPool)
    pool.lid_detection_events = events if events is not None else []
    pool.lid_buffer_segments.return_value = segments or []
    return pool


def _recent(pool, limit=4):
    return TaskManager._TaskManager__recent_detected_turns(pool, limit)


def _evidence(pool, active):
    return TaskManager._TaskManager__buffered_language_evidence(pool, active)


def _foreign(pool, active):
    return _evidence(pool, active)[1]


def _event(lang, seconds, flow="llm_switch"):
    return {"flow": flow, "detected_language": lang, "detector_segments": [{"lang": lang, "audio_s": seconds}]}


# ---- cross-turn evidence (issue 2) ----


def test_recent_turns_read_from_existing_telemetry():
    pool = _pool([_event("hi", 2.1), _event("en", 1.8), _event("en", 1.6)])
    assert _recent(pool) == [("hi", 2.1, None), ("en", 1.8, None), ("en", 1.6, None)]


def test_recent_turns_ignores_legacy_heuristic_records():
    # Legacy records have a different shape; mixing them would feed the judge nonsense.
    pool = _pool([{"lang": "te", "suppressed_reason": "cooldown"}, _event("en", 1.5)])
    assert _recent(pool) == [("en", 1.5, None)]


def test_recent_turns_is_bounded_and_oldest_first():
    pool = _pool([_event(lang, 1.0) for lang in ("hi", "hi", "en", "en", "en")])
    assert [entry[0] for entry in _recent(pool, limit=3)] == ["en", "en", "en"]


def test_recent_turns_empty_when_no_history():
    assert _recent(_pool()) == []


def test_recent_turns_duration_comes_from_the_detected_languages_own_segments():
    # A one-borrowed-word turn inside a 2s active utterance must not read as 2s of that
    # language.
    rec = {
        "flow": "llm_switch",
        "detected_language": "en",
        "buffered_max_segment_s": 2.0,
        "detector_segments": [
            {"lang": "hi", "audio_s": 2.0},
            {"lang": "en", "audio_s": 0.3},
        ],
    }
    assert _recent(_pool([rec])) == [("en", 0.3, None)]


def test_recent_turns_duration_is_zero_without_matching_segments():
    # No fallback to the buffer max: with no en-tagged segment the detector never heard
    # English — borrowing another language's duration fed rule 8 fake "real speech".
    rec = {"flow": "llm_switch", "detected_language": "en", "buffered_max_segment_s": 1.5, "detector_segments": []}
    assert _recent(_pool([rec])) == [("en", 0.0, None)]


def test_formatting_exposes_duration_so_short_mistags_stay_non_evidence():
    fmt = LanguageSwitcher._format_recent_turns([("hi", 2.1), ("en", 0.4)])
    assert fmt == "hi(2.1), en(0.4)"
    assert LanguageSwitcher._format_recent_turns([]) == "(none)"
    assert LanguageSwitcher._format_recent_turns(None) == "(none)"


def test_formatting_survives_missing_duration():
    assert LanguageSwitcher._format_recent_turns([("en", None)]) == "en(0.0)"


def test_turn_prompt_carries_recent_turns():
    rendered = LANGUAGE_SWITCH_TURN_PROMPT.format(
        active_language="hi",
        available_languages="en, hi",
        recent_turns="en(1.8), en(1.6)",
        detector_transcript="not sure sir",
        active_transcript="",
    )
    assert "RECENT TURNS" in rendered
    assert "en(1.8), en(1.6)" in rendered


def test_drift_rule_is_in_the_cacheable_system_prompt():
    # The rule belongs in the static block (cached); only the evidence varies per turn.
    assert "SUSTAINED DRIFT ACROSS TURNS IS A SWITCH" in LANGUAGE_SWITCH_SYSTEM_PROMPT
    # Rule 8 must still carry the DEFINITION of the RECENT TURNS line format (other rules —
    # 4a's flap check — may reference RECENT TURNS earlier, but only 8 explains its shape).
    assert "RECENT TURNS lists earlier turns" in LANGUAGE_SWITCH_SYSTEM_PROMPT.split("SUSTAINED DRIFT")[1]


async def test_decide_passes_recent_turns_into_the_prompt(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("LANGUAGE_SWITCH_HEDGE_AFTER_S", "0")
    sw = LanguageSwitcher(available_labels=["en", "hi"], run_id="r")
    sent = {}

    async def generate(messages, **kwargs):
        sent["user"] = messages[-1]["content"]
        return '{"target_language": null, "target_confidence": 0.0}', {}

    sw._llm = MagicMock()
    sw._llm.generate = generate
    sw._log_decision = MagicMock()
    await sw.decide("not sure sir", "", "hi", recent_turns=[("en", 1.8), ("en", 1.6)])
    assert "en(1.8), en(1.6)" in sent["user"]


# ---- same-language idle-flush skip (issue 3) ----


def test_foreign_tag_found_anywhere_in_the_buffer_not_just_latest():
    # A turn that opened in English and ended on a Hindi word is still evidence; reading only
    # the newest tag made it wait out the slower same-language window.
    pool = _pool(segments=[{"lang": "en"}, {"lang": "hi"}])
    assert _foreign(pool, "hi") == ["en"]


def test_all_active_language_buffer_has_no_foreign_evidence():
    pool = _pool(segments=[{"lang": "hi"}, {"lang": "hi"}])
    assert _foreign(pool, "hi") == []


def test_unsupported_tags_count_as_foreign_evidence():
    # Rule 4 lets the judge remap confusable clusters (kn↔te), so an unsupported tag must still
    # reach it rather than being skipped as "no evidence".
    pool = _pool(segments=[{"lang": "kn"}])
    assert _foreign(pool, "en") == ["kn"]


def test_region_tags_normalized_and_deduped():
    pool = _pool(segments=[{"lang": "en-US"}, {"lang": "en"}, {"lang": "mr-IN"}])
    assert _foreign(pool, "hi") == ["en", "mr"]


def test_untagged_segments_are_not_evidence():
    pool = _pool(segments=[{"lang": None}, {"lang": ""}])
    assert _evidence(pool, "hi") == (False, [], 0.0)


def test_saw_tags_distinguishes_all_active_from_no_information():
    # Only "read the buffer, all active language" may skip a decide.
    assert _evidence(_pool(segments=[{"lang": "hi"}]), "hi") == (True, [], 0.0)
    # A backend without buffer_segments returns [] — that is no information, so never skip:
    # skipping would make switching permanently inert on that backend.
    assert _evidence(_pool(segments=[]), "hi") == (False, [], 0.0)


def test_unreadable_segments_api_never_skips_and_never_raises():
    # The idle watcher's outer handler exits its loop on any exception, which would kill
    # stuck-language recovery for the whole call.
    pool = MagicMock(spec=TranscriberPool)
    pool.lid_buffer_segments.return_value = object()  # not iterable
    assert _evidence(pool, "hi") == (False, [], 0.0)
    pool.lid_buffer_segments.side_effect = AttributeError("no such method")
    assert _evidence(pool, "hi") == (False, [], 0.0)


async def test_watcher_skips_the_decide_when_buffer_is_all_active_language(monkeypatch):
    # When buffered_lang already equals the active language the judge can only answer "stay",
    # so the decide latency and lock hold buy nothing.
    import asyncio

    from unittest.mock import AsyncMock

    tm = MagicMock()
    tm.conversation_ended = False
    tm.hangup_triggered = False
    tm._end_call_in_progress = False
    tm.has_transfer = False
    tm.language = "en"
    tm.handle_language_switch = AsyncMock()
    pool = MagicMock(spec=TranscriberPool)
    pool.lid_buffer_age.return_value = 5.0  # well past both thresholds
    pool.lid_buffer_language.return_value = "en"
    pool.lid_buffer_segments.return_value = [{"lang": "en"}, {"lang": "en"}]
    pool.lid_buffer_event.return_value = None
    tm.tools = {"transcriber": pool}
    tm._should_ignore_transcriber_input = TaskManager._should_ignore_transcriber_input.__get__(tm, TaskManager)
    tm._TaskManager__buffered_language_evidence = TaskManager._TaskManager__buffered_language_evidence
    watcher = TaskManager._TaskManager__lid_idle_watcher.__get__(tm, TaskManager)

    task = asyncio.create_task(watcher())
    await asyncio.sleep(0.25)
    task.cancel()
    tm.handle_language_switch.assert_not_awaited()  # no judge call for a foregone "stay"


async def test_watcher_still_fires_when_a_foreign_tag_is_present(monkeypatch):
    import asyncio

    from unittest.mock import AsyncMock

    tm = MagicMock()
    tm.conversation_ended = False
    tm.hangup_triggered = False
    tm._end_call_in_progress = False
    tm.has_transfer = False
    tm.language = "hi"
    tm.handle_language_switch = AsyncMock()
    pool = MagicMock(spec=TranscriberPool)
    pool.lid_buffer_age.return_value = 5.0
    pool.lid_buffer_language.return_value = "hi"  # latest is active…
    pool.lid_buffer_segments.return_value = [{"lang": "en"}, {"lang": "hi"}]  # …but 'en' earlier
    pool.lid_buffer_event.return_value = None
    tm.tools = {"transcriber": pool}
    tm._should_ignore_transcriber_input = TaskManager._should_ignore_transcriber_input.__get__(tm, TaskManager)
    tm._TaskManager__buffered_language_evidence = TaskManager._TaskManager__buffered_language_evidence
    watcher = TaskManager._TaskManager__lid_idle_watcher.__get__(tm, TaskManager)

    task = asyncio.create_task(watcher())
    await asyncio.sleep(0.25)
    task.cancel()
    tm.handle_language_switch.assert_awaited()  # foreign evidence anywhere must reach the judge
