"""QA-driven fixes: the idle-flush LIVE marker (QA 7c7d4b00) and the always-on language pin
(QA 2fb6315f / a39f691c / 7c7d4b00 drift).

Marker: on idle-flush firings there is no main-ASR turn, so LIVE is empty because nobody
produced one — left as "" the judge reads it through the empty-LIVE-is-mismatch-evidence rule
and a detector transliteration gets "confirmed" by an absence we manufactured.

Pin: the language directive used to be installed only when a per-language prompt variant
existed AND a context_note was passed — tool-driven switches stripped it and agents without
multilingual variants never had one, so the main LLM drifted (a Hindi line mid-Telugu call,
an English closing line) and QA attributed the drift to LID.
"""

from unittest.mock import MagicMock

import pytest

from bolna.agent_manager.task_manager import TaskManager
from bolna.helpers.language_switcher import LIVE_UNAVAILABLE_MARKER, LanguageSwitcher
from bolna.prompts import LANGUAGE_SWITCH_SYSTEM_PROMPT


# ---- idle-flush LIVE marker in decide() ----


def _switcher_capturing_prompt(monkeypatch, sent):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("LANGUAGE_SWITCH_HEDGE_AFTER_S", "0")
    sw = LanguageSwitcher(available_labels=["en", "te", "hi"], run_id="r")

    async def generate(messages):
        sent["user"] = messages[-1]["content"]
        return '{"target_language": null, "target_confidence": 0.0}'

    sw._llm = MagicMock()
    sw._llm.generate = generate
    sw._log_decision = MagicMock()
    return sw


@pytest.mark.asyncio
async def test_empty_live_becomes_the_unavailable_marker(monkeypatch):
    sent = {}
    sw = _switcher_capturing_prompt(monkeypatch, sent)
    await sw.decide("హాయ్ హాయ్, వాట్సాప్.", "", "en")
    assert LIVE_UNAVAILABLE_MARKER in sent["user"]
    assert 'LIVE transcript: "()"' not in sent["user"]


@pytest.mark.asyncio
async def test_real_live_transcript_is_passed_through_unchanged(monkeypatch):
    sent = {}
    sw = _switcher_capturing_prompt(monkeypatch, sent)
    await sw.decide("అవును, చెప్పండి", "avunu cheppandi", "en")
    assert "avunu cheppandi" in sent["user"]
    assert LIVE_UNAVAILABLE_MARKER not in sent["user"]


def test_system_prompt_names_the_exact_marker_string():
    # The prompt voids the empty-LIVE inference by quoting the marker verbatim — if the
    # constant and the prompt drift apart the rule silently stops matching.
    assert LIVE_UNAVAILABLE_MARKER in LANGUAGE_SWITCH_SYSTEM_PROMPT


def test_system_prompt_covers_reverse_transliteration():
    # Rule 5 must handle English rendered in Indic script (the 7c7d4b00 failure), not just
    # romanized Indic mis-tagged as English.
    assert "వాట్సాప్" in LANGUAGE_SWITCH_SYSTEM_PROMPT


# ---- always-on language pin ----

APPLY = TaskManager._TaskManager__apply_language_directive
DIRECTIVE = TaskManager._TaskManager__language_directive


def _tm(base="## Agent Prompt:\n\nBe helpful.\n\n## Transcript:\n", multilingual=None):
    tm = MagicMock()
    tm.multilingual_prompts = multilingual or {}
    tm.system_prompt = {"content": base}
    tm.conversation_history = MagicMock()
    # MagicMock would silently absorb the name-mangled helper call — bind the real one.
    tm._TaskManager__language_directive = lambda label: DIRECTIVE(tm, label)
    return tm


def test_tool_switch_without_note_still_installs_a_directive():
    tm = _tm()
    APPLY(tm, "te")
    content = tm.system_prompt["content"]
    assert content.startswith("## Agent Prompt:")
    assert "## Language note:" in content
    assert "Telugu" in content  # full language name, not just the code
    tm.conversation_history.update_system_prompt.assert_called_once_with(content)


def test_directive_is_replaced_not_accumulated():
    tm = _tm()
    APPLY(tm, "te")
    APPLY(tm, "hi")
    content = tm.system_prompt["content"]
    assert content.count("## Language note:") == 1
    assert "Hindi" in content and "Telugu" not in content


def test_lid_context_note_wins_over_the_generic_directive():
    tm = _tm()
    note = '## Language note:\nThe caller is now speaking Telugu... "చెప్పండి".'
    APPLY(tm, "te", note)
    assert tm.system_prompt["content"].endswith(note)


def test_multilingual_variant_is_used_as_the_base_when_present():
    tm = _tm(multilingual={"te": "TE VARIANT PROMPT"})
    APPLY(tm, "te")
    content = tm.system_prompt["content"]
    assert content.startswith("TE VARIANT PROMPT")
    assert "## Language note:" in content


def test_missing_variant_falls_back_to_current_prompt_with_old_note_stripped():
    tm = _tm(multilingual={"te": "TE VARIANT PROMPT"})
    APPLY(tm, "te")  # installs on the variant
    APPLY(tm, "en")  # no en variant: must reuse the CURRENT prompt minus the te note
    content = tm.system_prompt["content"]
    assert content.startswith("TE VARIANT PROMPT")
    assert content.count("## Language note:") == 1
    assert "English" in content
