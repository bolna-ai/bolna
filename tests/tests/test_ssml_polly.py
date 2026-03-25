"""
Unit tests for PollySynthesizer SSML generation — all features.

Tests run fully offline (no AWS credentials needed).
_build_ssml() is pure string-building — safe to call without boto3.

Run with:
    cd bolna
    pytest tests/tests/test_ssml_polly.py -v
"""

import pytest
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_task_manager():
    tm = MagicMock()
    tm.is_sequence_id_in_current_ids.return_value = True
    return tm


def _make_polly(**kwargs):
    from bolna.synthesizer.polly_synthesizer import PollySynthesizer

    defaults = dict(
        voice="Joanna",
        language="en-US",
        engine="neural",
        task_manager_instance=_mock_task_manager(),
    )
    defaults.update(kwargs)
    return PollySynthesizer(**defaults)


# ===========================================================================
# Section 1 — No-op: returns None when nothing is configured
# ===========================================================================

class TestPollyNoOp:

    def test_plain_text_no_config_returns_none(self):
        assert _make_polly()._build_ssml("Hello world.") is None

    def test_neural_pitch_alone_returns_none(self):
        # pitch is Standard only — neural with only pitch returns None
        assert _make_polly(engine="neural", pitch="high")._build_ssml("Hello.") is None

    def test_neural_emphasis_alone_returns_none(self):
        assert _make_polly(engine="neural", emphasis="strong")._build_ssml("Hi.") is None

    def test_neural_auto_breaths_alone_returns_none(self):
        assert _make_polly(engine="neural", auto_breaths=True)._build_ssml("Hi.") is None

    def test_neural_standard_only_effects_return_none(self):
        # whispered, drc, vocal_tract_length, soft_phonation are Standard only
        s = _make_polly(engine="neural", whispered=True, drc=True)
        assert s._build_ssml("Hello.") is None

    def test_generative_pitch_returns_none(self):
        assert _make_polly(engine="generative", pitch="+10%")._build_ssml("Hello.") is None

    def test_all_defaults_returns_none(self):
        s = _make_polly(engine="neural", speaking_rate=None, volume=None,
                        pitch=None, emphasis=None, auto_breaths=False,
                        newscaster=False, conversational=False,
                        whispered=False, drc=False, vocal_tract_length=None,
                        soft_phonation=False, lang=None)
        assert s._build_ssml("Plain text.") is None


# ===========================================================================
# Section 2 — Pass-through (LLM already produced full SSML)
# ===========================================================================

class TestPollyPassThrough:

    RAW = "<speak>Hello world.</speak>"
    RAW_NAMESPACED = (
        '<speak xmlns:amazon="http://www.amazon.com/2021/ssml">'
        "<amazon:domain name=\"news\">Breaking news.</amazon:domain>"
        "</speak>"
    )

    def test_full_ssml_returned_unchanged(self):
        s = _make_polly(speaking_rate="slow")
        assert s._build_ssml(self.RAW) == self.RAW

    def test_no_double_speak_wrapping(self):
        s = _make_polly(speaking_rate="slow")
        result = s._build_ssml(self.RAW)
        assert result.count("<speak") == 1

    def test_passthrough_with_namespace_returned_unchanged(self):
        s = _make_polly(newscaster=True, speaking_rate="medium")
        assert s._build_ssml(self.RAW_NAMESPACED) == self.RAW_NAMESPACED

    def test_passthrough_ignores_all_config(self):
        s = _make_polly(engine="standard", speaking_rate="fast", pitch="high",
                        emphasis="strong", auto_breaths=True, whispered=True,
                        drc=True, vocal_tract_length="+15%", lang="hi-IN")
        assert s._build_ssml(self.RAW) == self.RAW


# ===========================================================================
# Section 3 — Rate and volume (all engines)
# ===========================================================================

class TestPollyProsodyAllEngines:

    @pytest.mark.parametrize("engine", ["neural", "standard", "long-form", "generative"])
    def test_speaking_rate_all_engines(self, engine):
        result = _make_polly(engine=engine, speaking_rate="slow")._build_ssml("Hello.")
        assert 'rate="slow"' in result

    @pytest.mark.parametrize("engine", ["neural", "standard", "long-form", "generative"])
    def test_volume_all_engines(self, engine):
        result = _make_polly(engine=engine, volume="loud")._build_ssml("Hello.")
        assert 'volume="loud"' in result

    def test_rate_percent_value(self):
        result = _make_polly(speaking_rate="85%")._build_ssml("Hello.")
        assert 'rate="85%"' in result

    def test_volume_db_value(self):
        result = _make_polly(volume="+6dB")._build_ssml("Hello.")
        assert 'volume="+6dB"' in result

    def test_rate_and_volume_combined(self):
        result = _make_polly(speaking_rate="120%", volume="loud")._build_ssml("Hello.")
        assert 'rate="120%"' in result
        assert 'volume="loud"' in result

    def test_prosody_tag_wrapped_around_content(self):
        result = _make_polly(speaking_rate="slow")._build_ssml("Hello.")
        assert "<prosody" in result
        assert "Hello." in result or "Hello." in result


# ===========================================================================
# Section 4 — Pitch (Standard engine ONLY)
# ===========================================================================

class TestPollyPitch:

    def test_pitch_standard_engine(self):
        result = _make_polly(engine="standard", pitch="high")._build_ssml("Hello.")
        assert 'pitch="high"' in result

    def test_pitch_percent_standard(self):
        result = _make_polly(engine="standard", pitch="+10%")._build_ssml("Hello.")
        assert 'pitch="+10%"' in result

    def test_pitch_ignored_neural(self):
        result = _make_polly(engine="neural", pitch="high")._build_ssml("Hello.")
        assert result is None

    def test_pitch_ignored_long_form(self):
        result = _make_polly(engine="long-form", pitch="low")._build_ssml("Hello.")
        assert result is None

    def test_pitch_ignored_generative(self):
        result = _make_polly(engine="generative", pitch="+5%")._build_ssml("Hello.")
        assert result is None

    def test_pitch_with_rate_standard(self):
        result = _make_polly(engine="standard", speaking_rate="slow", pitch="low")._build_ssml("Hello.")
        assert 'pitch="low"' in result
        assert 'rate="slow"' in result


# ===========================================================================
# Section 5 — Emphasis (Standard ONLY)
# ===========================================================================

class TestPollyEmphasis:

    @pytest.mark.parametrize("level", ["strong", "moderate", "reduced"])
    def test_emphasis_standard(self, level):
        result = _make_polly(engine="standard", emphasis=level, speaking_rate="medium")._build_ssml("Pay attention.")
        assert f'<emphasis level="{level}">' in result

    def test_emphasis_ignored_neural(self):
        result = _make_polly(engine="neural", emphasis="strong", speaking_rate="medium")._build_ssml("Pay attention.")
        assert "emphasis" not in (result or "")

    def test_emphasis_alone_on_standard_triggers_ssml(self):
        result = _make_polly(engine="standard", emphasis="strong", speaking_rate="slow")._build_ssml("Hello.")
        assert result is not None


# ===========================================================================
# Section 6 — Auto-breaths (Standard ONLY)
# ===========================================================================

class TestPollyAutoBreaths:

    def test_auto_breaths_standard(self):
        result = _make_polly(engine="standard", auto_breaths=True, speaking_rate="slow")._build_ssml("A long read.")
        assert "<amazon:auto-breaths>" in result

    def test_auto_breaths_ignored_neural(self):
        result = _make_polly(engine="neural", auto_breaths=True, speaking_rate="slow")._build_ssml("A long read.")
        assert "auto-breaths" not in (result or "")

    def test_auto_breaths_wraps_body(self):
        result = _make_polly(engine="standard", auto_breaths=True, speaking_rate="slow")._build_ssml("Content.")
        assert "<amazon:auto-breaths>" in result
        assert "</amazon:auto-breaths>" in result


# ===========================================================================
# Section 7 — Whispered voice (Standard ONLY)
# ===========================================================================

class TestPollyWhispered:

    def test_whispered_standard_adds_effect(self):
        result = _make_polly(engine="standard", whispered=True, speaking_rate="slow")._build_ssml("Can you hear me?")
        assert '<amazon:effect name="whispered">' in result

    def test_whispered_ignored_neural(self):
        result = _make_polly(engine="neural", whispered=True, speaking_rate="slow")._build_ssml("Hello.")
        assert "whispered" not in (result or "")

    def test_whispered_ignored_generative(self):
        result = _make_polly(engine="generative", whispered=True, speaking_rate="slow")._build_ssml("Hello.")
        assert "whispered" not in (result or "")

    def test_whispered_alone_standard_triggers_ssml(self):
        # whispered alone on standard (no rate/volume) is a Standard-only feature
        result = _make_polly(engine="standard", whispered=True)._build_ssml("Secret message.")
        assert result is not None
        assert "whispered" in result


# ===========================================================================
# Section 8 — Dynamic Range Compression / DRC (Standard ONLY)
# ===========================================================================

class TestPollyDRC:

    def test_drc_standard_adds_effect(self):
        result = _make_polly(engine="standard", drc=True, speaking_rate="slow")._build_ssml("Hello.")
        assert '<amazon:effect name="drc">' in result

    def test_drc_ignored_neural(self):
        result = _make_polly(engine="neural", drc=True, speaking_rate="slow")._build_ssml("Hello.")
        assert "drc" not in (result or "")

    def test_drc_wraps_content(self):
        result = _make_polly(engine="standard", drc=True, speaking_rate="slow")._build_ssml("Hello.")
        assert "<amazon:effect name=\"drc\">" in result
        assert "</amazon:effect>" in result


# ===========================================================================
# Section 9 — Vocal tract length (Standard ONLY)
# ===========================================================================

class TestPollyVocalTractLength:

    @pytest.mark.parametrize("vtl", ["+15%", "-10%", "+5%", "0%"])
    def test_vtl_standard_engine(self, vtl):
        result = _make_polly(engine="standard", vocal_tract_length=vtl, speaking_rate="slow")._build_ssml("Hello.")
        assert f'vocal-tract-length="{vtl}"' in result

    def test_vtl_ignored_neural(self):
        result = _make_polly(engine="neural", vocal_tract_length="+15%", speaking_rate="slow")._build_ssml("Hello.")
        assert "vocal-tract-length" not in (result or "")

    def test_vtl_alone_standard_triggers_ssml(self):
        result = _make_polly(engine="standard", vocal_tract_length="+10%")._build_ssml("Hello.")
        assert result is not None
        assert "vocal-tract-length" in result


# ===========================================================================
# Section 10 — Soft phonation (Standard ONLY)
# ===========================================================================

class TestPollySoftPhonation:

    def test_soft_phonation_standard(self):
        result = _make_polly(engine="standard", soft_phonation=True, speaking_rate="slow")._build_ssml("Hello.")
        assert 'phonation="soft"' in result

    def test_soft_phonation_ignored_neural(self):
        result = _make_polly(engine="neural", soft_phonation=True, speaking_rate="slow")._build_ssml("Hello.")
        assert "phonation" not in (result or "")

    def test_soft_phonation_wraps_content(self):
        result = _make_polly(engine="standard", soft_phonation=True, speaking_rate="slow")._build_ssml("Hello.")
        assert "<amazon:effect phonation=\"soft\">" in result
        assert "</amazon:effect>" in result


# ===========================================================================
# Section 11 — Newscaster domain (Neural ONLY)
# ===========================================================================

class TestPollyNewscaster:

    def test_newscaster_neural(self):
        result = _make_polly(engine="neural", newscaster=True, speaking_rate="medium")._build_ssml("Breaking news.")
        assert '<amazon:domain name="news">' in result

    def test_newscaster_ignored_standard(self):
        result = _make_polly(engine="standard", newscaster=True, speaking_rate="medium")._build_ssml("Breaking news.")
        assert "domain" not in (result or "")

    def test_newscaster_alone_neural_triggers_ssml(self):
        result = _make_polly(engine="neural", newscaster=True)._build_ssml("In today's headlines.")
        assert result is not None
        assert '<amazon:domain name="news">' in result


# ===========================================================================
# Section 12 — Conversational domain (Neural ONLY)
# ===========================================================================

class TestPollyConversational:

    def test_conversational_neural(self):
        result = _make_polly(engine="neural", conversational=True, speaking_rate="medium")._build_ssml("Hey, what's up?")
        assert '<amazon:domain name="conversational">' in result

    def test_conversational_ignored_standard(self):
        result = _make_polly(engine="standard", conversational=True, speaking_rate="medium")._build_ssml("Hey.")
        assert "conversational" not in (result or "")

    def test_conversational_alone_neural_triggers_ssml(self):
        result = _make_polly(engine="neural", conversational=True)._build_ssml("What's going on?")
        assert result is not None

    def test_newscaster_takes_priority_over_conversational(self):
        """When both are set, newscaster wins (first in if-elif chain)."""
        result = _make_polly(engine="neural", newscaster=True, conversational=True,
                             speaking_rate="medium")._build_ssml("Hello.")
        assert '<amazon:domain name="news">' in result
        assert "conversational" not in result


# ===========================================================================
# Section 13 — Language switching (lang)
# ===========================================================================

class TestPollyLanguageSwitching:

    @pytest.mark.parametrize("engine", ["neural", "standard"])
    def test_lang_all_engines(self, engine):
        result = _make_polly(engine=engine, speaking_rate="medium", lang="hi-IN")._build_ssml("Namaste.")
        assert '<lang xml:lang="hi-IN">' in result

    def test_lang_alone_triggers_ssml(self):
        result = _make_polly(lang="hi-IN")._build_ssml("Namaste.")
        assert result is not None
        assert '<lang xml:lang="hi-IN">' in result

    def test_lang_wraps_content(self):
        result = _make_polly(lang="fr-FR")._build_ssml("Bonjour.")
        assert '<lang xml:lang="fr-FR">Bonjour.</lang>' in result

    def test_lang_tag_inside_prosody(self):
        """lang wraps the raw content; prosody wraps the lang block."""
        result = _make_polly(speaking_rate="slow", lang="hi-IN")._build_ssml("Hello.")
        lang_pos = result.index('<lang xml:lang="hi-IN">')
        prosody_pos = result.index("<prosody")
        # lang is inner; prosody wraps it
        assert lang_pos > prosody_pos


# ===========================================================================
# Section 14 — Inline SSML handling
# ===========================================================================

class TestPollyInlineSSML:

    def test_inline_break_preserved(self):
        result = _make_polly(speaking_rate="slow")._build_ssml('Hello. <break time="1s"/> Goodbye.')
        assert '<break time="1s"/>' in result

    def test_inline_say_as_preserved(self):
        result = _make_polly(speaking_rate="slow")._build_ssml(
            'Call <say-as interpret-as="telephone">9876543210</say-as> now.'
        )
        assert '<say-as interpret-as="telephone">9876543210</say-as>' in result

    def test_inline_tags_trigger_ssml_without_other_config(self):
        s = _make_polly()
        result = s._build_ssml('Read <say-as interpret-as="characters">SSML</say-as> aloud.')
        assert result is not None
        assert "<speak>" in result

    def test_plain_text_xml_escaped(self):
        result = _make_polly(engine="standard", speaking_rate="slow")._build_ssml("AT&T score: 5 < 10.")
        assert "&amp;" in result
        assert "&lt;" in result

    def test_inline_text_not_double_escaped(self):
        # If text already has inline tags, it should not be run through sax.escape
        result = _make_polly(speaking_rate="slow")._build_ssml('Hello <break time="1s"/> world.')
        assert "<break" in result  # tag preserved, not escaped


# ===========================================================================
# Section 15 — SSML structure validation
# ===========================================================================

class TestPollySSMLStructure:

    def test_output_starts_with_speak(self):
        result = _make_polly(speaking_rate="slow")._build_ssml("Hello.")
        assert result.startswith("<speak>")

    def test_output_ends_with_speak(self):
        result = _make_polly(speaking_rate="slow")._build_ssml("Hello.")
        assert result.endswith("</speak>")

    def test_content_inside_speak(self):
        result = _make_polly(speaking_rate="slow")._build_ssml("Hello world.")
        assert "Hello world." in result


# ===========================================================================
# Section 16 — Full combination scenarios
# ===========================================================================

class TestPollyFullCombinations:

    def test_standard_engine_full_combo(self):
        """All Standard-engine features combined."""
        result = _make_polly(
            engine="standard",
            speaking_rate="85%",
            volume="loud",
            pitch="+5%",
            emphasis="moderate",
            auto_breaths=True,
            whispered=False,
            drc=True,
            vocal_tract_length="+10%",
            soft_phonation=True,
        )._build_ssml("This is a full standard test.")
        assert result is not None
        for fragment in [
            'rate="85%"', 'volume="loud"', 'pitch="+5%"',
            '<emphasis level="moderate">',
            '<amazon:auto-breaths>',
            '<amazon:effect name="drc">',
            'vocal-tract-length="+10%"',
            'phonation="soft"',
        ]:
            assert fragment in result, f"Expected '{fragment}' in SSML output"

    def test_neural_newscaster_full(self):
        """Neural newscaster with prosody and language switch."""
        result = _make_polly(
            engine="neural",
            speaking_rate="medium",
            volume="+3dB",
            newscaster=True,
            lang="en-GB",
        )._build_ssml("Good evening. In today's top stories.")
        assert '<amazon:domain name="news">' in result
        assert 'rate="medium"' in result
        assert 'volume="+3dB"' in result
        assert '<lang xml:lang="en-GB">' in result

    def test_neural_conversational_full(self):
        """Neural conversational style with prosody."""
        result = _make_polly(
            engine="neural",
            speaking_rate="120%",
            volume="x-loud",
            conversational=True,
        )._build_ssml("Hey there! What's going on?")
        assert '<amazon:domain name="conversational">' in result
        assert 'rate="120%"' in result
        assert 'volume="x-loud"' in result

    def test_standard_whispered_with_drc_and_vtl(self):
        """Whispered voice with DRC and vocal tract on standard engine."""
        result = _make_polly(
            engine="standard",
            speaking_rate="slow",
            whispered=True,
            drc=True,
            vocal_tract_length="+20%",
        )._build_ssml("Can you hear this whisper?")
        assert '<amazon:effect name="whispered">' in result
        assert '<amazon:effect name="drc">' in result
        assert 'vocal-tract-length="+20%"' in result

    def test_inline_ssml_with_neural_domain(self):
        """Inline break tags preserved inside newscaster domain."""
        result = _make_polly(
            engine="neural",
            speaking_rate="medium",
            newscaster=True,
        )._build_ssml('The weather today. <break time="500ms"/> Expect sunshine.')
        assert '<break time="500ms"/>' in result
        assert '<amazon:domain name="news">' in result

    def test_multilingual_standard(self):
        """Standard engine with language switching and prosody."""
        result = _make_polly(
            engine="standard",
            speaking_rate="90%",
            lang="hi-IN",
            emphasis="moderate",
        )._build_ssml("Namaste, aap kaise hain?")
        assert '<lang xml:lang="hi-IN">' in result
        assert '<emphasis level="moderate">' in result
        assert 'rate="90%"' in result
