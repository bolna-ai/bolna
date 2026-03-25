"""
Unit tests for AzureSynthesizer SSML generation — all features.

Tests run fully offline (no Azure keys needed).
_build_ssml() is a pure string-building function — safe to call without a live SDK.

Run with:
    cd bolna
    pytest tests/tests/test_ssml_azure.py -v
"""

import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_task_manager():
    tm = MagicMock()
    tm.is_sequence_id_in_current_ids.return_value = True
    return tm


def _make_azure(**kwargs):
    """
    Instantiate AzureSynthesizer with dummy credentials.
    Patches the Azure SDK so no real connection is made.
    """
    from bolna.synthesizer.azure_synthesizer import AzureSynthesizer

    defaults = dict(
        voice="Aria",
        language="en-US",
        model="Neural",
        task_manager_instance=_mock_task_manager(),
        synthesizer_key="fake-key",
        region="eastus",
    )
    defaults.update(kwargs)

    with patch("azure.cognitiveservices.speech.SpeechConfig"), \
         patch("azure.cognitiveservices.speech.SpeechSynthesizer"):
        return AzureSynthesizer(**defaults)


# ===========================================================================
# Section 1 — No-op: returns None when SSML is not needed
# ===========================================================================

class TestAzureNoOp:

    def test_plain_text_no_config_returns_none(self):
        assert _make_azure()._build_ssml("Hello world.") is None

    def test_speed_exactly_1_returns_none(self):
        assert _make_azure(speed=1.0)._build_ssml("Hello.") is None

    def test_all_defaults_returns_none(self):
        s = _make_azure(speed=None, pitch=None, volume=None, style=None,
                        role=None, telephony_optimized=False,
                        sentence_silence_ms=None, leading_silence_ms=None,
                        background_audio_url=None, lang=None)
        assert s._build_ssml("Plain text.") is None


# ===========================================================================
# Section 2 — Pass-through (LLM already produced full SSML)
# ===========================================================================

class TestAzurePassThrough:

    RAW = (
        '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" '
        'xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="en-US">'
        '<voice name="en-US-AriaNeuralNeural">Hello.</voice></speak>'
    )

    def test_full_ssml_returned_unchanged(self):
        s = _make_azure(style="empathetic")
        assert s._build_ssml(self.RAW) == self.RAW

    def test_no_double_wrapping(self):
        s = _make_azure(speed=1.2)
        result = s._build_ssml(self.RAW)
        # Should not gain an extra <speak> wrapper
        assert result.count("<speak") == 1

    def test_passthrough_ignores_config(self):
        """Even with every option set, a full SSML is returned as-is."""
        s = _make_azure(speed=1.5, style="cheerful", sentence_silence_ms=300,
                        background_audio_url="https://example.com/bg.mp3")
        assert s._build_ssml(self.RAW) == self.RAW


# ===========================================================================
# Section 3 — Prosody (speed / pitch / volume)
# ===========================================================================

class TestAzureProsody:

    def test_speed_above_1_produces_prosody(self):
        result = _make_azure(speed=1.2)._build_ssml("Hello.")
        assert 'rate="1.2"' in result
        assert "<prosody" in result

    def test_speed_below_1_produces_prosody(self):
        result = _make_azure(speed=0.8)._build_ssml("Hello.")
        assert 'rate="0.8"' in result

    def test_pitch_high(self):
        result = _make_azure(pitch="high")._build_ssml("Hello.")
        assert 'pitch="high"' in result

    def test_pitch_semitones(self):
        result = _make_azure(pitch="+2st")._build_ssml("Hello.")
        assert 'pitch="+2st"' in result

    def test_volume_loud(self):
        result = _make_azure(volume="loud")._build_ssml("Hello.")
        assert 'volume="loud"' in result

    def test_volume_db(self):
        result = _make_azure(volume="+6dB")._build_ssml("Hello.")
        assert 'volume="+6dB"' in result

    def test_all_three_prosody_combined(self):
        result = _make_azure(speed=0.85, pitch="+5%", volume="+6dB")._build_ssml("Hello.")
        assert 'rate="0.85"' in result
        assert 'pitch="+5%"' in result
        assert 'volume="+6dB"' in result

    def test_prosody_is_inside_voice_tag(self):
        result = _make_azure(speed=0.9)._build_ssml("Hello.")
        voice_pos = result.index("<voice")
        prosody_pos = result.index("<prosody")
        assert prosody_pos > voice_pos


# ===========================================================================
# Section 4 — Speaking style and role (mstts:express-as)
# ===========================================================================

class TestAzureStyleRole:

    def test_style_produces_express_as(self):
        result = _make_azure(style="empathetic")._build_ssml("I understand.")
        assert "mstts:express-as" in result
        assert 'style="empathetic"' in result

    def test_style_degree_present(self):
        result = _make_azure(style="cheerful", style_degree=1.5)._build_ssml("Hello!")
        assert 'styledegree="1.5"' in result

    def test_style_degree_absent_when_none(self):
        result = _make_azure(style="calm")._build_ssml("Hello.")
        assert "styledegree" not in result

    def test_role_only_no_style(self):
        result = _make_azure(role="YoungAdultFemale")._build_ssml("Hi there.")
        assert "mstts:express-as" in result
        assert 'role="YoungAdultFemale"' in result
        assert "style=" not in result

    def test_style_and_role_combined(self):
        result = _make_azure(style="calm", role="OlderAdultMale")._build_ssml("Relax.")
        assert 'style="calm"' in result
        assert 'role="OlderAdultMale"' in result

    def test_customerservice_style(self):
        result = _make_azure(style="customerservice")._build_ssml("How can I help?")
        assert 'style="customerservice"' in result

    def test_prosody_nested_inside_express_as(self):
        result = _make_azure(style="cheerful", speed=0.9)._build_ssml("Hello!")
        prosody_pos = result.index("<prosody")
        express_pos = result.index("mstts:express-as")
        assert prosody_pos > express_pos


# ===========================================================================
# Section 5 — Telephony optimisation
# ===========================================================================

class TestAzureTelephony:

    def test_effect_added_when_true(self):
        result = _make_azure(telephony_optimized=True, speed=0.9)._build_ssml("Hello.")
        assert 'effect="eq_telecomhp8k"' in result

    def test_telephony_alone_triggers_ssml(self):
        result = _make_azure(telephony_optimized=True)._build_ssml("Hello.")
        assert result is not None
        assert 'effect="eq_telecomhp8k"' in result

    def test_no_effect_when_false(self):
        result = _make_azure(speed=0.9, telephony_optimized=False)._build_ssml("Hello.")
        assert "eq_telecomhp8k" not in result

    def test_effect_on_voice_tag(self):
        result = _make_azure(telephony_optimized=True)._build_ssml("Hello.")
        assert '<voice name=' in result
        assert 'effect="eq_telecomhp8k"' in result


# ===========================================================================
# Section 6 — Sentence-boundary and leading silence
# ===========================================================================

class TestAzureSilence:

    def test_sentence_silence_tag_present(self):
        result = _make_azure(sentence_silence_ms=200, speed=1.0)._build_ssml("Hello. How are you?")
        # sentence_silence_ms alone should trigger SSML (even without other config)
        # It's injected before the voice tag
        assert 'mstts:silence' in result
        assert 'type="Sentenceboundary"' in result
        assert 'value="200ms"' in result

    def test_leading_silence_tag_present(self):
        result = _make_azure(leading_silence_ms=150, speed=1.0)._build_ssml("Hello.")
        assert 'type="Leading"' in result
        assert 'value="150ms"' in result

    def test_both_silence_tags_present(self):
        result = _make_azure(sentence_silence_ms=300, leading_silence_ms=100, speed=1.0)._build_ssml("Hi.")
        assert 'type="Sentenceboundary"' in result
        assert 'type="Leading"' in result

    def test_silence_tags_appear_before_voice_tag(self):
        result = _make_azure(sentence_silence_ms=200, speed=1.0)._build_ssml("Hello.")
        silence_pos = result.index("mstts:silence")
        voice_pos = result.index("<voice")
        assert silence_pos < voice_pos

    def test_no_silence_tag_when_zero(self):
        result = _make_azure(speed=0.9, sentence_silence_ms=None)._build_ssml("Hello.")
        assert "Sentenceboundary" not in result

    def test_sentence_silence_alone_triggers_ssml(self):
        result = _make_azure(sentence_silence_ms=200)._build_ssml("Hello world.")
        assert result is not None


# ===========================================================================
# Section 7 — Background audio
# ===========================================================================

class TestAzureBackgroundAudio:

    BG_URL = "https://example.com/background.mp3"

    def test_background_audio_src_present(self):
        result = _make_azure(background_audio_url=self.BG_URL, speed=1.0)._build_ssml("Hello.")
        assert "mstts:backgroundaudio" in result
        assert f'src="{self.BG_URL}"' in result

    def test_background_volume_included(self):
        result = _make_azure(background_audio_url=self.BG_URL,
                             background_audio_volume=0.3, speed=1.0)._build_ssml("Hello.")
        assert 'volume="0.3"' in result

    def test_background_fadein_included(self):
        result = _make_azure(background_audio_url=self.BG_URL,
                             background_audio_fadein=2000, speed=1.0)._build_ssml("Hello.")
        assert 'fadein="2000ms"' in result

    def test_background_fadeout_included(self):
        result = _make_azure(background_audio_url=self.BG_URL,
                             background_audio_fadeout=3000, speed=1.0)._build_ssml("Hello.")
        assert 'fadeout="3000ms"' in result

    def test_all_background_attrs_combined(self):
        result = _make_azure(
            background_audio_url=self.BG_URL,
            background_audio_volume=0.5,
            background_audio_fadein=1000,
            background_audio_fadeout=2000,
            speed=1.0,
        )._build_ssml("Hello.")
        assert f'src="{self.BG_URL}"' in result
        assert 'volume="0.5"' in result
        assert 'fadein="1000ms"' in result
        assert 'fadeout="2000ms"' in result

    def test_background_tag_is_inside_speak(self):
        result = _make_azure(background_audio_url=self.BG_URL, speed=1.0)._build_ssml("Hello.")
        speak_start = result.index("<speak")
        bg_pos = result.index("mstts:backgroundaudio")
        assert bg_pos > speak_start

    def test_no_background_tag_when_url_absent(self):
        result = _make_azure(speed=0.9)._build_ssml("Hello.")
        assert "backgroundaudio" not in result

    def test_background_audio_alone_triggers_ssml(self):
        result = _make_azure(background_audio_url=self.BG_URL)._build_ssml("Hello world.")
        assert result is not None
        assert "mstts:backgroundaudio" in result


# ===========================================================================
# Section 8 — Language switching (lang)
# ===========================================================================

class TestAzureLanguageSwitching:

    def test_lang_different_from_primary_adds_lang_tag(self):
        result = _make_azure(style="customerservice", lang="hi-IN")._build_ssml("Namaste.")
        assert '<lang xml:lang="hi-IN">' in result

    def test_lang_same_as_primary_no_extra_tag(self):
        # language="en-US", lang="en-US" → no redundant <lang> wrapper
        result = _make_azure(style="calm", lang="en-US")._build_ssml("Hello.")
        assert result.count("<lang") == 0

    def test_lang_alone_triggers_ssml_when_different(self):
        # lang != primary language counts as a config trigger on its own
        result = _make_azure(lang="hi-IN")._build_ssml("Hello.")
        assert result is not None
        assert '<lang xml:lang="hi-IN">' in result

    def test_lang_with_style_triggers_ssml(self):
        result = _make_azure(style="customerservice", lang="hi-IN")._build_ssml("Hello.")
        assert result is not None
        assert '<lang xml:lang="hi-IN">' in result

    def test_lang_tag_inside_voice_tag(self):
        result = _make_azure(style="calm", lang="fr-FR")._build_ssml("Bonjour.")
        voice_pos = result.index("<voice")
        lang_pos = result.index('<lang xml:lang="fr-FR">')
        assert lang_pos > voice_pos


# ===========================================================================
# Section 9 — Inline SSML handling
# ===========================================================================

class TestAzureInlineSSML:

    def test_inline_break_preserved(self):
        s = _make_azure(style="empathetic")
        result = s._build_ssml('Hello. <break time="500ms"/> How are you?')
        assert '<break time="500ms"/>' in result

    def test_inline_say_as_preserved(self):
        s = _make_azure(speed=0.9)
        text = 'Call <say-as interpret-as="telephone">9876543210</say-as> now.'
        result = s._build_ssml(text)
        assert '<say-as interpret-as="telephone">9876543210</say-as>' in result

    def test_inline_tags_trigger_ssml_without_other_config(self):
        # Inline tags alone should cause SSML wrapping even without prosody/style
        s = _make_azure()
        text = 'Read <say-as interpret-as="characters">SSML</say-as> aloud.'
        result = s._build_ssml(text)
        assert result is not None
        assert "<speak" in result

    def test_plain_text_is_xml_escaped(self):
        result = _make_azure(speed=0.9)._build_ssml("AT&T score: 5 < 10.")
        assert "&amp;" in result
        assert "&lt;" in result


# ===========================================================================
# Section 10 — SSML structure validation
# ===========================================================================

class TestAzureSSMLStructure:

    def test_output_starts_with_speak(self):
        result = _make_azure(style="customerservice")._build_ssml("Hello.")
        assert result.startswith("<speak")

    def test_output_ends_with_speak(self):
        result = _make_azure(style="customerservice")._build_ssml("Hello.")
        assert result.endswith("</speak>")

    def test_mstts_namespace_present(self):
        result = _make_azure(style="empathetic")._build_ssml("Hello.")
        assert 'xmlns:mstts=' in result

    def test_synthesis_namespace_present(self):
        result = _make_azure(speed=0.9)._build_ssml("Hello.")
        assert 'xmlns="http://www.w3.org/2001/10/synthesis"' in result

    def test_xml_lang_matches_language(self):
        result = _make_azure(speed=0.9)._build_ssml("Hello.")
        assert 'xml:lang="en-US"' in result

    def test_voice_tag_present(self):
        result = _make_azure(speed=0.9)._build_ssml("Hello.")
        assert '<voice name=' in result


# ===========================================================================
# Section 11 — Full combination scenarios
# ===========================================================================

class TestAzureFullCombinations:

    def test_telephony_agent_full(self):
        """Typical call-centre agent: style + speed + telephony optimised + sentence silence."""
        result = _make_azure(
            style="customerservice",
            style_degree=1.2,
            speed=1.1,
            telephony_optimized=True,
            sentence_silence_ms=150,
        )._build_ssml("Thank you for calling. How can I help you today?")
        assert result is not None
        assert 'style="customerservice"' in result
        assert 'styledegree="1.2"' in result
        assert 'rate="1.1"' in result
        assert 'effect="eq_telecomhp8k"' in result
        assert 'type="Sentenceboundary"' in result

    def test_multilingual_agent(self):
        """Agent with primary language en-US, switching spoken content to Hindi."""
        result = _make_azure(
            style="empathetic",
            lang="hi-IN",
            speed=0.95,
        )._build_ssml("Namaste, main aapki madad kar sakta hun.")
        assert '<lang xml:lang="hi-IN">' in result
        assert 'style="empathetic"' in result
        assert 'rate="0.95"' in result

    def test_background_music_agent(self):
        """Agent with background music and fade-in/out."""
        result = _make_azure(
            background_audio_url="https://cdn.example.com/hold_music.mp3",
            background_audio_volume=0.2,
            background_audio_fadein=1000,
            background_audio_fadeout=2000,
            style="cheerful",
            speed=1.0,
        )._build_ssml("Please hold while we connect you.")
        assert "mstts:backgroundaudio" in result
        assert 'volume="0.2"' in result
        assert 'fadein="1000ms"' in result
        assert 'fadeout="2000ms"' in result
        assert 'style="cheerful"' in result

    def test_drama_production_agent(self):
        """All possible features enabled simultaneously."""
        result = _make_azure(
            style="empathetic",
            style_degree=2.0,
            role="OlderAdultFemale",
            speed=0.85,
            pitch="+2st",
            volume="+3dB",
            telephony_optimized=True,
            sentence_silence_ms=200,
            leading_silence_ms=100,
            background_audio_url="https://cdn.example.com/bg.mp3",
            background_audio_volume=0.15,
            lang="en-GB",
        )._build_ssml("Good evening. Welcome to the experience.")
        assert result is not None
        for fragment in [
            'style="empathetic"', 'role="OlderAdultFemale"',
            'rate="0.85"', 'pitch="+2st"', 'volume="+3dB"',
            'eq_telecomhp8k', 'Sentenceboundary', 'Leading',
            'backgroundaudio', 'lang xml:lang="en-GB"',
        ]:
            assert fragment in result, f"Expected '{fragment}' in SSML output"
