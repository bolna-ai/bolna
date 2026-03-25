"""
Unit tests for SSML integration in Azure and Polly synthesizers.

These tests run fully offline — no Azure/Polly API keys required.
All synthesizer instances are created with mocked task_manager_instance
so the internal queue and should_synthesize_response machinery is bypassed.

Run with:
    cd bolna
    pytest tests/tests/test_ssml.py -v
"""

import pytest
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_task_manager():
    """Minimal task manager mock so BaseSynthesizer.__init__ doesn't blow up."""
    tm = MagicMock()
    tm.is_sequence_id_in_current_ids.return_value = True
    return tm


def _make_azure(**kwargs):
    """Instantiate AzureSynthesizer with dummy credentials (no real API call)."""
    from bolna.synthesizer.azure_synthesizer import AzureSynthesizer

    defaults = dict(
        voice="AriaNeural",
        language="en-US",
        model="Neural",
        task_manager_instance=_mock_task_manager(),
        synthesizer_key="fake-key",
        region="eastus",
    )
    defaults.update(kwargs)
    return AzureSynthesizer(**defaults)


def _make_polly(**kwargs):
    """Instantiate PollySynthesizer — no boto3 client is created until a call."""
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
# BaseSynthesizer — SSML utilities
# ===========================================================================

class TestBaseSynthesizerSSMLUtils:
    """Tests for is_ssml / has_inline_ssml / strip_ssml on any synthesizer instance."""

    def setup_method(self):
        self.synth = _make_azure()

    # --- is_ssml ---

    def test_is_ssml_detects_speak_tag(self):
        assert self.synth.is_ssml('<speak version="1.0">Hello</speak>')

    def test_is_ssml_detects_speak_with_namespace(self):
        ssml = (
            '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">'
            '<voice name="AriaNeural">Hi</voice></speak>'
        )
        assert self.synth.is_ssml(ssml)

    def test_is_ssml_false_for_plain_text(self):
        assert not self.synth.is_ssml("Hello, how are you?")

    def test_is_ssml_false_for_inline_tags_only(self):
        assert not self.synth.is_ssml('Call <say-as interpret-as="telephone">9876543210</say-as>')

    # --- has_inline_ssml ---

    def test_has_inline_ssml_break(self):
        assert self.synth.has_inline_ssml('Hello. <break time="500ms"/> Goodbye.')

    def test_has_inline_ssml_say_as(self):
        assert self.synth.has_inline_ssml('<say-as interpret-as="telephone">9876543210</say-as>')

    def test_has_inline_ssml_emphasis(self):
        assert self.synth.has_inline_ssml('This is <emphasis level="strong">important</emphasis>.')

    def test_has_inline_ssml_false_for_plain(self):
        assert not self.synth.has_inline_ssml("Plain text with no tags.")

    # --- strip_ssml ---

    def test_strip_ssml_removes_all_tags(self):
        text = (
            '<speak><voice name="AriaNeural">'
            'Call <say-as interpret-as="telephone">9876543210</say-as>'
            '</voice></speak>'
        )
        assert self.synth.strip_ssml(text) == "Call 9876543210"

    def test_strip_ssml_cleans_break_tag(self):
        result = self.synth.strip_ssml('Hello. <break time="500ms"/> World.')
        # strip_ssml collapses whitespace — the tag is removed, spaces normalized
        assert result == "Hello. World."
        assert "<" not in result

    def test_strip_ssml_plain_text_unchanged(self):
        text = "Just plain text."
        assert self.synth.strip_ssml(text) == text


# ===========================================================================
# AzureSynthesizer — _build_ssml
# ===========================================================================

class TestAzureBuildSSML:

    # --- Returns None when nothing is configured ---

    def test_returns_none_for_plain_text_no_config(self):
        synth = _make_azure()
        assert synth._build_ssml("Hello world.") is None

    def test_returns_none_when_speed_is_1(self):
        synth = _make_azure(speed=1.0)
        assert synth._build_ssml("Hello world.") is None

    # --- Pass-through ---

    def test_passthrough_when_text_is_already_ssml(self):
        synth = _make_azure(style="empathetic")
        raw_ssml = '<speak version="1.0" xml:lang="en-US"><voice name="AriaNeural">Hi</voice></speak>'
        result = synth._build_ssml(raw_ssml)
        assert result == raw_ssml  # returned as-is, no double-wrapping

    # --- Prosody ---

    def test_speed_produces_prosody_rate(self):
        synth = _make_azure(speed=0.9)
        result = synth._build_ssml("Hello.")
        assert result is not None
        assert 'rate="0.9"' in result
        assert "<prosody" in result

    def test_pitch_produces_prosody_pitch(self):
        synth = _make_azure(pitch="high")
        result = synth._build_ssml("Hello.")
        assert 'pitch="high"' in result

    def test_volume_produces_prosody_volume(self):
        synth = _make_azure(volume="loud")
        result = synth._build_ssml("Hello.")
        assert 'volume="loud"' in result

    def test_multiple_prosody_attrs_combined(self):
        synth = _make_azure(speed=0.85, pitch="+5%", volume="+6dB")
        result = synth._build_ssml("Hello.")
        assert 'rate="0.85"' in result
        assert 'pitch="+5%"' in result
        assert 'volume="+6dB"' in result

    # --- Style / role ---

    def test_style_produces_express_as(self):
        synth = _make_azure(style="empathetic")
        result = synth._build_ssml("I understand.")
        assert 'mstts:express-as' in result
        assert 'style="empathetic"' in result

    def test_style_degree_added_when_set(self):
        synth = _make_azure(style="cheerful", style_degree=1.5)
        result = synth._build_ssml("Hello!")
        assert 'styledegree="1.5"' in result

    def test_role_only_produces_express_as(self):
        synth = _make_azure(role="YoungAdultFemale")
        result = synth._build_ssml("Hi there.")
        assert 'mstts:express-as' in result
        assert 'role="YoungAdultFemale"' in result

    def test_style_and_role_combined(self):
        synth = _make_azure(style="calm", role="OlderAdultMale")
        result = synth._build_ssml("Take it easy.")
        assert 'style="calm"' in result
        assert 'role="OlderAdultMale"' in result

    # --- Telephony optimisation ---

    def test_telephony_optimized_adds_effect(self):
        synth = _make_azure(telephony_optimized=True, speed=0.9)
        result = synth._build_ssml("Hello.")
        assert 'effect="eq_telecomhp8k"' in result

    def test_telephony_optimized_alone_triggers_ssml(self):
        synth = _make_azure(telephony_optimized=True)
        result = synth._build_ssml("Hello.")
        assert result is not None
        assert 'effect="eq_telecomhp8k"' in result

    def test_no_telephony_effect_when_false(self):
        synth = _make_azure(speed=0.9, telephony_optimized=False)
        result = synth._build_ssml("Hello.")
        assert "eq_telecomhp8k" not in result

    # --- SSML structure ---

    def test_output_has_speak_root(self):
        synth = _make_azure(style="customerservice")
        result = synth._build_ssml("Hello.")
        assert result.startswith("<speak")
        assert result.endswith("</speak>")

    def test_output_has_mstts_namespace(self):
        synth = _make_azure(style="empathetic")
        result = synth._build_ssml("Hello.")
        assert 'xmlns:mstts=' in result

    def test_voice_name_in_output(self):
        # voice = "AriaNeural", language = "en-US", model = "Neural"
        # → self.voice = f"{language}-{voice}{model}" = "en-US-AriaNeuralNeural"
        synth = _make_azure(style="calm")
        result = synth._build_ssml("Hello.")
        assert 'name="en-US-AriaNeuralNeural"' in result

    # --- Inline SSML pass-through ---

    def test_inline_tags_preserved_in_wrap(self):
        synth = _make_azure(style="empathetic")
        text = 'Call <say-as interpret-as="telephone">9876543210</say-as> now.'
        result = synth._build_ssml(text)
        assert result is not None
        assert '<say-as interpret-as="telephone">9876543210</say-as>' in result
        assert 'style="empathetic"' in result

    def test_plain_text_is_xml_escaped(self):
        synth = _make_azure(speed=0.9)
        result = synth._build_ssml("AT&T is a company. Score: 5 < 10.")
        assert "&amp;" in result
        assert "&lt;" in result

    # --- Nesting order: prosody inside express-as ---

    def test_prosody_nested_inside_express_as(self):
        synth = _make_azure(style="cheerful", speed=0.9)
        result = synth._build_ssml("Hello!")
        prosody_pos = result.index("<prosody")
        express_pos = result.index("mstts:express-as")
        # prosody tag must appear after express-as opening tag
        assert prosody_pos > express_pos


# ===========================================================================
# PollySynthesizer — _build_ssml
# ===========================================================================

class TestPollyBuildSSML:

    # --- Returns None when nothing is configured ---

    def test_returns_none_no_config(self):
        synth = _make_polly()
        assert synth._build_ssml("Hello world.") is None

    # --- Pass-through ---

    def test_passthrough_for_full_ssml(self):
        synth = _make_polly(speaking_rate="slow")
        raw = "<speak>Hello world.</speak>"
        assert synth._build_ssml(raw) == raw

    # --- Rate and volume (all engines) ---

    def test_speaking_rate_neural(self):
        synth = _make_polly(engine="neural", speaking_rate="slow")
        result = synth._build_ssml("Hello.")
        assert result is not None
        assert 'rate="slow"' in result
        assert "<prosody" in result

    def test_volume_neural(self):
        synth = _make_polly(engine="neural", volume="loud")
        result = synth._build_ssml("Hello.")
        assert 'volume="loud"' in result

    def test_rate_and_volume_combined(self):
        synth = _make_polly(engine="neural", speaking_rate="85%", volume="+6dB")
        result = synth._build_ssml("Hello.")
        assert 'rate="85%"' in result
        assert 'volume="+6dB"' in result

    # --- Pitch — Standard engine ONLY ---

    def test_pitch_standard_engine(self):
        synth = _make_polly(engine="standard", pitch="high")
        result = synth._build_ssml("Hello.")
        assert 'pitch="high"' in result

    def test_pitch_ignored_for_neural(self):
        synth = _make_polly(engine="neural", pitch="high")
        result = synth._build_ssml("Hello.")
        assert result is None  # pitch alone does nothing on neural

    def test_pitch_ignored_for_generative(self):
        synth = _make_polly(engine="generative", pitch="+10%")
        result = synth._build_ssml("Hello.")
        assert result is None

    # --- Emphasis — Standard engine ONLY ---

    def test_emphasis_standard(self):
        synth = _make_polly(engine="standard", emphasis="strong")
        result = synth._build_ssml("I really mean it.")
        assert '<emphasis level="strong">' in result

    def test_emphasis_ignored_for_neural(self):
        synth = _make_polly(engine="neural", emphasis="strong")
        result = synth._build_ssml("I really mean it.")
        assert result is None

    # --- Auto-breaths — Standard engine ONLY ---

    def test_auto_breaths_standard(self):
        synth = _make_polly(engine="standard", auto_breaths=True, speaking_rate="slow")
        result = synth._build_ssml("A long passage of text.")
        assert "<amazon:auto-breaths>" in result

    def test_auto_breaths_ignored_for_neural(self):
        synth = _make_polly(engine="neural", auto_breaths=True, speaking_rate="slow")
        result = synth._build_ssml("A long passage of text.")
        assert "auto-breaths" not in (result or "")

    # --- Newscaster — Neural ONLY ---

    def test_newscaster_neural(self):
        synth = _make_polly(engine="neural", newscaster=True, speaking_rate="medium")
        result = synth._build_ssml("Breaking news today.")
        assert '<amazon:domain name="news">' in result

    def test_newscaster_ignored_for_standard(self):
        synth = _make_polly(engine="standard", newscaster=True, speaking_rate="medium")
        result = synth._build_ssml("Breaking news today.")
        assert "domain" not in (result or "")

    # --- SSML structure ---

    def test_output_wrapped_in_speak(self):
        synth = _make_polly(speaking_rate="slow")
        result = synth._build_ssml("Hello.")
        assert result.startswith("<speak>")
        assert result.endswith("</speak>")

    def test_plain_text_xml_escaped(self):
        synth = _make_polly(engine="standard", speaking_rate="slow")
        result = synth._build_ssml("AT&T score: 5 < 10.")
        assert "&amp;" in result
        assert "&lt;" in result

    # --- Standard engine full combination ---

    def test_standard_full_combo(self):
        synth = _make_polly(
            engine="standard",
            speaking_rate="85%",
            volume="loud",
            pitch="+5%",
            emphasis="moderate",
            auto_breaths=True,
        )
        result = synth._build_ssml("This is a test.")
        assert result is not None
        assert 'rate="85%"' in result
        assert 'volume="loud"' in result
        assert 'pitch="+5%"' in result
        assert '<emphasis level="moderate">' in result
        assert "<amazon:auto-breaths>" in result

    # --- Inline SSML preserved ---

    def test_inline_break_preserved(self):
        synth = _make_polly(speaking_rate="slow")
        text = 'Hello. <break time="1s"/> Goodbye.'
        result = synth._build_ssml(text)
        assert '<break time="1s"/>' in result
