"""Polly TTS: voice-name normalization strips diacritics so an accented voice ID configured
via the dashboard (e.g. AWS Polly's own "Léa"/"Céline" French voices) still matches Polly's
ASCII-only VoiceId values instead of failing the synthesize_speech call every turn."""

import pytest

from bolna.synthesizer.polly_synthesizer import PollySynthesizer


# ----------------------------------------------------------------------
# _resolve_voice
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "given,expected",
    [
        ("Joanna", "Joanna"),  # already-ASCII voice IDs pass through unchanged
        ("Léa", "Lea"),  # precomposed accent (NFC) is stripped
        ("Celiné", "Celine"),  # an already-decomposed combining mark is stripped too
        ("Amélie", "Amelie"),  # multiple accented characters in one name
        ("", ""),  # empty string stays empty rather than raising
    ],
)
def test_resolve_voice_strips_combining_marks(given, expected):
    assert PollySynthesizer._resolve_voice(given) == expected


def test_resolve_voice_is_idempotent():
    # A voice loaded from cache/config a second time must resolve to the same value, or a
    # re-resolved name could silently diverge from the one used on the first call.
    once = PollySynthesizer._resolve_voice("Léa")
    twice = PollySynthesizer._resolve_voice(once)
    assert once == twice == "Lea"


# ----------------------------------------------------------------------
# Construction — the resolved voice is what actually reaches AWS
# ----------------------------------------------------------------------


def test_constructor_resolves_voice_before_storing_it():
    """PollySynthesizer.voice is what gets sent as VoiceId in _generate_http — if the raw,
    unresolved name were stored instead, an accented voice from agent config would fail the
    Polly API call for every turn of the call."""
    s = PollySynthesizer(voice="Léa", language="fr-FR", caching=False)
    assert s.voice == "Lea"


def test_constructor_leaves_plain_ascii_voice_untouched():
    s = PollySynthesizer(voice="Matthew", language="en-US", caching=False)
    assert s.voice == "Matthew"
