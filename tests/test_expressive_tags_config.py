"""The expressive_tags setting: off by default, and only honoured where tags are supported."""

from bolna.models import Synthesizer

CARTESIA = {"voice": "Neha", "voice_id": "v1", "model": "sonic-3.6", "language": "en"}
SARVAM = {"voice": "Pooja", "voice_id": "pooja", "model": "bulbul:v3", "language": "hi"}


def test_expressive_tags_is_off_by_default():
    assert Synthesizer(provider="cartesia", provider_config=CARTESIA).expressive_tags is False


def test_expressive_tags_is_kept_for_cartesia_and_forced_off_for_other_providers():
    assert Synthesizer(provider="cartesia", provider_config=CARTESIA, expressive_tags=True).expressive_tags is True
    assert Synthesizer(provider="sarvam", provider_config=SARVAM, expressive_tags=True).expressive_tags is False


def test_expressive_tags_stays_off_when_the_llm_cut_is_too_small_to_keep_a_tag_whole():
    for size, kept in ((1, False), (19, False), (20, True), (40, True)):
        synth = Synthesizer(provider="cartesia", provider_config=CARTESIA, expressive_tags=True, buffer_size=size)
        assert synth.expressive_tags is kept
