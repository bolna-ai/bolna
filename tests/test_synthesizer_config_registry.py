"""One config class per synthesizer provider, resolved through SYNTHESIZER_CONFIG_MODELS."""

import pytest

from bolna.enums import SynthesizerProvider
from bolna.models import (
    SYNTHESIZER_CONFIG_MODELS,
    CartesiaConfig,
    ElevenLabsConfig,
    GeminiConfig,
    PixaConfig,
    RimeConfig,
    SarvamConfig,
    SmallestConfig,
    StandardVoiceConfig,
    Synthesizer,
)

# provider -> {field: is_required}. Stored agent snapshots are validated against these classes,
# so a field silently appearing, vanishing, or changing requiredness breaks existing agents.
EXPECTED_FIELDS = {
    "polly": {"voice": True, "engine": True, "language": True},
    "elevenlabs": {
        "voice": True,
        "voice_id": True,
        "model": True,
        "temperature": False,
        "similarity_boost": False,
        "speed": False,
        "style": False,
    },
    "openai": {"voice": True, "model": True},
    "deepgram": {"voice_id": True, "voice": True, "model": True},
    "azuretts": {"voice": True, "model": True, "language": True, "speed": False},
    "cartesia": {"voice": True, "voice_id": True, "model": True, "language": True, "speed": False},
    "smallest": {"voice": True, "voice_id": True, "model": True, "language": True},
    "sarvam": {"voice": True, "voice_id": True, "model": True, "language": True, "speed": False},
    "rime": {"voice": True, "voice_id": True, "model": True, "language": True},
    "pixa": {
        "voice": True,
        "voice_id": True,
        "model": True,
        "language": True,
        "top_p": False,
        "repetition_penalty": False,
    },
    "maya": {"voice_id": True, "voice": True, "model": True, "language": False},
    "gemini": {"voice": True, "voice_id": True, "model": True, "language": True, "style": False},
    "kalpa": {
        "voice": False,
        "voice_id": False,
        "model": False,
        "temperature": False,
        "acoustic_temperature": False,
        "max_new_tokens": False,
        "audio_quality": False,
        "chunk_length_schedule": False,
    },
}


# Every required field on every config is a plain string, so one filler serves them all.
def _minimal_config(provider):
    return {field: "x" for field, required in EXPECTED_FIELDS[provider].items() if required}


def test_every_provider_resolves_to_a_config_model():
    assert set(SYNTHESIZER_CONFIG_MODELS) == set(SynthesizerProvider.all_values())


@pytest.mark.parametrize("provider", sorted(EXPECTED_FIELDS))
def test_config_shapes_are_stable(provider):
    fields = SYNTHESIZER_CONFIG_MODELS[provider].model_fields
    assert {name: f.is_required() for name, f in fields.items()} == EXPECTED_FIELDS[provider]


@pytest.mark.parametrize(
    "config_model", [CartesiaConfig, RimeConfig, SmallestConfig, SarvamConfig, PixaConfig, GeminiConfig]
)
def test_standard_shape_providers_extend_the_base(config_model):
    assert issubclass(config_model, StandardVoiceConfig)


@pytest.mark.parametrize("provider", sorted(EXPECTED_FIELDS))
def test_preprocess_builds_the_registered_config(provider):
    synth = Synthesizer(provider=provider, provider_config=_minimal_config(provider))
    assert type(synth.provider_config) is SYNTHESIZER_CONFIG_MODELS[provider]


def test_an_already_built_config_is_left_alone():
    config = ElevenLabsConfig(voice="George", voice_id="JBFqnCBsd6RMkjVDRZzb", model="eleven_turbo_v2_5")
    assert Synthesizer(provider="elevenlabs", provider_config=config).provider_config is config


def test_elevenlabs_still_requires_both_voice_and_voice_id():
    with pytest.raises(ValueError):
        Synthesizer(provider="elevenlabs", provider_config={"voice": "George", "model": "eleven_turbo_v2_5"})


def test_an_unknown_provider_is_rejected():
    with pytest.raises(ValueError):
        Synthesizer(provider="nope", provider_config={"voice": "x"})
