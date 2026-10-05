"""Per-model TTS control data: which models take a control, and per-model ranges the validators enforce."""

import pytest
from pydantic import ValidationError

from bolna.models import (
    SYNTHESIZER_CONFIG_MODELS,
    AzureConfig,
    SarvamConfig,
    tts_control_applies,
    tts_control_limits,
    tts_control_range,
    tts_provider_config_error,
    tts_render_settings,
)
from bolna.constants import TTS_AUDIO_SETTINGS
from bolna.helpers.utils import get_md5_hash, static_node_audio_key


def _sarvam(model, **controls):
    return SarvamConfig(voice="Anushka", voice_id="anushka", model=model, language="hi", **controls)


@pytest.mark.parametrize("speed", [0.5, 2.0])
def test_sarvam_v3_accepts_speed_inside_its_range(speed):
    assert _sarvam("bulbul:v3", speed=speed).speed == speed


@pytest.mark.parametrize("speed", [0.49, 2.01])
def test_sarvam_v3_rejects_speed_outside_its_range(speed):
    with pytest.raises(ValidationError, match="Sarvam bulbul:v3 speed must be between 0.5 and 2.0"):
        _sarvam("bulbul:v3", speed=speed)


def test_sarvam_v2_keeps_the_provider_wide_speed_range():
    assert _sarvam("bulbul:v2", speed=3.0).speed == 3.0


@pytest.mark.parametrize("speed", [0.5, 2.0])
def test_azure_accepts_speed_inside_its_range(speed):
    assert AzureConfig(voice="Aria", model="neural", language="en-US", speed=speed).speed == speed


@pytest.mark.parametrize("speed", [0.49, 2.01])
def test_azure_rejects_speed_outside_its_range(speed):
    with pytest.raises(ValidationError):
        AzureConfig(voice="Aria", model="neural", language="en-US", speed=speed)


@pytest.mark.parametrize(
    "provider,key,model,expected",
    [
        ("cartesia", "volume", "sonic-3", True),
        ("cartesia", "volume", "sonic-3.5", True),
        ("cartesia", "volume", "sonic-english", False),
        ("cartesia", "speed", "sonic-english", True),
        ("sarvam", "loudness", "bulbul:v2", True),
        ("sarvam", "loudness", "bulbul:v3", False),
        ("deepgram", "speed", "aura-2", True),
        ("deepgram", "speed", "aura-2-thalia-en", True),
        ("deepgram", "speed", "aura", False),
        ("rime", "time_scale_factor", "Coda", True),
        ("rime", "time_scale_factor", "mistv2", False),
        ("elevenlabs", "style", "eleven_turbo_v2_5", True),
        ("elevenlabs", "style", "eleven_v3", False),
        ("elevenlabs", "similarity_boost", "eleven_v3", True),
        ("openai", "speed", None, True),
    ],
)
def test_tts_control_applies(provider, key, model, expected):
    assert tts_control_applies(provider, key, model) is expected


def test_tts_control_limits():
    assert tts_control_limits("sarvam", "speed", "bulbul:v3") == (0.5, 2.0)
    assert tts_control_limits("sarvam", "speed", "bulbul:v2") is None
    assert tts_control_limits("cartesia", "speed", "sonic-3") is None


@pytest.mark.parametrize(
    "provider,key,model,expected",
    [
        ("sarvam", "speed", "bulbul:v3", (0.5, 2.0)),
        ("sarvam", "speed", "bulbul:v2", (0.3, 3.0)),
        ("cartesia", "volume", "sonic-3", (0.5, 2.0)),
        ("azuretts", "speed", None, (0.5, 2.0)),
        ("polly", "speed", None, None),
        ("kalpa", "temperature", None, None),
        ("not-a-provider", "speed", None, None),
    ],
)
def test_tts_control_range(provider, key, model, expected):
    assert tts_control_range(provider, key, model) == expected


@pytest.mark.parametrize(
    "provider,provider_config,expected",
    [
        ("elevenlabs", {"model": "eleven_turbo_v2_5", "speed": 1.2}, None),
        ("elevenlabs", {"model": "eleven_turbo_v2_5", "speed": 1.3}, "Elevenlabs speed must be between 0.7 and 1.2"),
        ("elevenlabs", {"similarity_boost": 1.5}, "Elevenlabs similarity_boost must be between 0.0 and 1.0"),
        ("azuretts", {"model": "neural", "speed": 2.5}, "Azuretts speed must be between 0.5 and 2.0"),
        ("sarvam", {"model": "bulbul:v3", "speed": 2.5}, "Sarvam bulbul:v3 speed must be between 0.5 and 2.0"),
        ("sarvam", {"model": "bulbul:v2", "speed": 2.5}, None),
        ("sarvam", {"model": "bulbul:v2", "speed": 3.5}, "Sarvam speed must be between 0.3 and 3.0"),
        ("cartesia", {"model": "sonic-3", "volume": 2.5}, "Cartesia volume must be between 0.5 and 2.0"),
        ("rime", {"model": "coda", "time_scale_factor": 3}, "Rime time_scale_factor must be between 0.4 and 2.5"),
        ("elevenlabs", {"speed": "fast"}, "Elevenlabs speed must be a number"),
        ("elevenlabs", {"speed": None}, None),
        ("polly", {"engine": "neural", "voice": "Joanna"}, None),
        ("not-a-provider", {"speed": 99}, None),
        ("cartesia", None, None),
    ],
)
def test_tts_provider_config_error(provider, provider_config, expected):
    assert tts_provider_config_error(provider, provider_config) == expected


# Every config model accepts these; each ignores the fields it doesn't declare.
BASE_CONFIG = {"voice": "v", "voice_id": "v", "language": "en", "model": "any-model"}

BOUNDED_FIELDS = [
    pytest.param(provider, key, *bounds, id=f"{provider}.{key}")
    for provider, config_model in SYNTHESIZER_CONFIG_MODELS.items()
    for key in config_model.model_fields
    for bounds in [tts_control_range(provider, key, "any-model")]
    if bounds is not None and None not in bounds
]


def test_every_bounded_field_is_covered():
    assert len(BOUNDED_FIELDS) >= 14


@pytest.mark.parametrize("provider,key,low,high", BOUNDED_FIELDS)
def test_config_error_rejects_exactly_what_the_config_model_rejects(provider, key, low, high):
    config_model = SYNTHESIZER_CONFIG_MODELS[provider]
    for value in (low, high):
        config_model.model_validate({**BASE_CONFIG, key: value})
        assert tts_provider_config_error(provider, {**BASE_CONFIG, key: value}) is None
    for value in (low - 0.01, high + 0.01):
        with pytest.raises(ValidationError):
            config_model.model_validate({**BASE_CONFIG, key: value})
        assert tts_provider_config_error(provider, {**BASE_CONFIG, key: value}) is not None


@pytest.mark.parametrize(
    "provider,provider_config,exclude,expected",
    [
        ("cartesia", {"model": "sonic-3", "speed": 1.0, "volume": 1.0}, (), ""),
        ("cartesia", {"model": "sonic-3", "speed": 1, "volume": 1.5}, (), "volume=1.5"),
        ("cartesia", {"speed": 1.2, "volume": 1.5}, (), "speed=1.2,volume=1.5"),
        ("cartesia", {"speed": 1.2, "volume": 1.5}, ("speed",), "volume=1.5"),
        ("elevenlabs", {"similarity_boost": 0.75, "temperature": 0.5, "style": 0}, (), ""),
        ("elevenlabs", {"similarity_boost": 0.8}, (), "similarity_boost=0.8"),
        ("sarvam", {"model": "bulbul:v2", "loudness": 1.5}, (), "loudness=1.5"),
        ("rime", {"model": "coda", "time_scale_factor": 0.8}, (), "time_scale_factor=0.8"),
        ("soniox", {"speed": 1.0}, (), "speed=1"),
        ("soniox", {"reduce_silence": True}, (), "reduce_silence=true"),
        ("kalpa", {"temperature": 0.7, "acoustic_temperature": 0.4}, (), "acoustic_temperature=0.4,temperature=0.7"),
        ("kalpa", {"audio_quality": "high", "chunk_length_schedule": [50, 120]}, (), "audio_quality=high"),
        ("pixa", {"top_p": 0.95, "repetition_penalty": 1.3}, (), ""),
        ("pixa", {"top_p": 0.8, "repetition_penalty": 1.3}, (), "top_p=0.8"),
        ("gemini", {"style": "cheerful"}, (), "style=cheerful"),
        ("gemini", {"style": ""}, (), ""),
        ("deepgram", {"model": "aura-2", "mip_opt_out": True}, (), ""),
        ("cartesia", {"voice": "Sonic", "sampling_rate": 8000, "speed": None}, (), ""),
        ("polly", {"engine": "neural"}, (), ""),
        ("not-a-provider", {"speed": 2}, (), ""),
        ("cartesia", None, (), ""),
    ],
)
def test_tts_render_settings(provider, provider_config, exclude, expected):
    assert tts_render_settings(provider, provider_config, exclude=exclude) == expected


@pytest.mark.parametrize("provider", sorted(TTS_AUDIO_SETTINGS))
def test_audio_settings_name_real_config_fields(provider):
    assert TTS_AUDIO_SETTINGS[provider] <= set(SYNTHESIZER_CONFIG_MODELS[provider].model_fields)


def test_static_node_audio_key_is_unchanged_without_render_settings():
    # Clips pre-rendered before render settings joined the key must still be found.
    assert static_node_audio_key("Hi", "cartesia", "Sonic", "vid", "sonic-3") == get_md5_hash(
        "cartesia|Sonic|vid|sonic-3|Hi"
    )


def test_static_node_audio_key_changes_with_render_settings():
    plain = static_node_audio_key("Hi", "cartesia", "Sonic", "vid", "sonic-3")
    louder = static_node_audio_key("Hi", "cartesia", "Sonic", "vid", "sonic-3", render_settings="volume=1.5")
    assert louder != plain
