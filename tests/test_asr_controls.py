import pytest

from bolna.constants import ASR_CONTROLS, asr_controls_for, supports_endpointing
from bolna.models import Transcriber


def _transcriber_with(control, value):
    config = {"provider": control.providers[0] if control.providers else "deepgram"}
    head, _, tail = control.key.partition(".")
    config[head] = {tail: value} if tail else value
    return Transcriber(**config)


def _read(transcriber, key):
    value = transcriber
    for part in key.split("."):
        value = getattr(value, part)
    return value


@pytest.mark.parametrize("control", ASR_CONTROLS, ids=lambda c: f"{c.key}:{c.default}")
def test_every_value_offered_is_one_bolna_accepts(control):
    values = control.options or (True, False)
    assert control.default in values
    for value in values:
        assert _read(_transcriber_with(control, value), control.key) == value


@pytest.mark.parametrize(
    "provider,model,expected",
    [
        ("openai", "gpt-realtime-whisper", ["delay", "noise_reduction"]),
        (
            "assembly",
            "universal-3-6-pro",
            ["assemblyai_config.mode", "assemblyai_config.vad_threshold", "assemblyai_config.max_turn_silence"],
        ),
        ("assembly", "universal-streaming", ["assemblyai_config.vad_threshold", "assemblyai_config.max_turn_silence"]),
        ("deepgram", "flux-general-en", ["eot_threshold", "eot_timeout_ms", "eager_eot_threshold"]),
        ("deepgram-test", "flux-general-multi", ["eot_threshold", "eot_timeout_ms", "eager_eot_threshold"]),
        ("deepgram", "nova-3", []),
        ("sarvam", "saaras:v4", []),
        (None, None, []),
    ],
)
def test_asr_controls_for(provider, model, expected):
    assert [c.key for c in asr_controls_for(provider, model)] == expected


@pytest.mark.parametrize("model,default", [("universal-3-6-pro", 0.2), ("universal-streaming", 0.4)])
def test_assemblyai_vad_default_follows_the_model(model, default):
    [vad] = [c for c in asr_controls_for("assembly", model) if c.key.endswith("vad_threshold")]
    assert vad.default == default


@pytest.mark.parametrize(
    "provider,model,expected",
    [
        ("deepgram", "nova-3", True),
        ("Deepgram", "nova-2", True),
        ("deepgram", "flux-general-en", False),
        ("elevenlabs", "scribe_v2_realtime", True),
        ("openai", "gpt-realtime-whisper", True),
        ("sarvam", "saaras:v4", False),
        ("assembly", "universal-3-6-pro", False),
        (None, None, False),
    ],
)
def test_supports_endpointing(provider, model, expected):
    assert supports_endpointing(provider, model) is expected
