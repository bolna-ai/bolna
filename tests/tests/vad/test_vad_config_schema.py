from __future__ import annotations

import pytest
from pydantic import ValidationError

from bolna.models import IOModel, VadConfig


def test_defaults() -> None:
    cfg = VadConfig()
    assert cfg.sample_rate == 8000
    assert cfg.threshold == 0.5
    assert cfg.min_silence_ms == 100
    assert cfg.speech_pad_ms == 30
    assert cfg.pre_speech_ms == 500


def test_io_model_accepts_nested_vad_config() -> None:
    io = IOModel.model_validate(
        {
            "provider": "twilio",
            "format": "mulaw",
            "vad_config": {
                "sample_rate": 8000,
                "threshold": 0.4,
                "pre_speech_ms": 750,
            },
        }
    )
    assert io.vad_config is not None
    assert io.vad_config.threshold == 0.4
    assert io.vad_config.pre_speech_ms == 750
    # Fields not supplied fall back to defaults
    assert io.vad_config.min_silence_ms == 100


def test_io_model_without_vad_config_is_unchanged() -> None:
    io = IOModel.model_validate({"provider": "twilio"})
    assert io.vad_config is None


def test_threshold_bounds() -> None:
    with pytest.raises(ValidationError):
        VadConfig(threshold=1.5)
    with pytest.raises(ValidationError):
        VadConfig(threshold=-0.1)


def test_sample_rate_must_be_supported() -> None:
    with pytest.raises(ValidationError):
        VadConfig(sample_rate=22050)
    assert VadConfig(sample_rate=16000).sample_rate == 16000


def test_ms_fields_reject_negatives() -> None:
    with pytest.raises(ValidationError):
        VadConfig(pre_speech_ms=-1)
    with pytest.raises(ValidationError):
        VadConfig(min_silence_ms=-1)
    with pytest.raises(ValidationError):
        VadConfig(speech_pad_ms=-1)


def test_ms_fields_reject_unreasonably_large_values() -> None:
    with pytest.raises(ValidationError):
        VadConfig(pre_speech_ms=5000)
    with pytest.raises(ValidationError):
        VadConfig(speech_pad_ms=1000)
