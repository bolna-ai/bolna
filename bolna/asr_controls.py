"""Configurable transcriber settings per provider and model, with the values each one takes and its default."""

from dataclasses import dataclass
from typing import Optional, Union

AsrControlValue = Union[str, int, float, bool, None]


@dataclass(frozen=True)
class AsrControl:
    # Dotted path into the transcriber config, e.g. "assemblyai_config.mode".
    key: str
    default: AsrControlValue
    # Empty options means an on/off switch.
    options: tuple = ()
    # Empty providers or model_prefixes match every provider or model.
    providers: tuple = ()
    model_prefixes: tuple = ()
    exclude_model_prefixes: tuple = ()

    def applies(self, provider: Optional[str], model: Optional[str]) -> bool:
        return _matches(provider, model, self.providers, self.model_prefixes, self.exclude_model_prefixes)


def _matches(provider, model, providers, model_prefixes=(), exclude_model_prefixes=()) -> bool:
    provider = (provider or "").lower()
    model = (model or "").lower()
    if providers and provider not in providers:
        return False
    if model_prefixes and not model.startswith(model_prefixes):
        return False
    return not (exclude_model_prefixes and model.startswith(exclude_model_prefixes))


def _steps(start: float, stop: float, step: float) -> tuple:
    count = round((stop - start) / step)
    return tuple(round(start + i * step, 2) for i in range(count + 1))


_ASSEMBLY = ("assembly",)
_UNIVERSAL_3 = ("universal-3",)
_FLUX = ("flux-",)

ASR_CONTROLS: tuple[AsrControl, ...] = (
    AsrControl("delay", "medium", ("minimal", "low", "medium", "high", "xhigh"), providers=("openai",)),
    AsrControl("noise_reduction", False, providers=("openai",)),
    # Defaults are what AssemblyAI applies when the setting is unset.
    AsrControl("assemblyai_config.mode", "balanced", ("balanced", "max_accuracy"), _ASSEMBLY, _UNIVERSAL_3),
    AsrControl("assemblyai_config.vad_threshold", 0.2, _steps(0.1, 0.9, 0.1), _ASSEMBLY, _UNIVERSAL_3),
    AsrControl("assemblyai_config.vad_threshold", 0.4, _steps(0.1, 0.9, 0.1), _ASSEMBLY, (), _UNIVERSAL_3),
    AsrControl("assemblyai_config.max_turn_silence", 1280, (800, 1000, 1280, 1500, 2000, 3000, 5000), _ASSEMBLY),
    AsrControl("eot_threshold", 0.7, _steps(0.5, 0.9, 0.05), model_prefixes=_FLUX),
    AsrControl("eot_timeout_ms", 1000, (300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000), model_prefixes=_FLUX),
    # None leaves eager end-of-turn off.
    AsrControl("eager_eot_threshold", None, (None, *_steps(0.3, 0.9, 0.05)), model_prefixes=_FLUX),
)

# Transcribers that forward `endpointing` to the vendor; the rest detect end of turn themselves.
ENDPOINTING_PROVIDERS = ("deepgram", "elevenlabs", "gladia", "openai", "gemini")
# Deepgram Flux ignores `endpointing` and uses the eot_* settings instead.
NO_ENDPOINTING_MODEL_PREFIXES = _FLUX

# Response Rate presets as (transcriber.endpointing, task_config.incremental_delay) in ms.
RESPONSE_RATE_PRESETS = {
    "Rapid": {"endpointing": 100, "incremental_delay": 200},
    "Normal": {"endpointing": 700, "incremental_delay": 1200},
    "Patient": {"endpointing": 1000, "incremental_delay": 1800},
}
# Allowed ranges for a custom endpointing and incremental_delay, in ms.
TURN_TIMING_RANGES = {
    "endpointing": {"min": 0, "max": 5000, "step": 50},
    "incremental_delay": {"min": 0, "max": 2500, "step": 50},
}


def asr_controls_for(provider: Optional[str], model: Optional[str]) -> list[AsrControl]:
    return [control for control in ASR_CONTROLS if control.applies(provider, model)]


def supports_endpointing(provider: Optional[str], model: Optional[str]) -> bool:
    return _matches(provider, model, ENDPOINTING_PROVIDERS, exclude_model_prefixes=NO_ENDPOINTING_MODEL_PREFIXES)
