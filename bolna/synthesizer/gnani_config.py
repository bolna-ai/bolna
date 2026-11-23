"""Gnani Timbre model and voice settings shared by agent validation and the synthesizer."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationInfo, field_validator, model_validator


# Published stock voice catalogs: https://docs.gnani.ai/api/TTS/available-voices
TIMBRE_VOICES = {
    "timbre-v2.5": (
        "Ambuja",
        "Asmita",
        "Bhavna",
        "Brinda",
        "Chitra",
        "Deepak",
        "Devika",
        "Dhruva",
        "Falak",
        "Girish",
        "Hemraj",
        "Ishaan",
        "Jalaj",
        "Jwala",
        "Kaveri",
        "Kavin",
        "Kirra",
        "Lavanya",
        "Lehara",
        "Mehuli",
        "Nalini",
        "Noopur",
        "Omkar",
        "Poorvi",
        "Pranav",
        "Reshma",
        "Riyaan",
        "Roopesh",
        "Saanvi",
        "Shlok",
        "Suhana",
        "Trisha",
        "Trupti",
        "Urmila",
        "Varuni",
        "Vedika",
        "Veera",
        "Vikrant",
        "Yashvi",
        "Yukti",
        "Zahira",
        "Zayan",
    ),
    "timbre-v3.0": (
        "Anitasingh",
        "Arko",
        "Avinash",
        "Chetana",
        "Devansh",
        "Harshad",
        "Kalpana",
        "Lakshmi",
        "Lakshmidevi",
        "Mahendra",
        "Manjunath",
        "Neelam",
        "Nitin",
        "Prakash",
        "Priyasharm",
        "Rahulmehta",
        "Rajeev",
        "Rajni",
        "Rakesh",
        "Rekha",
        "Riyaa",
        "Santosh",
        "Shivani",
        "Shruthi",
        "Shubham",
        "Sneha",
        "Sohini",
        "Sunaina",
        "Swathi",
        "Tarika",
        "Vasudha",
        "Vikramdasu",
        "Vinod",
        "Yashraj",
    ),
}


def timbre_speed(model, value):
    if model == "timbre-v2.5" and isinstance(value, str):
        value = {"slow": 0.85, "medium": 1.0, "fast": 1.15}.get(value, value)
    try:
        speed = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid speed for {model}") from None
    if model == "timbre-v2.5" and not 0.85 <= speed <= 1.15:
        raise ValueError("Timbre v2.5 speed must be between 0.85 and 1.15")
    if model == "timbre-v3.0" and speed not in (0.8, 1.0, 1.2):
        raise ValueError("Timbre v3.0 speed must be 0.8, 1.0 or 1.2")
    return speed


class GnaniConfig(BaseModel):
    model_config = ConfigDict(json_schema_extra={"x-model-voices": TIMBRE_VOICES})

    model: Literal["timbre-v2.5", "timbre-v3.0"] = "timbre-v2.5"
    voice: str
    language: Literal[
        "auto", "hi-IN", "en-IN", "ta-IN", "te-IN", "kn-IN", "ml-IN", "mr-IN", "pa-IN", "bn-IN", "gu-IN", "hi-en"
    ] = "auto"
    speed: float = 1.0
    stream_transport: Literal["websocket", "sse"] = "websocket"

    @field_validator("speed", mode="before")
    @classmethod
    def validate_speed(cls, value, info: ValidationInfo):
        return timbre_speed(info.data.get("model"), value)

    @model_validator(mode="after")
    def validate_voice(self):
        if self.voice not in TIMBRE_VOICES[self.model]:
            raise ValueError(f"Unknown {self.model} voice: {self.voice}")
        return self
