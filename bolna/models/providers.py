from typing import Optional
from pydantic import BaseModel


class PollyConfig(BaseModel):
    voice: str
    engine: str
    language: str
    # volume: Optional[str] = '0dB'
    # rate: Optional[str] = '100%'


class ElevenLabsConfig(BaseModel):
    voice: str
    voice_id: str
    model: str
    temperature: Optional[float] = 0.5
    similarity_boost: Optional[float] = 0.75
    speed: Optional[float] = 1.0


class OpenAIConfig(BaseModel):
    voice: str
    model: str


class DeepgramConfig(BaseModel):
    voice_id: str
    voice: str
    model: str


class CartesiaConfig(BaseModel):
    voice_id: str
    voice: str
    model: str
    language: str


class RimeConfig(BaseModel):
    voice_id: str
    language: str
    voice: str
    model: str


class SmallestConfig(BaseModel):
    voice_id: str
    language: str
    voice: str
    model: str


class SarvamConfig(BaseModel):
    voice_id: str
    language: str
    voice: str
    model: str
    speed: Optional[float] = 1.0


class AzureConfig(BaseModel):
    voice: str
    model: str
    language: str
    speed: Optional[float] = 1.0
