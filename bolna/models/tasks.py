from typing import Optional, List, Union, Dict
from pydantic import BaseModel, Field, field_validator, model_validator
from ..providers import *
from .providers import PollyConfig, ElevenLabsConfig, AzureConfig, RimeConfig, SmallestConfig, SarvamConfig, CartesiaConfig, DeepgramConfig, OpenAIConfig
from .agents import LlmAgent, SimpleLlmAgent

AGENT_WELCOME_MESSAGE = "This call is being recorded for quality assurance and training. Please speak now."


def validate_attribute(value, allowed_values, value_type='provider'):
    if value not in allowed_values:
        raise ValueError(f"Invalid value for {value_type}:'{value}' provided. Supported values: {allowed_values}.")
    return value


class Transcriber(BaseModel):
    model: Optional[str] = "nova-2"
    language: Optional[str] = None
    stream: bool = False
    sampling_rate: Optional[int] = 16000
    encoding: Optional[str] = "linear16"
    endpointing: Optional[int] = 500
    keywords: Optional[str] = None
    task:Optional[str] = "transcribe"
    provider: Optional[str] = "deepgram"

    @field_validator("provider")
    def validate_model(cls, value):
        return validate_attribute(value, list(SUPPORTED_TRANSCRIBER_PROVIDERS.keys()))


class Synthesizer(BaseModel):
    provider: str
    provider_config: Union[PollyConfig, ElevenLabsConfig, AzureConfig, RimeConfig, SmallestConfig, SarvamConfig, CartesiaConfig, DeepgramConfig, OpenAIConfig] = Field(union_mode='smart')
    stream: bool = False
    buffer_size: Optional[int] = 40  # 40 characters in a buffer
    audio_format: Optional[str] = "pcm"
    caching: Optional[bool] = True

    @model_validator(mode="before")
    def preprocess(cls, values):
        provider = values.get("provider")
        config = values.get("provider_config", {})

        if provider == "elevenlabs":
            if not config.get("voice") or not config.get("voice_id"):
                raise ValueError("ElevenLabs config requires 'voice' or 'voice_id'.")

        return values

    @field_validator("provider")
    def validate_model(cls, value):
        return validate_attribute(value, ["polly", "elevenlabs", "azuretts", "openai", "deepgram", "cartesia", "smallest", "sarvam", "rime"])



class IOModel(BaseModel):
    provider: str
    format: Optional[str] = "wav"

    @field_validator("provider")
    def validate_provider(cls, value):
        return validate_attribute(value, ["twilio", "default", "database", "exotel", "plivo"])


class ToolFunction(BaseModel):
    name: str
    description: str
    parameters: Dict
    strict: bool = True


class ToolDescription(BaseModel):
    type: str = "function"
    function: ToolFunction


class ToolDescriptionLegacy(BaseModel):
    name: str
    description: str
    parameters: Dict


class APIParams(BaseModel):
    url: Optional[str] = None
    method: Optional[str] = "POST"
    api_token: Optional[str] = None
    param: Optional[Union[str, dict]] = None
    headers: Optional[Union[str, dict]] = None


class ToolModel(BaseModel):
    tools: Optional[Union[str, List[Union[ToolDescription, ToolDescriptionLegacy]]]] = None
    tools_params: Dict[str, APIParams]


class ToolsConfig(BaseModel):
    llm_agent: Optional[Union[LlmAgent, SimpleLlmAgent]] = None
    synthesizer: Optional[Synthesizer] = None
    transcriber: Optional[Transcriber] = None
    input: Optional[IOModel] = None
    output: Optional[IOModel] = None
    api_tools: Optional[ToolModel] = None


class ToolsChainModel(BaseModel):
    execution: str = Field(..., pattern="^(parallel|sequential)$")
    pipelines: List[List[str]]


class ConversationConfig(BaseModel):
    optimize_latency: Optional[bool] = True  # This will work on in conversation
    hangup_after_silence: Optional[int] = 20
    incremental_delay: Optional[int] = 900  # use this to incrementally delay to handle long pauses
    number_of_words_for_interruption: Optional[
        int] = 1  # Maybe send half second of empty noise if needed for a while as soon as we get speaking true in nitro, use that to delay
    interruption_backoff_period: Optional[int] = 100
    hangup_after_LLMCall: Optional[bool] = False
    call_cancellation_prompt: Optional[str] = None
    backchanneling: Optional[bool] = False
    backchanneling_message_gap: Optional[int] = 5
    backchanneling_start_delay: Optional[int] = 5
    ambient_noise: Optional[bool] = False
    ambient_noise_track: Optional[str] = "convention_hall"
    call_terminate: Optional[int] = 90
    use_fillers: Optional[bool] = False
    trigger_user_online_message_after: Optional[int] = 10
    check_user_online_message: Optional[str] = "Hey, are you still there"
    check_if_user_online: Optional[bool] = True
    generate_precise_transcript: Optional[bool] = False
    dtmf_enabled: Optional[bool] = False

    @field_validator('hangup_after_silence', mode='before')
    def set_hangup_after_silence(cls, v):
        return v if v is not None else 10  # Set default value if None is passed


class Task(BaseModel):
    tools_config: ToolsConfig
    toolchain: ToolsChainModel
    task_type: Optional[str] = "conversation"  # extraction, summarization, notification
    task_config: ConversationConfig = dict()


class AgentModel(BaseModel):
    agent_name: str
    agent_type: str = "other"
    tasks: List[Task]
    agent_welcome_message: Optional[str] = AGENT_WELCOME_MESSAGE
