import json
from typing import Optional, List, Union, Dict
from pydantic import BaseModel, Field, field_validator, ValidationError, Json
from pydantic_core import PydanticCustomError
from .providers import *

AGENT_WELCOME_MESSAGE = "This call is being recorded for quality assurance and training. Please speak now."


def validate_attribute(value, allowed_values):
    if value not in allowed_values:
        raise ValidationError(f"Invalid provider {value}. Supported values: {allowed_values}")
    return value


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
    similarity_boost: Optional[float] = 0.5


class OpenAIConfig(BaseModel):
    voice: str
    model: str



class DeepgramConfig(BaseModel):
    voice: str
    model: str


class AzureConfig(BaseModel):
    voice: str
    model: str
    language: str


class Transcriber(BaseModel):
    model: Optional[str] = "nova-2"
    language: Optional[str] = None
    stream: bool = False
    sampling_rate: Optional[int] = 16000
    encoding: Optional[str] = "linear16"
    endpointing: Optional[int] = 400
    keywords: Optional[str] = None
    task:Optional[str] = "transcribe"
    provider: Optional[str] = "deepgram"

    @field_validator("provider")
    def validate_model(cls, value):
        print(f"value {value}, PROVIDERS {list(SUPPORTED_TRANSCRIBER_PROVIDERS.keys())}")
        return validate_attribute(value, list(SUPPORTED_TRANSCRIBER_PROVIDERS.keys()))


class Synthesizer(BaseModel):
    provider: str
    provider_config: Union[PollyConfig, ElevenLabsConfig, AzureConfig, DeepgramConfig, OpenAIConfig] = Field(union_mode='smart')
    stream: bool = False
    buffer_size: Optional[int] = 40  # 40 characters in a buffer
    audio_format: Optional[str] = "pcm"
    caching: Optional[bool] = True

    @field_validator("provider")
    def validate_model(cls, value):
        return validate_attribute(value, ["polly", "elevenlabs", "openai", "deepgram", "azuretts"])


class IOModel(BaseModel):
    provider: str
    format: Optional[str] = "wav"

    @field_validator("provider")
    def validate_provider(cls, value):
        return validate_attribute(value, ["twilio", "default", "database", "exotel", "plivo", "daily"])


# Can be used to route across multiple prompts as well
class Route(BaseModel):
    route_name: str
    utterances: List[str]
    response: Union[List[
        str], str]  # If length of responses is less than utterances, a random sentence will be used as a response and if it's equal, respective index will be used to use it as FAQs caching
    score_threshold: Optional[float] = 0.85  # this is the required threshold for cosine similarity


# Routes can be used for FAQs caching, prompt routing, guard rails, agent assist function calling
class Routes(BaseModel):
    embedding_model: Optional[str] = "Snowflake/snowflake-arctic-embed-l"
    routes: Optional[List[Route]] = []


class OpenaiAssistant(BaseModel):
    name: Optional[str] = None
    assistant_id: str = None
    max_tokens: Optional[int] =100
    temperature: Optional[float] = 0.2
    buffer_size: Optional[int] = 100
    provider: Optional[str] = "openai"
    model: Optional[str] = "gpt-3.5-turbo"


class MongoDBProviderConfig(BaseModel):
    connection_string: Optional[str] = None
    db_name: Optional[str] = None
    collection_name: Optional[str] = None
    index_name: Optional[str] = None
    llm_model: Optional[str] = "gpt-3.5-turbo"
    embedding_model: Optional[str] = "text-embedding-3-small"
    embedding_dimensions: Optional[int] = 256


class LanceDBProviderConfig(BaseModel):
    vector_id: str


class VectorStore(BaseModel):
    provider: str
    provider_config: Union[LanceDBProviderConfig, MongoDBProviderConfig]


class Llm(BaseModel):
    model: Optional[str] = "gpt-3.5-turbo"
    max_tokens: Optional[int] = 100
    family: Optional[str] = "openai"
    temperature: Optional[float] = 0.1
    request_json: Optional[bool] = False
    stop: Optional[List[str]] = None
    top_k: Optional[int] = 0
    top_p: Optional[float] = 0.9
    min_p: Optional[float] = 0.1
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0
    provider: Optional[str] = "openai"
    base_url: Optional[str] = None
    routes: Optional[Routes] = None


class SimpleLlmAgent(Llm):
    agent_flow_type: Optional[str] = "streaming" #It is used for backwards compatibility
    routes: Optional[Routes] = None
    extraction_details: Optional[str] = None
    summarization_details: Optional[str] = None


class Node(BaseModel):
    id: str
    type: str  # Can be router or conversation for now
    llm: Llm
    exit_criteria: str
    exit_response: Optional[str] = None
    exit_prompt: Optional[str] = None
    is_root: Optional[bool] = False


class Edge(BaseModel):
    start_node: str  # Node ID
    end_node: str
    condition: Optional[tuple] = None  # extracted value from previous step and it's value


class LlmAgentGraph(BaseModel):
    nodes: List[Node]
    edges: List[Edge]


class AgentRouteConfig(BaseModel):
    utterances: List[str]
    threshold: Optional[float] = 0.85


class MultiAgent(BaseModel):
    agent_map: Dict[str, Union[Llm, OpenaiAssistant]]
    agent_routing_config: Dict[str, AgentRouteConfig]
    default_agent: str
    embedding_model: Optional[str] = "Snowflake/snowflake-arctic-embed-l"


class KnowledgebaseAgent(Llm):
    vector_store: VectorStore
    provider: Optional[str] = "openai"
    model: Optional[str] = "gpt-3.5-turbo"


class LlmAgent(BaseModel):
    agent_flow_type: str
    agent_type: str
    routes: Optional[Routes] = None
    llm_config: Union[OpenaiAssistant, KnowledgebaseAgent, LlmAgentGraph, MultiAgent, SimpleLlmAgent]

    @field_validator('llm_config', mode='before')
    def validate_llm_config(cls, value, info):
        agent_type = info.data.get('agent_type')
        print(f"Agent type: {agent_type}")
        print(f"Value type: {type(value)}")
        print(f"Value: {value}")

        valid_config_types = {
            'openai_assistant': OpenaiAssistant,
            'knowledgebase_agent': KnowledgebaseAgent,
            'llm_agent_graph': LlmAgentGraph,
            'multiagent': MultiAgent,
            'simple_llm_agent': SimpleLlmAgent,
        }

        if agent_type not in valid_config_types:
            raise ValueError(f'Unsupported agent_type: {agent_type}')

        expected_type = valid_config_types[agent_type]

        if not isinstance(value, dict):
            raise ValueError(f"llm_config must be a dict, got {type(value)}")

        try:
            return expected_type(**value)
        except Exception as e:
            raise ValueError(f"Failed to create {expected_type.__name__} from llm_config: {str(e)}")


class ToolDescription(BaseModel):
    name: str
    description: str
    parameters: Dict


class APIParams(BaseModel):
    url: Optional[str] = None
    method: Optional[str] = "POST"
    api_token: Optional[str] = None
    param: Optional[str] = None #Payload for the URL


class ToolModel(BaseModel):
    tools:  Optional[Union[str, List[ToolDescription]]] = None
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
    hangup_after_silence: Optional[int] = 10
    incremental_delay: Optional[int] = 100  # use this to incrementally delay to handle long pauses
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
    trigger_user_online_message_after: Optional[int] = 6
    check_user_online_message: Optional[str] = "Hey, are you still there"
    check_if_user_online: Optional[bool] = True

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
