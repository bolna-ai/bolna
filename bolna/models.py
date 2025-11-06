import json
from typing import Optional, List, Union, Dict, Callable
from pydantic import BaseModel, Field, field_validator, ValidationError, Json, model_validator
from pydantic_core import PydanticCustomError
from .providers import *

AGENT_WELCOME_MESSAGE = "This call is being recorded for quality assurance and training. Please speak now."


def validate_attribute(value, allowed_values, value_type='provider'):
    if value not in allowed_values:
        raise ValueError(f"Invalid value for {value_type}:'{value}' provided. Supported values: {allowed_values}.")
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


class MongoDBProviderConfig(BaseModel):
    connection_string: Optional[str] = None
    db_name: Optional[str] = None
    collection_name: Optional[str] = None
    index_name: Optional[str] = None
    llm_model: Optional[str] = "gpt-3.5-turbo"
    embedding_model: Optional[str] = "text-embedding-3-small"
    embedding_dimensions: Optional[int] = 256


class RerankerConfig(BaseModel):
    """Configuration for document reranking in RAG systems."""
    enabled: bool = False
    model_type: str = "minilm-l6-v2"  # bge-base, bge-large, bge-multilingual, minilm-l6-v2
    candidate_count: int = 20     # How many candidates to retrieve before reranking
    final_count: int = 5          # Final number of results to return after reranking
    
    @field_validator("model_type")
    def validate_reranker_model(cls, value):
        allowed_models = ["bge-base", "bge-large", "bge-multilingual", "minilm-l6-v2"]
        if value not in allowed_models:
            raise ValueError(f"Invalid reranker model: '{value}'. Supported models: {allowed_models}")
        return value
    
    @field_validator("candidate_count")
    def validate_candidate_count(cls, value):
        if value < 1 or value > 100:
            raise ValueError("candidate_count must be between 1 and 100")
        return value
    
    @field_validator("final_count") 
    def validate_final_count(cls, value):
        if value < 1 or value > 50:
            raise ValueError("final_count must be between 1 and 50")
        return value

class LanceDBProviderConfig(BaseModel):
    vector_id: str
    similarity_top_k: Optional[int] = 5
    score_threshold: Optional[float] = 0.1
    reranker: Optional[RerankerConfig] = RerankerConfig()  # Default to disabled reranker


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

class GraphEdge(BaseModel):
    to_node_id: str
    condition: str

class GraphNodeRAGConfig(BaseModel):
    """RAG configuration for Graph Agent nodes."""
    vector_store: VectorStore
    temperature: Optional[float] = 0.7
    model: Optional[str] = "gpt-4" 
    max_tokens: Optional[int] = 150

class GraphNode(BaseModel):
    id: str
    description: Optional[str] = None
    prompt: str
    edges: List[GraphEdge] = Field(default_factory=list)
    completion_check: Optional[Callable[[List[dict]], bool]] = None
    rag_config: Optional[GraphNodeRAGConfig] = None


class GraphAgentConfig(Llm):
    agent_information: str
    nodes: List[GraphNode]
    current_node_id: str
    context_data: Optional[dict] = None

class KnowledgeAgentConfig(Llm):
    agent_information: Optional[str] = "Knowledge-based AI assistant"
    prompt: Optional[str] = None
    rag_config: Optional[Dict] = None
    llm_provider: Optional[str] = "openai"
    context_data: Optional[dict] = None

class AgentRouteConfig(BaseModel):
    utterances: List[str]
    threshold: Optional[float] = 0.85


class MultiAgent(BaseModel):
    agent_map: Dict[str, Union[Llm]]
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
    llm_config: Union[KnowledgebaseAgent, LlmAgentGraph, MultiAgent, SimpleLlmAgent, GraphAgentConfig, KnowledgeAgentConfig]

    @field_validator('llm_config', mode='before')
    def validate_llm_config(cls, value, info):
        agent_type = info.data.get('agent_type')

        valid_config_types = {
            'knowledgebase_agent': KnowledgeAgentConfig,
            'graph_agent': GraphAgentConfig,
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
