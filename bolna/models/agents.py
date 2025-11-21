from typing import Optional, List, Union, Dict, Callable
from pydantic import BaseModel, Field, field_validator
from .rag import VectorStore, GraphNodeRAGConfig


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
