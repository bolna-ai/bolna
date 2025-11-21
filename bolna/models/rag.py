from typing import Optional, Union
from pydantic import BaseModel, field_validator


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


class GraphNodeRAGConfig(BaseModel):
    """RAG configuration for Graph Agent nodes."""
    vector_store: VectorStore
    temperature: Optional[float] = 0.7
    model: Optional[str] = "gpt-4"
    max_tokens: Optional[int] = 150
