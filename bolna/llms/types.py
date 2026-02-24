from typing import Any, Optional

from pydantic import BaseModel, ConfigDict


class LatencyData(BaseModel):
    sequence_id: Optional[int] = None
    first_token_latency_ms: Optional[float] = None
    total_stream_duration_ms: Optional[float] = None
    service_tier: Optional[str] = None
    llm_host: Optional[str] = None


class FunctionCallPayload(BaseModel):
    model_config = ConfigDict(extra='allow', protected_namespaces=())

    url: Optional[str] = None
    method: Optional[str] = None
    param: Any = None
    api_token: Optional[str] = None
    headers: Optional[dict] = None
    model_args: dict = {}
    meta_info: dict = {}
    called_fun: str = ""
    model_response: list[dict] = []
    tool_call_id: str = ""
    textual_response: Optional[str] = None
    resp: Any = None


class LLMStreamChunk(BaseModel):
    """Single chunk yielded from LLM generate_stream methods."""
    data: Any = None
    end_of_stream: bool = False
    latency: Optional[LatencyData] = None
    is_function_call: bool = False
    function_name: Optional[str] = None
    function_message: Optional[str] = None
