from typing import Any, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from bolna.enums import ToolScope


class APIParams(BaseModel):
    url: Optional[str] = None
    method: Optional[str] = "POST"
    api_token: Optional[str] = None
    param: Optional[Union[str, dict]] = None
    headers: Optional[Union[str, dict]] = None
    pre_call_message: Optional[Union[str, dict]] = None
    pre_call_webhook_url: Optional[str] = None
    pre_call_webhook_param: Optional[Union[str, dict]] = None
    # Graph-agent tool scope; None == GLOBAL. NODE limits visibility to ``nodes``.
    scope: Optional[ToolScope] = None
    nodes: Optional[List[str]] = None


class LatencyData(BaseModel):
    sequence_id: Optional[int] = None
    first_token_latency_ms: Optional[float] = None
    total_stream_duration_ms: Optional[float] = None
    connection_latency_ms: Optional[float] = None
    service_tier: Optional[str] = None
    llm_host: Optional[str] = None
    reasoning_effort: Optional[str] = None
    verbosity: Optional[str] = None


class FunctionCallPayload(BaseModel):
    """A tool invocation with separate execution settings and model arguments.

    Put model-generated values in ``arguments``, including names such as ``url``.
    Top-level fields come from tool configuration or call bookkeeping. Consumers
    constructing this payload directly must use the mapping instead of extra fields.
    """

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

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
    # Model-generated data must never share the execution configuration namespace.
    arguments: dict = Field(default_factory=dict)


class LLMStreamChunk(BaseModel):
    """Single chunk yielded from LLM generate_stream methods."""

    data: Any = None
    end_of_stream: bool = False
    latency: Optional[LatencyData] = None
    is_function_call: bool = False
    function_name: Optional[str] = None
    function_message: Optional[Union[str, dict]] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None
    overflowed: bool = False
    reasoning_content: Optional[str] = None
