from typing import Any, NamedTuple, TypedDict


class LLMStreamChunk(NamedTuple):
    """Single chunk yielded from LLM generate_stream methods."""
    data: Any  # str for text, dict for function call payload
    end_of_stream: bool
    latency: dict | None = None
    is_function_call: bool = False
    function_name: str | None = None
    function_message: str | None = None


class LatencyData(TypedDict, total=False):
    sequence_id: int | None
    first_token_latency_ms: float | None
    total_stream_duration_ms: float | None
    service_tier: str | None
    llm_host: str | None


class FunctionCallPayload(TypedDict, total=False):
    url: str | None
    method: str | None
    param: Any
    api_token: str | None
    headers: dict | None
    model_args: dict
    meta_info: dict
    called_fun: str
    model_response: list[dict]
    tool_call_id: str
    textual_response: str | None
    resp: Any
