from pydantic import BaseModel, Field
from typing import Optional


class ComponentLatencies(BaseModel):
    connection_latency_ms: Optional[float] = None
    turn_latencies: list = Field(default_factory=list)
    other_latencies: list = Field(default_factory=list)
