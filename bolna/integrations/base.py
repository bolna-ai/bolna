from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PostCallContext:
    agent_name: str
    run_id: str
    call_sid: Optional[str] = None
    duration_seconds: Optional[float] = None
    hangup_reason: Optional[str] = None
    summary: Optional[str] = None
    extracted_data: dict = field(default_factory=dict)
    recording_url: Optional[str] = None


class PostCallIntegration(ABC):
    @classmethod
    @abstractmethod
    def from_config(cls, config) -> "PostCallIntegration":
        ...

    # may raise; the runner swallows exceptions and logs
    @abstractmethod
    async def execute(self, ctx: PostCallContext) -> None:
        ...
