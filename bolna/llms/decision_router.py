"""Graph routing on a decision model: the hop's edges become one choice question, scored rather than decoded.

A decision model weighs every edge against the conversation at once instead of committing to the first
tokens of one function name, and writes no text. An edge's own parameters therefore come from the LLM
router the agent would otherwise use, which also routes any hop the decision API cannot answer.
"""

import json
import os
from dataclasses import dataclass
from typing import Callable, Optional

from bolna.enums import DecisionProvider
from bolna.helpers.logger_config import configure_logger
from bolna.llms.http_client_pool import get_shared_http_client
from bolna.llms.routing_stream import CONFIDENCE_KEY, TRAILING_KEYS

logger = configure_logger(__name__)

QUESTION = "next_node"
INSTRUCTIONS = (
    "Select the one edge that matches the caller's latest completed response. "
    "If no edge matches, stay on the current node."
)
_SPEAKER = {"assistant": "Agent", "user": "Caller", "tool": "Tool result"}


class DecisionError(RuntimeError):
    pass


def routing_state(messages: list) -> str:
    """The routing prompt as one context: its system prompt, then the turns the router was given."""
    system = "\n".join(m.get("content") or "" for m in messages if m["role"] == "system")
    turns = "\n".join(
        f"{_SPEAKER.get(m['role'], m['role'])}: {(m.get('content') or '').strip()}"
        for m in messages
        if m["role"] != "system"
    )
    return f"{system}\n\n# Conversation so far\n{turns}"


def edge_criteria(tools: list) -> dict:
    """Each edge's guidance is its routing tool's description, verbatim."""
    return {t["function"]["name"]: t["function"].get("description") or t["function"]["name"] for t in tools}


def _keyed_question(criteria: dict) -> dict:
    return {QUESTION: {"type": "choice", "instructions": INSTRUCTIONS, "criteria": criteria}}


def _keyed_answer(body: dict) -> tuple:
    answer = body["answers"][QUESTION]
    return answer["choice"], answer.get("confidence")


def _listed_answer(body: dict) -> tuple:
    answer = next(a for a in body["answers"] if a["name"] == QUESTION)
    if answer["type"] == "refusal":
        raise DecisionError("the model declined the question")
    return answer["choice"], answer.get("confidence")


@dataclass(frozen=True)
class _Api:
    url: str
    key_env: str
    default_model: str
    headers: Callable[[str], dict]
    body: Callable[[str, str, dict], dict]
    answer: Callable[[dict], tuple]


def _bearer(key: str) -> dict:
    return {"Authorization": f"Bearer {key}"}


APIS = {
    DecisionProvider.TYPESAFE: _Api(
        url="https://api.typesafe.ai/v1/systemone",
        key_env="TYPESAFE_API_KEY",
        default_model="jev-latest",
        headers=_bearer,
        body=lambda model, state, criteria: {"model": model, "state": state, "questions": _keyed_question(criteria)},
        answer=_keyed_answer,
    ),
    DecisionProvider.GEMINI_DECISIONS: _Api(
        url="https://generativelanguage.googleapis.com/v1beta/decisions",
        key_env="GOOGLE_API_KEY",
        default_model="gemini-decision-1",
        headers=lambda key: {"x-goog-api-key": key},
        body=lambda model, state, criteria: {"model": model, "input": state, "questions": _keyed_question(criteria)},
        answer=_keyed_answer,
    ),
    DecisionProvider.OPENAI_DECISIONS: _Api(
        url="https://api.openai.com/v1/decisions",
        key_env="OPENAI_API_KEY",
        default_model="gpt-6-luna",
        headers=_bearer,
        body=lambda model, state, criteria: {
            "model": model,
            "input": state,
            "questions": [
                {
                    "type": "choice",
                    "name": QUESTION,
                    "instructions": INSTRUCTIONS,
                    "choices": [{"value": name, "description": text} for name, text in criteria.items()],
                }
            ],
        },
        answer=_listed_answer,
    ),
}


class DecisionRouter:
    """Routes a hop on a decision model, with the LLM router's route() contract."""

    def __init__(self, provider: str, fallback, model: Optional[str] = None):
        self.provider = DecisionProvider(provider)
        self.api = APIS[self.provider]
        self.model = model or self.api.default_model
        self.key = os.getenv(self.api.key_env)
        if not self.key:
            raise ValueError(f"No API key for decision routing on {self.provider.value}: set {self.api.key_env}")
        # The LLM router the agent would otherwise use: fills edge parameters and routes when the decision API fails.
        self.fallback = fallback
        self.http = get_shared_http_client(base_url=self.api.url, http2=False)

    async def route(self, messages, tools, tool_choice="required", meta_info=None):
        tools = json.loads(tools) if isinstance(tools, str) else tools
        try:
            name, confidence, usage = await self._decide(messages, tools)
        except Exception as e:
            logger.error(f"Decision routing on {self.provider.value} failed, routing on the LLM router: {e!r}")
            return await self.fallback.route(messages, tools, tool_choice, meta_info)
        arguments = await self._edge_parameters(messages, tools, name, meta_info)
        return {
            "function_name": name,
            "arguments": {**arguments, CONFIDENCE_KEY: confidence},
            "usage": usage,
            "service_tier": None,
            "overflowed": False,
        }

    async def _decide(self, messages, tools) -> tuple:
        criteria = edge_criteria(tools)
        response = await self.http.post(
            self.api.url,
            headers=self.api.headers(self.key),
            json=self.api.body(self.model, routing_state(messages), criteria),
        )
        response.raise_for_status()
        body = response.json()
        name, confidence = self.api.answer(body)
        if name not in criteria:
            raise DecisionError(f"answered {name!r}, which is not an edge of this node")
        usage = body.get("usage") or {}
        return (
            name,
            confidence,
            {"input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens")},
        )

    async def _edge_parameters(self, messages, tools, name: str, meta_info) -> dict:
        """The chosen edge's own parameters, from the LLM router given the routing request and forced onto that edge."""
        edge = next(t for t in tools if t["function"]["name"] == name)
        if not any(k not in TRAILING_KEYS for k in (edge["function"].get("parameters") or {}).get("properties") or {}):
            return {}
        # The whole request, not the one edge alone: offered only that edge, or without its trailing fields, the
        # LLM copies the caller's words where the parameter's description asks for a normalised value.
        filled = await self.fallback.route(messages, tools, {"type": "function", "function": {"name": name}}, meta_info)
        if not filled or filled.get("function_name") != name:
            logger.warning(f"Edge parameters for {name} could not be filled; transitioning without them")
            return {}
        return {k: v for k, v in filled["arguments"].items() if k not in TRAILING_KEYS}
