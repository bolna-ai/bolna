"""Decision models: one choice question about a context, answered with a choice and a confidence, never text."""

import asyncio
import os
from dataclasses import dataclass
from typing import Callable, Optional

from bolna.enums import DecisionProvider
from bolna.helpers.logger_config import configure_logger
from bolna.llms.http_client_pool import get_shared_http_client

logger = configure_logger(__name__)

# Deadline for one decision, several times what a slow one takes, so a caller's fallback is not kept waiting.
DECISION_TIMEOUT_S = 2.0
_QUESTION = "decision"


class DecisionError(RuntimeError):
    """The API replied, but not with one of the choices."""


@dataclass(frozen=True)
class Decision:
    choice: str
    confidence: Optional[float]
    usage: dict


def _keyed_question(instructions: str, choices: dict) -> dict:
    return {_QUESTION: {"type": "choice", "instructions": instructions, "criteria": choices}}


def _keyed_answer(body: dict) -> tuple:
    answer = body["answers"][_QUESTION]
    return answer["choice"], answer.get("confidence")


def _listed_question(instructions: str, choices: dict) -> list:
    listed = [{"value": value, "description": meaning} for value, meaning in choices.items()]
    return [{"type": "choice", "name": _QUESTION, "instructions": instructions, "choices": listed}]


def _listed_answer(body: dict) -> tuple:
    answer = next(a for a in body["answers"] if a["name"] == _QUESTION)
    return answer["choice"], answer.get("confidence")


def _bearer(key: str) -> dict:
    return {"Authorization": f"Bearer {key}"}


@dataclass(frozen=True)
class _Api:
    url: str
    key_env: str
    default_model: str
    headers: Callable[[str], dict]
    # The request field that carries the context.
    context_field: str
    question: Callable[[str, dict], object]
    answer: Callable[[dict], tuple]
    # What the usage record puts before "input_tokens" and "output_tokens".
    usage_prefix: str = ""


APIS = {
    DecisionProvider.TYPESAFE: _Api(
        url="https://api.typesafe.ai/v1/systemone",
        key_env="TYPESAFE_API_KEY",
        default_model="jev-latest",
        headers=_bearer,
        context_field="state",
        question=_keyed_question,
        answer=_keyed_answer,
    ),
    DecisionProvider.GEMINI_DECISIONS: _Api(
        url="https://generativelanguage.googleapis.com/v1beta/decisions",
        key_env="GOOGLE_API_KEY",
        default_model="gemini-decision-1",
        headers=lambda key: {"x-goog-api-key": key},
        context_field="input",
        question=_keyed_question,
        answer=_keyed_answer,
        usage_prefix="total_",
    ),
    DecisionProvider.OPENAI_DECISIONS: _Api(
        url="https://api.openai.com/v1/decisions",
        key_env="OPENAI_API_KEY",
        default_model="gpt-6-luna",
        headers=_bearer,
        context_field="input",
        question=_listed_question,
        answer=_listed_answer,
    ),
}


class DecisionModel:
    """One decision provider's API, asked one choice question at a time."""

    def __init__(self, provider: str, model: Optional[str] = None, key: Optional[str] = None):
        self.provider = DecisionProvider(provider)
        self._api = APIS[self.provider]
        self.model = model or self._api.default_model
        self._key = key or os.getenv(self._api.key_env)
        if not self._key:
            logger.error(f"No {self._api.key_env} for the {self.provider.value} decision model")
        self._http = get_shared_http_client(base_url=self._api.url, http2=False)

    @property
    def available(self) -> bool:
        return bool(self._key)

    async def choose(self, context: str, instructions: str, choices: dict) -> Decision:
        """Pick one of `choices`, which maps each value to what choosing it means."""
        request = {
            "model": self.model,
            self._api.context_field: context,
            "questions": self._api.question(instructions, choices),
        }
        post = self._http.post(self._api.url, headers=self._api.headers(self._key), json=request)
        response = await asyncio.wait_for(post, DECISION_TIMEOUT_S)
        response.raise_for_status()
        body = response.json()
        try:
            choice, confidence = self._api.answer(body)
        except (KeyError, TypeError, StopIteration) as e:
            raise DecisionError(f"no choice in the answer: {e!r}") from e
        if choice not in choices:
            raise DecisionError(f"answered {choice!r}, which is not one of the choices")
        reported, prefix = body.get("usage") or {}, self._api.usage_prefix
        usage = {key: reported.get(prefix + key) for key in ("input_tokens", "output_tokens")}
        return Decision(choice, confidence, usage)
