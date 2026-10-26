"""Veto for the speculative reply an eager end of turn starts.

An eager end of turn starts the reply before the caller has provably stopped. When the caller
goes on talking, that reply is thrown away (TurnResumed) after it has already spent LLM tokens
and, sometimes, TTS. The gate asks TypeSafe's Jev whether the caller has finished, in parallel
with the speculative reply, and drops the reply early when Jev says the caller is mid-sentence.
It never delays a reply: the speculation starts exactly as it would without the gate.

What Jev is asked is plain agent config (task_config.speculation_gate), so an agent can swap the
default "has the caller finished?" question for its own noul / choice / score questions and say,
per question, which answers keep the speculation.
"""

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from bolna.helpers.logger_config import configure_logger
from bolna.llms.http_client_pool import get_shared_http_client

logger = configure_logger(__name__)

TYPESAFE_API_BASE = "https://api.typesafe.ai"
SYSTEM_ONE_PATH = "/v1/systemone"
DEFAULT_JEV_MODEL = "jev-latest"
# Jev answers in ~0.5s; past this the turn has settled one way or the other and the verdict is moot.
DEFAULT_TIMEOUT_MS = 1500
# Prior turns sent as context; the caller's half-sentence is judged against what was just asked.
DEFAULT_HISTORY_TURNS = 4

DEFAULT_QUESTIONS = {
    "complete": {
        "type": "noul",
        "instructions": (
            "`caller_so_far` is a live phone transcript of what the caller has said in this turn; "
            "it may be cut off mid-sentence. Has the caller finished what they wanted to say, "
            "so the agent can reply now without waiting for more words?"
        ),
        "criteria": {
            "true": "The turn reads as complete: a full question, answer, or request the agent can respond to.",
            "false": "The caller is mid-sentence, mid-number, or has only started their request.",
        },
    }
}
DEFAULT_RULES = {"complete": {"min": 0.5}}


def resolve_typesafe_api_key(provider_api_keys: Optional[Dict[str, str]] = None) -> str:
    """The call's own typesafe key wins; else TYPESAFE_API_KEY."""
    return (provider_api_keys or {}).get("typesafe") or os.getenv("TYPESAFE_API_KEY") or ""


@dataclass
class GateVerdict:
    allow: bool
    # Rule names that failed; empty when allowed or when the gate failed open.
    failed: List[str] = field(default_factory=list)
    answers: Dict[str, Any] = field(default_factory=dict)
    latency_ms: Optional[float] = None
    usage: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


def _in_range(value: Optional[float], rule: Dict[str, Any]) -> bool:
    if value is None:
        return False
    if rule.get("min") is not None and value < rule["min"]:
        return False
    if rule.get("max") is not None and value > rule["max"]:
        return False
    return True


def _rule_passes(rule: Dict[str, Any], answer: Dict[str, Any]) -> bool:
    kind = answer.get("type")
    if kind == "noul":
        return _in_range(answer.get("noul"), rule)
    if kind == "score":
        return _in_range(answer.get("score"), rule)
    if kind == "choice":
        choice = answer.get("choice")
        if rule.get("allow") is not None and choice not in rule["allow"]:
            return False
        if rule.get("deny") is not None and choice in rule["deny"]:
            return False
        # min/max on a choice bound the confidence in the chosen label.
        return _in_range(answer.get("confidence"), rule) if rule.get("min") or rule.get("max") else True
    # An answer type this code does not know cannot veto.
    return True


def evaluate_rules(rules: Dict[str, Dict[str, Any]], answers: Dict[str, Any]) -> List[str]:
    """Names of the rules the answers fail. A rule whose answer is missing fails."""
    failed = []
    for name, rule in rules.items():
        answer = answers.get(name)
        if not isinstance(answer, dict) or not _rule_passes(rule, answer):
            failed.append(name)
    return failed


class SpeculationGate:
    def __init__(self, config: Dict[str, Any], provider_api_keys: Optional[Dict[str, str]] = None, run_id=None):
        self.run_id = run_id
        self.model = config.get("model") or os.getenv("JEV_MODEL") or DEFAULT_JEV_MODEL
        self.timeout_ms = config.get("timeout_ms") or DEFAULT_TIMEOUT_MS
        history_turns = config.get("history_turns")
        self.history_turns = DEFAULT_HISTORY_TURNS if history_turns is None else history_turns
        self.questions = config.get("questions") or DEFAULT_QUESTIONS
        # Custom questions with no rules would never veto; default to the default question's rule
        # only when the default question is the one being asked.
        self.rules = config.get("rules") or (DEFAULT_RULES if self.questions is DEFAULT_QUESTIONS else {})
        # Static facts about the agent (e.g. what it books) that make the questions answerable.
        self.context = config.get("context")
        self.api_key = resolve_typesafe_api_key(provider_api_keys)
        self.base_url = os.getenv("TYPESAFE_BASE_URL") or TYPESAFE_API_BASE
        self.enabled = bool(self.api_key) and bool(self.rules)
        if not self.api_key:
            logger.error("SpeculationGate: no TypeSafe API key (TYPESAFE_API_KEY) — gate disabled")
        elif not self.rules:
            logger.error("SpeculationGate: custom questions without rules can never veto — gate disabled")
        self.usage_totals = {"input_tokens": 0, "output_tokens": 0, "requests": 0, "vetoes": 0}

    def build_state(self, transcript: str, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        turns = [
            {"role": row.get("role"), "content": row.get("content")}
            for row in history
            if row.get("role") in ("user", "assistant") and isinstance(row.get("content"), str) and row.get("content")
        ]
        state: Dict[str, Any] = {
            "conversation": turns[-self.history_turns :] if self.history_turns else [],
            "caller_so_far": transcript,
        }
        if self.context:
            state["context"] = self.context
        return state

    async def evaluate(self, transcript: str, history: List[Dict[str, Any]]) -> GateVerdict:
        """Fails open: any error or timeout keeps the speculation, exactly as without the gate."""
        body = {"model": self.model, "state": self.build_state(transcript, history), "questions": self.questions}
        start = time.perf_counter()
        try:
            client = get_shared_http_client()
            response = await client.post(
                f"{self.base_url}{SYSTEM_ONE_PATH}",
                json=body,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=self.timeout_ms / 1000,
            )
            latency_ms = round((time.perf_counter() - start) * 1000, 1)
            if response.status_code != 200:
                return GateVerdict(
                    allow=True, latency_ms=latency_ms, error=f"HTTP {response.status_code}: {response.text[:200]}"
                )
            payload = response.json()
        except Exception as e:
            return GateVerdict(
                allow=True, latency_ms=round((time.perf_counter() - start) * 1000, 1), error=f"{type(e).__name__}: {e}"
            )

        answers = payload.get("answers") or {}
        usage = payload.get("usage") or {}
        self.usage_totals["requests"] += 1
        self.usage_totals["input_tokens"] += usage.get("input_tokens") or 0
        self.usage_totals["output_tokens"] += usage.get("output_tokens") or 0
        failed = evaluate_rules(self.rules, answers)
        if failed:
            self.usage_totals["vetoes"] += 1
        return GateVerdict(allow=not failed, failed=failed, answers=answers, latency_ms=latency_ms, usage=usage)
