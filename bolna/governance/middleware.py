"""Opt-in governance for the voice pipeline.

Disabled by default. When enabled, every decision is regex/policy only — no extra
LLM call — and emits a JSON receipt that never contains the original PII span.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from bolna.helpers.logger_config import configure_logger

from .patterns import DEFAULT_PII_TYPES, find_and_redact

logger = configure_logger(__name__)

# Rough USD / 1M tokens when the agent does not override rates. Estimates only.
_DEFAULT_RATES = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-nano": (0.10, 0.40),
}
_FALLBACK_RATE = (0.50, 1.50)


def _normalize_model(model: str | None) -> str:
    if not model:
        return ""
    return model.split("/")[-1].strip().lower()


@dataclass
class GovernanceDecision:
    allow: bool
    action: str
    reason: str
    receipt: dict


@dataclass
class GovernanceLedger:
    """Shared across tasks in one run so per-session caps survive extraction steps."""

    estimated_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    receipts: list[dict] = field(default_factory=list)
    source_config: dict | None = None


class GovernanceMiddleware:
    def __init__(self, config: dict | None, run_id: str | None, ledger: GovernanceLedger | None = None):
        cfg = dict(config or {})
        self.enabled = bool(cfg.get("enabled"))
        self.mode = (cfg.get("mode") or "enforce").lower()
        if self.mode not in ("enforce", "monitor"):
            self.mode = "enforce"
        self.redact_pii = cfg.get("redact_pii", True)
        self.output_dlp = cfg.get("output_dlp", True)
        pii_types = cfg.get("pii_types")
        self.pii_types = tuple(pii_types) if pii_types else DEFAULT_PII_TYPES
        self.max_cost_per_call = cfg.get("max_cost_per_call")
        self.max_cost_per_session = cfg.get("max_cost_per_session")
        self.usd_per_1m_input = cfg.get("usd_per_1m_input")
        self.usd_per_1m_output = cfg.get("usd_per_1m_output")
        allowed = cfg.get("allowed_tools")
        denied = cfg.get("denied_tools")
        self.allowed_tools = {str(x) for x in allowed} if allowed is not None else None
        self.denied_tools = {str(x) for x in denied} if denied else set()
        self.run_id = run_id
        self.ledger = ledger or GovernanceLedger()
        if self.enabled and getattr(self.ledger, "source_config", None) is None:
            self.ledger.source_config = dict(cfg)
        elif not self.enabled and getattr(self.ledger, "source_config", None):
            inherited = dict(self.ledger.source_config)
            self.enabled = True
            self.mode = (inherited.get("mode") or "enforce").lower()
            self.redact_pii = bool(inherited.get("redact_pii", True))
            self.output_dlp = bool(inherited.get("output_dlp", True))
            self.max_cost_per_call = None
            self.max_cost_per_session = inherited.get("max_cost_per_session")
            self.usd_per_1m_input = inherited.get("usd_per_1m_input")
            self.usd_per_1m_output = inherited.get("usd_per_1m_output")
        self._call_usd = 0.0
        self._blocked_llm = False

    @classmethod
    def from_conversation_config(cls, conversation_config, run_id, ledger=None):
        cfg = conversation_config or {}
        if hasattr(cfg, "model_dump"):
            cfg = cfg.model_dump()
        elif hasattr(cfg, "dict"):
            cfg = cfg.dict()
        gov = cfg.get("governance") if isinstance(cfg, dict) else None
        if hasattr(gov, "model_dump"):
            gov = gov.model_dump()
        elif hasattr(gov, "dict"):
            gov = gov.dict()
        return cls(gov, run_id, ledger=ledger)

    @classmethod
    def disabled(cls, run_id=None):
        return cls({"enabled": False}, run_id)

    def _receipt(self, *, stage, action, reason, extra=None):
        body = {
            "id": str(uuid.uuid4()),
            "ts": round(time.time(), 6),
            "run_id": self.run_id,
            "stage": stage,
            "action": action,
            "reason": reason,
            "mode": self.mode,
        }
        if extra:
            body.update(extra)
        self.ledger.receipts.append(body)
        return body

    def _enforce(self) -> bool:
        return self.mode == "enforce"

    def receipts(self) -> list[dict]:
        return list(self.ledger.receipts)

    def allows_llm(self) -> bool:
        if not self.enabled:
            return True
        return not (self._enforce() and self._blocked_llm)

    def inspect_transcript(self, text: str, meta_info=None) -> tuple[str, GovernanceDecision]:
        if not self.enabled or not self.redact_pii:
            return text, GovernanceDecision(True, "allow", "governance_disabled", {})
        redacted, labels = find_and_redact(text, self.pii_types)
        if not labels:
            return text, GovernanceDecision(True, "allow", "no_pii", {})
        receipt = self._receipt(
            stage="transcript_pii",
            action="redact",
            reason="pii_detected",
            extra={"pii_types": labels, "origin": (meta_info or {}).get("origin", "transcriber")},
        )
        logger.info(f"governance redacted transcript pii={labels} run_id={self.run_id}")
        return redacted, GovernanceDecision(True, "redact", "pii_detected", receipt)

    def inspect_output(self, text: str, meta_info=None) -> tuple[str, GovernanceDecision]:
        if not self.enabled or not self.output_dlp:
            return text, GovernanceDecision(True, "allow", "governance_disabled", {})
        if not isinstance(text, str):
            return text, GovernanceDecision(True, "allow", "non_text", {})
        redacted, labels = find_and_redact(text, self.pii_types)
        if not labels:
            return text, GovernanceDecision(True, "allow", "no_pii", {})
        receipt = self._receipt(
            stage="output_dlp",
            action="redact",
            reason="pii_detected",
            extra={"pii_types": labels},
        )
        logger.info(f"governance redacted synthesizer pii={labels} run_id={self.run_id}")
        return redacted, GovernanceDecision(True, "redact", "pii_detected", receipt)

    def authorize_tool(self, name: str, arguments=None, meta_info=None) -> GovernanceDecision:
        if not self.enabled:
            return GovernanceDecision(True, "allow", "governance_disabled", {})
        extra = {"tool": name}
        if name in self.denied_tools:
            receipt = self._receipt(stage="tool_auth", action="deny", reason="tool_denied", extra=extra)
            if self._enforce():
                logger.warning(f"governance denied tool {name} run_id={self.run_id}")
                return GovernanceDecision(False, "deny", "tool_denied", receipt)
            return GovernanceDecision(True, "monitor_deny", "tool_denied", receipt)
        if self.allowed_tools is not None and name not in self.allowed_tools:
            receipt = self._receipt(stage="tool_auth", action="deny", reason="tool_not_allowlisted", extra=extra)
            if self._enforce():
                logger.warning(f"governance blocked non-allowlisted tool {name} run_id={self.run_id}")
                return GovernanceDecision(False, "deny", "tool_not_allowlisted", receipt)
            return GovernanceDecision(True, "monitor_deny", "tool_not_allowlisted", receipt)
        if arguments:
            blob = arguments if isinstance(arguments, str) else str(arguments)
            _, labels = find_and_redact(blob, self.pii_types)
            if labels:
                extra["pii_types"] = labels
                receipt = self._receipt(stage="tool_auth", action="deny", reason="tool_args_pii", extra=extra)
                if self._enforce():
                    return GovernanceDecision(False, "deny", "tool_args_pii", receipt)
                return GovernanceDecision(True, "monitor_deny", "tool_args_pii", receipt)
        receipt = self._receipt(stage="tool_auth", action="allow", reason="tool_authorized", extra=extra)
        return GovernanceDecision(True, "allow", "tool_authorized", receipt)

    def _estimate_usd(self, model, input_tokens, output_tokens) -> float:
        in_tok = int(input_tokens or 0)
        out_tok = int(output_tokens or 0)
        default_in, default_out = _DEFAULT_RATES.get(_normalize_model(model), _FALLBACK_RATE)
        in_rate = float(self.usd_per_1m_input) if self.usd_per_1m_input is not None else default_in
        out_rate = float(self.usd_per_1m_output) if self.usd_per_1m_output is not None else default_out
        return (in_tok / 1_000_000.0) * in_rate + (out_tok / 1_000_000.0) * out_rate

    def record_usage(self, *, model=None, input_tokens=None, output_tokens=None, meta_info=None) -> GovernanceDecision:
        if not self.enabled:
            return GovernanceDecision(True, "allow", "governance_disabled", {})
        usd = self._estimate_usd(model, input_tokens, output_tokens)
        self._call_usd += usd
        self.ledger.estimated_usd += usd
        self.ledger.input_tokens += int(input_tokens or 0)
        self.ledger.output_tokens += int(output_tokens or 0)
        extra = {
            "model": model,
            "input_tokens": int(input_tokens or 0),
            "output_tokens": int(output_tokens or 0),
            "estimated_usd": round(usd, 8),
            "call_estimated_usd": round(self._call_usd, 8),
            "session_estimated_usd": round(self.ledger.estimated_usd, 8),
        }
        breached = self._budget_breach()
        if breached:
            self._blocked_llm = True
            receipt = self._receipt(stage="cost_cap", action="deny", reason=breached, extra=extra)
            if self._enforce():
                logger.warning(f"governance cost cap {breached} run_id={self.run_id} extra={extra}")
                return GovernanceDecision(False, "deny", breached, receipt)
            return GovernanceDecision(True, "monitor_deny", breached, receipt)
        receipt = self._receipt(stage="cost_cap", action="allow", reason="within_budget", extra=extra)
        return GovernanceDecision(True, "allow", "within_budget", receipt)

    def _budget_breach(self) -> str | None:
        if self.max_cost_per_call is not None and self._call_usd >= float(self.max_cost_per_call):
            return "max_cost_per_call"
        if self.max_cost_per_session is not None and self.ledger.estimated_usd >= float(self.max_cost_per_session):
            return "max_cost_per_session"
        return None

    def check_budget(self) -> GovernanceDecision:
        if not self.enabled:
            return GovernanceDecision(True, "allow", "governance_disabled", {})
        if self._blocked_llm:
            return GovernanceDecision(
                not self._enforce(), "deny" if self._enforce() else "monitor_deny", "already_blocked", {}
            )
        breached = self._budget_breach()
        if not breached:
            return GovernanceDecision(True, "allow", "within_budget", {})
        self._blocked_llm = True
        receipt = self._receipt(
            stage="cost_cap",
            action="deny",
            reason=breached,
            extra={
                "call_estimated_usd": round(self._call_usd, 8),
                "session_estimated_usd": round(self.ledger.estimated_usd, 8),
            },
        )
        if self._enforce():
            return GovernanceDecision(False, "deny", breached, receipt)
        return GovernanceDecision(True, "monitor_deny", breached, receipt)
