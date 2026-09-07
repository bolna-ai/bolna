"""Governance middleware: PII redaction, tool auth, cost caps, receipts.

Guards the opt-in path added for issue 933. The original span of any PII match
must never appear in a receipt. Disabled config is a no-op so existing agents
do not change behaviour.
"""

import json

from bolna.governance import GovernanceLedger, GovernanceMiddleware
from bolna.governance.patterns import find_and_redact, luhn_ok
from bolna.models import ConversationConfig, GovernanceConfig


VALID_VISA = "4111 1111 1111 1111"
INVALID_CARD = "4111 1111 1111 1112"
SSN = "123-45-6789"


def _enabled(**kwargs):
    cfg = {"enabled": True, "mode": "enforce"}
    cfg.update(kwargs)
    return GovernanceMiddleware(cfg, run_id="run-1")


def test_disabled_is_noop():
    g = GovernanceMiddleware({"enabled": False}, run_id="r")
    text, decision = g.inspect_transcript(f"my ssn is {SSN}")
    assert SSN in text
    assert decision.allow
    assert g.receipts() == []
    assert g.allows_llm()
    assert g.authorize_tool("transfer_call").allow


def test_ssn_is_redacted_and_receipt_has_no_raw_span():
    g = _enabled()
    text, decision = g.inspect_transcript(f"my social is {SSN}")
    assert SSN not in text
    assert "[SSN]" in text
    assert decision.action == "redact"
    blob = json.dumps(g.receipts())
    assert SSN not in blob
    assert "ssn" in g.receipts()[0]["pii_types"]


def test_luhn_valid_card_redacted_invalid_left_alone():
    g = _enabled()
    redacted, decision = g.inspect_transcript(f"card {VALID_VISA}")
    assert VALID_VISA not in redacted
    assert "[CREDIT_CARD]" in redacted
    assert decision.action == "redact"

    untouched, skip = g.inspect_transcript(f"order {INVALID_CARD}")
    assert INVALID_CARD in untouched
    assert skip.action == "allow"


def test_email_and_phone():
    g = _enabled()
    text, _ = g.inspect_transcript("reach me at ada@example.com or 415-555-0100")
    assert "ada@example.com" not in text
    assert "415-555-0100" not in text
    assert "[EMAIL]" in text
    assert "[PHONE]" in text


def test_account_number_only_when_opted_in():
    default = _enabled()
    text, _ = default.inspect_transcript("acct 12345678901234")
    assert "12345678901234" in text

    opted = _enabled(pii_types=["account_number"])
    redacted, _ = opted.inspect_transcript("acct 12345678901234")
    assert "[ACCOUNT_NUMBER]" in redacted


def test_tool_allowlist_and_denylist():
    g = _enabled(allowed_tools=["lookup_order"], denied_tools=["transfer_call"])
    assert not g.authorize_tool("transfer_call").allow
    assert not g.authorize_tool("end_call").allow
    assert g.authorize_tool("lookup_order").allow


def test_tool_args_with_pii_are_denied():
    g = _enabled()
    decision = g.authorize_tool("update_account", {"ssn": SSN})
    assert not decision.allow
    assert decision.reason == "tool_args_pii"
    assert SSN not in json.dumps(g.receipts())


def test_monitor_mode_logs_but_does_not_block_tools():
    g = _enabled(mode="monitor", denied_tools=["transfer_call"])
    decision = g.authorize_tool("transfer_call")
    assert decision.allow
    assert decision.action == "monitor_deny"
    assert g.receipts()[-1]["action"] == "deny"


def test_cost_cap_blocks_further_llm_after_breach():
    g = _enabled(max_cost_per_call=0.0001, usd_per_1m_input=10.0, usd_per_1m_output=10.0)
    first = g.record_usage(model="gpt-4o-mini", input_tokens=20_000, output_tokens=20_000)
    assert not first.allow
    assert first.reason == "max_cost_per_call"
    assert not g.allows_llm()
    assert not g.check_budget().allow


def test_session_ledger_is_shared_across_middleware_instances():
    ledger = GovernanceLedger()
    first = GovernanceMiddleware(
        {"enabled": True, "max_cost_per_session": 0.0001, "usd_per_1m_input": 10.0, "usd_per_1m_output": 0},
        run_id="r",
        ledger=ledger,
    )
    first.record_usage(model="x", input_tokens=50_000, output_tokens=0)
    second = GovernanceMiddleware(
        {"enabled": True, "max_cost_per_session": 0.0001, "usd_per_1m_input": 10.0, "usd_per_1m_output": 0},
        run_id="r",
        ledger=ledger,
    )
    assert not second.check_budget().allow
    assert len(ledger.receipts) >= 2


def test_output_dlp_redacts_synthesizer_text():
    g = _enabled()
    spoken, decision = g.inspect_output(f"your ssn is {SSN}")
    assert "[SSN]" in spoken
    assert decision.action == "redact"


def test_from_conversation_config_reads_nested_model():
    cfg = ConversationConfig(governance=GovernanceConfig(enabled=True, denied_tools=["transfer_call"]))
    g = GovernanceMiddleware.from_conversation_config(cfg, run_id="r")
    assert g.enabled
    assert not g.authorize_tool("transfer_call").allow


def test_luhn_helper():
    assert luhn_ok(VALID_VISA)
    assert not luhn_ok(INVALID_CARD)
    assert not luhn_ok("123")


def test_find_and_redact_empty():
    text, labels = find_and_redact("", ["ssn"])
    assert text == ""
    assert labels == []


def test_missing_governance_key_is_disabled():
    g = GovernanceMiddleware.from_conversation_config({}, "r")
    assert not g.enabled
    text, _ = g.inspect_transcript(f"ssn {SSN}")
    assert SSN in text


def test_task_manager_helpers_redact_and_block():
    from unittest.mock import MagicMock

    from bolna.agent_manager.task_manager import TaskManager

    tm = MagicMock()
    tm.run_id = "r"
    tm.governance = _enabled(max_cost_per_call=0.0001, usd_per_1m_input=10.0, usd_per_1m_output=10.0)
    tm._govern_user_text = TaskManager._govern_user_text.__get__(tm, TaskManager)
    tm._governance_blocks_generation = TaskManager._governance_blocks_generation.__get__(tm, TaskManager)

    assert "[SSN]" in tm._govern_user_text(f"ssn {SSN}")
    assert not tm._governance_blocks_generation()
    tm.governance.record_usage(model="x", input_tokens=50_000, output_tokens=50_000)
    assert tm._governance_blocks_generation()
