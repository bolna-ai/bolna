"""Tests for structure_system_prompt's "### Variables:" rendering.

Covers: the five default variables, agent-declared SIP header pass-through
variables (sip-trunk), empty/missing-value skipping, ordering, and backward
compatibility when no sip_header_variables are present.
"""

import pytz

from bolna.helpers.utils import structure_system_prompt

TZ = pytz.timezone("Asia/Kolkata")


def _build(context_data, system_prompt="You are a helpful agent."):
    return structure_system_prompt(
        system_prompt=system_prompt,
        run_id="exec-123",
        assistant_id="agent-abc",
        call_sid="01KTDVSTHP23A8WSX9N3ZHSDCG",
        context_data=context_data,
        timezone=TZ,
    )


def test_default_variables_rendered():
    out = _build({"recipient_data": {"agent_number": "+918065755297", "user_number": "09445981430"}})
    assert "## Call information:\n\n### Variables:\n" in out
    assert 'agent_id is "agent-abc"' in out
    assert 'execution_id is "exec-123"' in out
    assert 'agent_number is "+918065755297"' in out
    assert 'user_number is "09445981430"' in out
    assert 'call_sid is "01KTDVSTHP23A8WSX9N3ZHSDCG"' in out


def test_sip_header_variables_rendered_after_defaults():
    out = _build(
        {
            "recipient_data": {
                "agent_number": "+918065755297",
                "user_number": "09445981430",
                "campaign_id": "DIWALI-2026",
                "lead_score": "87",
            },
            "sip_header_variables": ["campaign_id", "lead_score"],
        }
    )
    assert 'campaign_id is "DIWALI-2026"' in out
    assert 'lead_score is "87"' in out
    # defaults come before the SIP header pass-through variables
    assert out.index('call_sid is') < out.index('campaign_id is')


def test_sip_header_variable_with_empty_or_missing_value_skipped():
    out = _build(
        {
            "recipient_data": {"campaign_id": "DIWALI-2026", "empty_one": ""},
            "sip_header_variables": ["campaign_id", "empty_one", "missing_var"],
        }
    )
    assert 'campaign_id is "DIWALI-2026"' in out
    assert "empty_one is" not in out  # present but empty value
    assert "missing_var is" not in out  # not in recipient_data


def test_no_sip_header_variables_key_is_backward_compatible():
    out = _build({"recipient_data": {"agent_number": "+10000000000"}})
    assert 'agent_id is "agent-abc"' in out
    assert "campaign_id" not in out


def test_sip_header_variable_does_not_duplicate_default():
    # A SIP variable named like a default must not produce a second line.
    out = _build(
        {
            "recipient_data": {"agent_number": "+918065755297", "user_number": "09445981430"},
            "sip_header_variables": ["agent_number"],
        }
    )
    assert out.count('agent_number is') == 1


def test_none_context_data_has_no_variables_section():
    out = structure_system_prompt("P", "r", "a", "CS", None, TZ)
    assert "## Call information" not in out
