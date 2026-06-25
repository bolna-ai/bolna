"""Regression tests for the 'agent reads the call_sid aloud during transfer' bug.

Incident run_id a599f8b6-...: an Azure model was given call_sid in its system prompt and
echoed it into spoken text as "{call_sid:6646c0f7-...}", which ElevenLabs read aloud.

Two-layer fix:
1. structure_system_prompt no longer exposes call_sid as a prompt variable.
2. redact_call_identifiers strips leaked {call_sid:...}/{stream_sid:...} fragments before TTS.
"""

import pytz

from bolna.helpers.utils import redact_call_identifiers, structure_system_prompt


# The exact spoken text from the incident (Hindi transfer message + leaked sid).
INCIDENT_TEXT = (
    "अपनी preferred language confirm करने के लिए धन्यवाद। मैं आपकी call अब हमारे "
    "support executive को transfer कर रही हूँ। कृपया line पर बने रहें। "
    "{call_sid:6646c0f7-b8ac-4303-bc42-0717491fcd5a}"
)


def test_incident_call_sid_stripped():
    cleaned = redact_call_identifiers(INCIDENT_TEXT)
    assert "6646c0f7-b8ac-4303-bc42-0717491fcd5a" not in cleaned
    assert "call_sid" not in cleaned
    assert cleaned.endswith("कृपया line पर बने रहें।")


def test_redacts_json_and_kv_and_stream_sid_forms():
    assert redact_call_identifiers('Transferring now {"call_sid": "abc-123"}') == "Transferring now"
    assert redact_call_identifiers("hold on {call_sid=xyz} please") == "hold on please"
    assert redact_call_identifiers("with {stream_sid:99} ok") == "with ok"


def test_legitimate_text_is_untouched():
    # No fragment, even when the substring 'sid' appears (e.g. 'outside', 'reside').
    assert redact_call_identifiers("the kids are outside") == "the kids are outside"
    assert redact_call_identifiers("normal sentence with no id") == "normal sentence with no id"
    assert redact_call_identifiers("<end_of_stream>") == "<end_of_stream>"


def test_non_string_and_empty_inputs_pass_through():
    assert redact_call_identifiers(None) is None
    assert redact_call_identifiers("") == ""
    assert redact_call_identifiers(123) == 123


def test_system_prompt_no_longer_exposes_call_sid():
    ctx = {"recipient_data": {"call_sid": "6646c0f7-SECRET", "agent_number": "+91999", "user_number": "+91888"}}
    out = structure_system_prompt(
        "You are a bot.", "run-123", "agent-7", "6646c0f7-SECRET", ctx, pytz.timezone("Asia/Kolkata")
    )
    # The sid value and its variable line must be gone...
    assert "6646c0f7-SECRET" not in out
    assert "call_sid is" not in out
    # ...but the other call-information variables stay intact.
    assert 'execution_id is "run-123"' in out
    assert 'agent_id is "agent-7"' in out
    assert 'agent_number is "+91999"' in out
