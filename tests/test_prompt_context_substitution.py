"""Filling a prompt template from caller-supplied recipient data.

The server-owned telephony ids must render empty rather than handing the model a real call_sid,
and an unknown placeholder must render empty rather than leaving braces in what the agent says.
"""

from bolna.helpers.utils import update_prompt_with_context

TEMPLATE = "Hi {name}, ref {call_sid}, stream {stream_sid}, missing {absent}."


def test_server_owned_call_ids_never_reach_the_prompt():
    context = {"recipient_data": {"name": "Rahul", "call_sid": "CA123", "stream_sid": "ST9"}}
    rendered = update_prompt_with_context(TEMPLATE, context)
    assert "CA123" not in rendered
    assert "ST9" not in rendered
    assert rendered == "Hi Rahul, ref , stream , missing ."


def test_an_unknown_placeholder_renders_empty():
    """Leaving "{absent}" in place would have the agent read the braces aloud."""
    rendered = update_prompt_with_context("Value: {absent}", {"recipient_data": {"other": "x"}})
    assert rendered == "Value: "


def test_no_context_still_clears_placeholders():
    assert update_prompt_with_context("Hi {name}", None) == "Hi "
    assert update_prompt_with_context("Hi {name}", {"no_recipient_data": True}) == "Hi "


def test_a_prompt_without_placeholders_is_untouched():
    assert update_prompt_with_context("No placeholders here", {"recipient_data": {"a": 1}}) == "No placeholders here"
