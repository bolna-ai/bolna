"""A call whose context_data carries "recipient_data": null still builds its system prompt,
as update_prompt_with_context and enrich_context_with_time_variables already allow."""

import pytz

from bolna.helpers.utils import enrich_context_with_time_variables, structure_system_prompt, update_prompt_with_context

TZ = pytz.timezone("Asia/Kolkata")


def test_sibling_helpers_already_accept_null_recipient_data():
    context_data = {"recipient_data": None}
    enrich_context_with_time_variables(context_data, TZ)
    assert update_prompt_with_context("Hello {name}", context_data) == "Hello "


def test_structure_system_prompt_accepts_null_recipient_data():
    prompt = structure_system_prompt(
        "You are a helpful agent. Caller: {name}",
        run_id="run-1",
        assistant_id="agent-1",
        call_sid="CA1",
        context_data={"recipient_data": None},
        timezone=TZ,
    )
    assert prompt.startswith("You are a helpful agent. Caller: ")
    assert 'agent_id is "agent-1"' in prompt


def test_primary_language_data_still_merges_into_recipient_data():
    context_data = {"recipient_data": {"name": "Asha", "hi": {"name": "आशा"}}}
    prompt = structure_system_prompt(
        "Caller: {name}",
        run_id="run-1",
        assistant_id="agent-1",
        call_sid="CA1",
        context_data=context_data,
        timezone=TZ,
        primary_language="hi",
    )
    assert prompt.startswith("Caller: आशा")


def test_null_primary_language_data_is_ignored():
    prompt = structure_system_prompt(
        "Caller: {name}",
        run_id="run-1",
        assistant_id="agent-1",
        call_sid="CA1",
        context_data={"recipient_data": {"name": "Asha", "hi": None}},
        timezone=TZ,
        primary_language="hi",
    )
    assert prompt.startswith("Caller: Asha")
