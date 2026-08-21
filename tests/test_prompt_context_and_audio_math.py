"""Prompt variable substitution, and the byte maths that timing decisions rest on.

A prompt template is filled from caller-supplied recipient data, so the server-owned telephony
ids must render empty rather than handing the model a real call_sid, and an unknown placeholder
must render empty rather than leaving braces in what the agent says.

calculate_audio_duration converts a buffer size to seconds, and telephony mulaw carries one byte
per sample where PCM carries two — getting that wrong misjudges when the agent has finished
speaking.
"""

from bolna.helpers.utils import calculate_audio_duration, clean_json_string, update_prompt_with_context

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


def test_pcm_duration_counts_two_bytes_per_sample():
    assert calculate_audio_duration(32000, 16000) == 1.0


def test_mulaw_duration_counts_one_byte_per_sample():
    """Halving this would have the agent think it finished speaking twice as fast."""
    assert calculate_audio_duration(8000, 8000, format="mulaw") == 1.0


def test_both_mulaw_spellings_agree():
    assert calculate_audio_duration(8000, 8000, format="ulaw") == calculate_audio_duration(8000, 8000, format="mulaw")


def test_channels_and_bit_depth_are_honoured():
    assert calculate_audio_duration(32000, 8000, channels=2) == 1.0
    assert calculate_audio_duration(8000, 8000, bit_depth=8) == 1.0


def test_a_fenced_json_block_is_unwrapped():
    assert clean_json_string('```json\n{"a": 1}\n```') == '{"a": 1}'


def test_a_bare_json_string_is_untouched():
    assert clean_json_string('{"a": 1}') == '{"a": 1}'


def test_the_structure_header_is_removed():
    assert clean_json_string('###JSON Structure\n{"a": 1}') == '{"a": 1}'


def test_a_non_string_passes_straight_through():
    """The extraction path may already hold a parsed dict."""
    assert clean_json_string({"a": 1}) == {"a": 1}
