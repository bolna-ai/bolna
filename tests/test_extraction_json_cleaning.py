"""Unwrapping the model's JSON before post-call extraction parses it."""

from bolna.helpers.utils import clean_json_string


def test_a_fenced_json_block_is_unwrapped():
    assert clean_json_string('```json\n{"a": 1}\n```') == '{"a": 1}'


def test_a_bare_json_string_is_untouched():
    assert clean_json_string('{"a": 1}') == '{"a": 1}'


def test_the_structure_header_is_removed():
    assert clean_json_string('###JSON Structure\n{"a": 1}') == '{"a": 1}'


def test_a_non_string_passes_straight_through():
    """The extraction path may already hold a parsed dict."""
    assert clean_json_string({"a": 1}) == {"a": 1}
