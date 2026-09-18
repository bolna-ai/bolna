import pytest

from bolna.helpers.asr_keywords import (
    format_weighted_keywords,
    keyword_entries,
    keyword_terms,
    parse_keywords,
)


@pytest.mark.parametrize(
    "keywords,expected",
    [
        ("bolna", [("bolna", 1.0)]),
        ("bolna:2", [("bolna", 2.0)]),
        ("bolna:2.5", [("bolna", 2.5)]),
        ("bolna, account number:3", [("bolna", 1.0), ("account number", 3.0)]),
        # A colon that is not a weight belongs to the term.
        ("3:30 pm", [("3:30 pm", 1.0)]),
        ("acme:corp:2", [("acme:corp", 2.0)]),
        ("", []),
        (None, []),
        (" , ,bolna, ", [("bolna", 1.0)]),
        (":2", []),
    ],
)
def test_parse_keywords(keywords, expected):
    assert parse_keywords(keywords) == expected


def test_keyword_terms_drops_weights():
    assert keyword_terms("bolna:5,account number:2,plivo") == ["bolna", "account number", "plivo"]


def test_keyword_entries_preserves_the_agents_own_syntax():
    assert keyword_entries(" bolna:5 , plivo ") == ["bolna:5", "plivo"]


def test_format_weighted_keywords_makes_every_weight_explicit():
    assert format_weighted_keywords(parse_keywords("bolna:2,plivo")) == "bolna:2,plivo:1"
