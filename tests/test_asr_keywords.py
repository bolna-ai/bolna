import pytest

from bolna.helpers.asr_keywords import keyword_entries, keyword_terms


@pytest.mark.parametrize(
    "keywords,expected",
    [
        ("bolna", ["bolna"]),
        ("bolna:2", ["bolna"]),
        ("bolna:2.5", ["bolna"]),
        ("bolna, account number:3", ["bolna", "account number"]),
        # A colon that is not a weight belongs to the term.
        ("3:30 pm", ["3:30 pm"]),
        ("acme:corp:2", ["acme:corp"]),
        ("", []),
        (None, []),
        (" , ,bolna, ", ["bolna"]),
        (":2", []),
    ],
)
def test_keyword_terms_drops_weights(keywords, expected):
    assert keyword_terms(keywords) == expected


def test_keyword_entries_preserves_the_agents_own_syntax():
    assert keyword_entries(" bolna:5 , plivo ") == ["bolna:5", "plivo"]
    assert keyword_entries(None) == []
