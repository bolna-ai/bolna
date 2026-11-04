"""Tool params reach a GET query or a form body as JSON values, not Python reprs."""

from urllib.parse import parse_qsl

import pytest

from bolna.helpers.function_calling_helpers import build_get_url, normalize_for_form

CASES = [
    (True, "true"),
    (False, "false"),
    (None, "null"),
    (["a", "b"], '["a", "b"]'),
    ({"k": 1}, '{"k": 1}'),
    (2, "2"),
    (2.5, "2.5"),
    ("Asha", "Asha"),
]


@pytest.mark.parametrize("value, wire", CASES)
def test_get_query_value(value, wire):
    url = build_get_url("https://api.example.com/slots", {"v": value})
    assert dict(parse_qsl(url.raw_query_string)) == {"v": wire}


@pytest.mark.parametrize("value, wire", CASES)
def test_form_value(value, wire):
    assert normalize_for_form({"v": value}) == {"v": wire}
