"""A tool argument holding a quote, backslash or newline still builds a valid request body.

The legacy param template quotes each placeholder ("%(name)s"), so a string value lands
inside a JSON string and must be escaped for it, or json.loads fails and the API is never called.
"""

import pytest

from bolna.helpers.function_calling_helpers import prepare_api_request

TEMPLATE = {"customer_name": "%(customer_name)s", "notes": "%(notes)s", "channel": "voice"}


@pytest.mark.parametrize(
    "notes",
    [
        'Customer said "call me after 5"',
        "Flat 4B\nMG Road",
        "path C:\\Users\\asha",
        "tab\there",
    ],
)
def test_special_characters_survive_substitution(notes):
    prepared = prepare_api_request(TEMPLATE, None, None, customer_name="Asha", notes=notes)

    assert prepared["api_params"] == {"customer_name": "Asha", "notes": notes, "channel": "voice"}


def test_plain_values_render_unchanged():
    prepared = prepare_api_request(TEMPLATE, None, None, customer_name="आशा", notes="after 5")

    assert prepared["request_body"] == '{"customer_name": "आशा", "notes": "after 5", "channel": "voice"}'


def test_lists_and_numbers_keep_their_json_form():
    template = '{"items": %(items)s, "count": %(count)s}'

    prepared = prepare_api_request(template, None, None, items=["a", "b"], count=2)

    assert prepared["api_params"] == {"items": ["a", "b"], "count": 2}
