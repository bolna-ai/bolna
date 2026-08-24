"""Tests for render_prompt — the JSON-safe replacement for str.format_map.

The bug being fixed: format_map parsed every brace, so one JSON literal in a prompt
raised, and because callers swallowed the exception EVERY variable in that prompt
silently went unsubstituted.
"""

from bolna.helpers.utils import (
    parse_json_container,
    render_prompt,
    render_variable_value,
    resolve_variable_path,
    update_prompt_with_context,
)

DATA = {
    "name": "Puneet",
    "score": 720,
    "prior": {"loans": [{"amount": 5000, "status": "closed"}], "score": 720},
    "empty": "",
    "none_value": None,
}


class TestLegacySingleBrace:
    """Existing prompts must render exactly as before."""

    def test_known_variable(self):
        assert render_prompt("Hello {name}, welcome.", DATA) == "Hello Puneet, welcome."

    def test_unknown_variable_renders_empty(self):
        assert render_prompt("Hi {nonexistent}!", DATA) == "Hi !"

    def test_numeric_value(self):
        assert render_prompt("Score {score}", DATA) == "Score 720"

    def test_empty_string_value(self):
        assert render_prompt("[{empty}]", DATA) == "[]"

    def test_legacy_bracket_nesting_still_works(self):
        assert render_prompt("{prior[loans][0][amount]}", DATA) == "5000"

    def test_double_braced_json_keeps_both_braces(self):
        # Accepted regression: no blanket unescape, since nested JSON ends in }} and would lose a brace.
        assert render_prompt('{{"status": "ok"}}', DATA) == '{{"status": "ok"}}'

    def test_double_braced_identifier_still_unescapes(self):
        assert render_prompt("literal {{Unknown}} here", DATA) == "literal {Unknown} here"


class TestJsonInPromptNoLongerBreaks:
    """The reported bug. Previously each of these raised and lost every variable."""

    def test_json_literal_and_variable_together(self):
        out = render_prompt('Hello {name}. Respond as: {"status": "ok", "score": 1}', DATA)
        assert out == 'Hello Puneet. Respond as: {"status": "ok", "score": 1}'

    def test_json_only_is_untouched(self):
        assert render_prompt('Respond as: {"status": "ok"}', DATA) == 'Respond as: {"status": "ok"}'

    def test_nested_json_literal(self):
        template = 'Schema: {"user": {"name": "x", "tags": ["a"]}} and {name}'
        assert render_prompt(template, DATA) == 'Schema: {"user": {"name": "x", "tags": ["a"]}} and Puneet'

    def test_empty_braces_untouched(self):
        assert render_prompt("Use {} for blanks, {name}", DATA) == "Use {} for blanks, Puneet"

    def test_pseudo_json_type_annotation_stays_literal(self):
        # Looks like {name:spec} but the spec is nonsense — must never be deleted.
        template = "Return {name: string, age: number}"
        assert render_prompt(template, DATA) == template

    def test_never_raises_on_pathological_input(self):
        for bad in ["{", "}", "{{{", "}}}", "{a{b}c}", '{"unclosed": ']:
            render_prompt(bad, DATA)


class TestLegacyFormatSpecs:
    """{price:.2f} worked under format_map and must keep working."""

    def test_float_precision(self):
        assert render_prompt("{price:.2f}", {"price": 12.5}) == "12.50"

    def test_padding(self):
        assert render_prompt("[{score:>6}]", DATA) == "[   720]"

    def test_unresolved_with_spec_stays_literal(self):
        assert render_prompt("{unknown:.2f}", DATA) == "{unknown:.2f}"

    def test_bad_spec_for_value_stays_literal(self):
        assert render_prompt("{name:.2f}", DATA) == "{name:.2f}"

    def test_spec_not_applied_during_partial_fill(self):
        # Seeding must leave the spec for the live render, not bake in a seed-time value.
        assert render_prompt("{price:.2f}", {"price": 12.5}, missing=None) == "{price:.2f}"

    def test_spec_failure_only_loses_its_own_token(self):
        class Hostile:
            def __format__(self, spec):
                raise KeyError("boom")  # not ValueError/TypeError

        out = render_prompt("Hi {name}, {x:>4}", {"name": "Puneet", "x": Hostile()})
        assert out == "Hi Puneet, {x:>4}"  # sibling variable survives

    def test_spec_not_applied_to_double_brace(self):
        # New syntax has no legacy specs; a colon means it is not a variable token.
        assert render_prompt("{{name:.2f}}", DATA) == "{{name:.2f}}"


class TestDoubleBraceSyntax:
    def test_known_variable(self):
        assert render_prompt("Hello {{name}}!", DATA) == "Hello Puneet!"

    def test_inner_whitespace_allowed(self):
        assert render_prompt("Hello {{ name }}!", DATA) == "Hello Puneet!"

    def test_both_syntaxes_in_one_prompt(self):
        assert render_prompt("{name} and {{name}}", DATA) == "Puneet and Puneet"

    def test_unresolved_falls_back_to_legacy_unescape(self):
        # Guarantees no existing {{X}} escape changes output when X isn't a variable.
        assert render_prompt("literal {{Unknown}} here", DATA) == "literal {Unknown} here"

    def test_dot_notation(self):
        assert render_prompt("{{prior.score}}", DATA) == "720"

    def test_deep_dot_notation_with_list_index(self):
        assert render_prompt("{{prior.loans.0.amount}}", DATA) == "5000"
        assert render_prompt("{{prior.loans.0.status}}", DATA) == "closed"

    def test_missing_nested_path_renders_unescaped(self):
        assert render_prompt("{{prior.missing.deep}}", DATA) == "{prior.missing.deep}"

    def test_out_of_range_index(self):
        assert render_prompt("{{prior.loans.5.amount}}", DATA) == "{prior.loans.5.amount}"


class TestObjectValues:
    """JSON for the new syntax only. Prod recipient_data already carries object-valued
    variables (product_details, items, cart_data_json, nearest_store), so the legacy
    {path} syntax must keep Python str() or live output would change."""

    def test_double_brace_dict_is_json(self):
        out = render_prompt("{{prior}}", DATA)
        assert out == '{"loans": [{"amount": 5000, "status": "closed"}], "score": 720}'
        assert "'" not in out

    def test_double_brace_list_is_json(self):
        assert render_prompt("{{prior.loans}}", DATA) == '[{"amount": 5000, "status": "closed"}]'

    def test_single_brace_dict_keeps_legacy_repr(self):
        prod_like = {"product_details": {"name": "Shirt", "price": 100, "in_stock": True}}
        assert render_prompt("{product_details}", prod_like) == str(prod_like["product_details"])

    def test_single_brace_list_keeps_legacy_repr(self):
        prod_like = {"items": [{"sku": "A1"}, {"sku": "B2"}]}
        assert render_prompt("{items}", prod_like) == str(prod_like["items"])

    def test_non_serialisable_falls_back_to_str(self):
        assert render_variable_value({"when": object()}, as_json=True).startswith("{")


class TestMissingMarker:
    def test_custom_marker(self):
        assert render_prompt("{absent}", DATA, missing="NULL") == "NULL"

    def test_partial_fill_leaves_unknowns_untouched(self):
        template = 'Hi {name}, {other} and {{third}} plus {"a": 1}'
        assert render_prompt(template, {"name": "P"}, missing=None) == 'Hi P, {other} and {{third}} plus {"a": 1}'

    def test_partial_fill_then_live_render(self):
        # Seeding runs before the live render; JSON must survive both passes intact.
        once = render_prompt('{"a": 1} {name} {callee}', {"name": "P"}, missing=None)
        assert once == '{"a": 1} P {callee}'
        assert render_prompt(once, {"callee": "Sam"}) == '{"a": 1} P Sam'


class TestGuards:
    def test_none_template(self):
        assert render_prompt(None, DATA) is None

    def test_empty_template(self):
        assert render_prompt("", DATA) == ""

    def test_non_dict_data(self):
        assert render_prompt("{name}", None) == ""

    def test_non_string_template_passthrough(self):
        assert render_prompt(123, DATA) == 123


class TestResolveVariablePath:
    def test_found_and_missing(self):
        assert resolve_variable_path("prior.score", DATA) == (True, 720)
        assert resolve_variable_path("prior.nope", DATA) == (False, None)

    def test_scalar_cannot_be_traversed(self):
        assert resolve_variable_path("name.deeper", DATA) == (False, None)

    def test_negative_index(self):
        assert resolve_variable_path("prior.loans.-1.amount", DATA) == (True, 5000)

    def test_non_numeric_index_into_list(self):
        assert resolve_variable_path("prior.loans.key", DATA) == (False, None)


class TestUpdatePromptWithContext:
    def test_substitutes_from_recipient_data(self):
        ctx = {"recipient_data": {"name": "Puneet"}}
        assert update_prompt_with_context("Hi {name}", ctx) == "Hi Puneet"

    def test_server_owned_ids_never_leak(self):
        ctx = {"recipient_data": {"call_sid": "CA123", "stream_sid": "ST456", "name": "P"}}
        out = update_prompt_with_context("{call_sid}|{stream_sid}|{name}", ctx)
        assert out == "||P"

    def test_server_owned_ids_never_leak_via_double_brace(self):
        ctx = {"recipient_data": {"call_sid": "CA123"}}
        assert "CA123" not in update_prompt_with_context("{{call_sid}}", ctx)

    def test_no_context(self):
        assert update_prompt_with_context("Hi {name}", None) == "Hi "

    def test_json_survives_with_no_context(self):
        assert update_prompt_with_context('{"a": 1}', None) == '{"a": 1}'


class TestStringifiedJsonVariables:
    """Telephony /call and API callers commonly send a nested variable as a JSON *string* —
    only the web-call panel parses client-side. Nested paths must still resolve, WITHOUT
    changing what a bare {var} renders.
    """

    STR_DATA = {
        "prior": '{"score": 720, "loans": [{"amount": 5000, "status": "closed"}]}',
        "product_details": '{"name": "Shirt", "price": 100}',
        "items": '[{"sku": "A1"}, {"sku": "B2"}]',
        "price": "12.5",
        "note": "not json at all",
    }

    def test_nested_path_walks_into_a_json_string(self):
        assert render_prompt("{{prior.score}}", self.STR_DATA) == "720"
        assert render_prompt("{{prior.loans.0.amount}}", self.STR_DATA) == "5000"
        assert render_prompt("{{prior.loans.0.status}}", self.STR_DATA) == "closed"

    def test_legacy_bracket_path_also_walks_in(self):
        assert render_prompt("{prior[loans][0][amount]}", self.STR_DATA) == "5000"

    def test_json_array_string_indexes(self):
        assert render_prompt("{{items.1.sku}}", self.STR_DATA) == "B2"

    def test_bare_variable_is_NOT_parsed(self):
        # No path walked -> no parsing -> renders the exact string it always did.
        for template in ("{product_details}", "{{product_details}}"):
            assert render_prompt(template, self.STR_DATA) == self.STR_DATA["product_details"]

    def test_numeric_string_is_never_coerced(self):
        assert render_prompt("{price}", self.STR_DATA) == "12.5"

    def test_non_json_string_path_still_fails_cleanly(self):
        assert render_prompt("{{note.deeper}}", self.STR_DATA) == "{note.deeper}"

    def test_malformed_json_string_path_fails_cleanly(self):
        assert render_prompt("{{broken.x}}", {"broken": '{"unclosed": '}) == "{broken.x}"


class TestParseJsonContainer:
    def test_object_and_array_are_parsed(self):
        assert parse_json_container('{"a": 1}') == {"a": 1}
        assert parse_json_container("[1, 2]") == [1, 2]

    def test_scalars_and_non_strings_pass_through(self):
        for value in ("720", "hello", '"quoted"', 42, None, {"a": 1}):
            assert parse_json_container(value) == value

    def test_malformed_json_passes_through(self):
        assert parse_json_container('{"unclosed": ') == '{"unclosed": '


class TestHyphenatedKeys:
    """Real payloads key objects by date ("31-03-2024"), so dot segments allow hyphens.
    Hyphens are permitted ONLY after a dot — the leading identifier still cannot contain one,
    which is what keeps {price-list} a non-match (0/76,907 prod agents use a dotted-hyphen token).
    """

    DATA = {"cc": {"history": [{"31-03-2024": 628}]}, "price": 10, "name": "A"}

    def test_hyphenated_key_via_dots(self):
        assert render_prompt("{{cc.history.0.31-03-2024}}", self.DATA) == "628"

    def test_hyphenated_key_via_brackets_still_works(self):
        assert render_prompt("{cc[history][0][31-03-2024]}", self.DATA) == "628"

    def test_leading_identifier_hyphen_is_still_not_a_variable(self):
        assert render_prompt("{price-list}", self.DATA) == "{price-list}"

    def test_pseudo_json_with_hyphen_key_untouched(self):
        template = "Return {name: string, age-x: number}"
        assert render_prompt(template, self.DATA) == template

    def test_json_literal_with_hyphen_key_untouched(self):
        assert render_prompt('{"a-b": 1}', self.DATA) == '{"a-b": 1}'


class TestUnresolvedNestedPaths:
    """A failed nested path leaves the literal token in the prompt rather than dropping content.

    Reproduces prod run aafd4625, where `list` arrived as the Python repr "['apple', 'bananna']"
    (single quotes) and so could not be parsed as JSON. Diagnostic logging for this used to live
    here; it was removed as too noisy — a graph agent re-renders every turn, so one misconfigured
    agent emitted ~1,500 INFO lines per call, all repeats of the same paths.
    """

    def test_python_repr_container_stays_literal(self):
        assert render_prompt("{{list.0}}", {"list": "['apple', 'bananna']"}) == "{list.0}"

    def test_json_string_container_resolves(self):
        assert render_prompt("{{list.0}}", {"list": '["apple", "bananna"]'}) == "apple"

    def test_flat_miss_renders_missing_and_partial_fill_keeps_the_token(self):
        assert render_prompt("{nope}", {"a": 1}) == ""
        assert render_prompt("{{a.b}}", {}, missing=None) == "{{a.b}}"
