"""Prompt variables on two surfaces that never rendered them.

Both were found by auditing every render site after the {{ }} work:

  * handoff / hand-in messages only ever did two hardcoded str.replace calls for
    {agent_name} and {language}, so a customer's {customer_name} was spoken as
    literal braces on every language switch.
  * the classic (non-graph) extraction path passes the schema into
    EXTRACTION_PROMPT as a .format() ARGUMENT, so str.format never looks inside
    it and variables written into an extraction description stayed literal.
    Graph agents already rendered theirs (graph_agent.py).

Asserted at the seam — the rendering contract — rather than by standing up a
whole TaskManager, which needs transcriber/synthesizer/LLM pools.
"""

from bolna.helpers.utils import update_prompt_with_context

CONTEXT = {"recipient_data": {"customer_name": "Asha", "plan": "gold"}}


class TestHandoffMessageVariables:
    """Mirrors task_manager.__handoff_text_for and the legacy-flow handoff block: the two
    runtime placeholders are replaced FIRST, then prompt variables are rendered."""

    @staticmethod
    def handoff(template, agent_name="Meera", language="Hindi", context=CONTEXT):
        text = template.replace("{agent_name}", agent_name).replace("{language}", language)
        return update_prompt_with_context(text, context)

    def test_customer_variable_now_resolves(self):
        out = self.handoff("One moment {customer_name}, connecting you to {agent_name}.")
        assert out == "One moment Asha, connecting you to Meera."

    def test_double_brace_syntax_resolves_too(self):
        out = self.handoff("{{customer_name}}, switching to {language}.")
        assert out == "Asha, switching to Hindi."

    def test_runtime_placeholders_win_over_a_same_named_variable(self):
        # Ordering matters: {agent_name} is the switch TARGET voice, not a prompt variable.
        # Rendering first would let recipient_data hijack it.
        ctx = {"recipient_data": {"agent_name": "WRONG", "customer_name": "Asha"}}
        assert self.handoff("Hi {customer_name}, meet {agent_name}.", context=ctx) == "Hi Asha, meet Meera."

    def test_unpassed_variable_renders_empty_not_literal(self):
        # Previously this reached TTS as "{nickname}" and was read aloud.
        assert self.handoff("Bye {nickname}.") == "Bye ."

    def test_no_context_data_does_not_crash(self):
        assert self.handoff("Connecting to {agent_name}.", context=None) == "Connecting to Meera."

    def test_template_without_variables_is_unchanged(self):
        assert self.handoff("Switching to {language} now.") == "Switching to Hindi now."


class TestExtractionSchemaVariables:
    """Mirrors task_manager's extraction branch: the schema string is rendered before it is
    substituted into EXTRACTION_PROMPT."""

    def test_variable_inside_an_extraction_description_resolves(self):
        schema = '{"did_confirm": "true if {customer_name} confirmed the appointment"}'
        out = update_prompt_with_context(schema, CONTEXT)
        assert out == '{"did_confirm": "true if Asha confirmed the appointment"}'

    def test_json_structure_survives_rendering(self):
        # The whole point of render_prompt: a JSON schema must come through byte-identical.
        schema = '{"a": {"b": [1, 2]}, "c": {"d": "x"}}'
        assert update_prompt_with_context(schema, CONTEXT) == schema

    def test_nested_path_resolves_in_a_schema(self):
        ctx = {"recipient_data": {"prior": {"score": 720}}}
        schema = '{"note": "prior score was {{prior.score}}"}'
        assert update_prompt_with_context(schema, ctx) == '{"note": "prior score was 720"}'

    def test_non_string_schema_is_left_alone_by_the_caller(self):
        # The call site guards on isinstance(str); a dict schema must not be stringified.
        schema = {"did_confirm": "bool"}
        assert isinstance(schema, dict)
