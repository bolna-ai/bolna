"""Per-key definitions on graph edge parameters: what the routing model sees on each key."""

import pytest
from pydantic import ValidationError

from bolna.models import GraphEdge
from tests.test_router_nodes import _base_config, _make_agent


def _agent_with_edge(parameters):
    nodes = [
        {
            "id": "ask",
            "prompt": "Ask availability.",
            "edges": [
                {
                    "to_node_id": "done",
                    "condition": "The candidate replied.",
                    "function_name": "record_availability",
                    "parameters": parameters,
                }
            ],
        },
        {"id": "done", "prompt": "Done.", "edges": []},
    ]
    return _make_agent(_base_config(nodes, "ask"))


def _properties(agent):
    edge = agent.get_node_by_id("ask")["edges"][0]
    tools = agent._build_transition_tools_for_edges([edge])
    return tools[0]["function"]["parameters"]["properties"]


class TestEdgeParameterSchema:
    def test_definition_and_allowed_values_reach_the_key(self):
        agent = _agent_with_edge(
            {
                "availability": {
                    "type": "string",
                    "description": "Yes if they can talk now, No if busy or driving.",
                    "allowed_values": ["Yes", "No", "Busy"],
                }
            }
        )

        assert _properties(agent)["availability"] == {
            "type": "string",
            "description": "Yes if they can talk now, No if busy or driving.",
            "enum": ["Yes", "No", "Busy"],
        }

    def test_bare_type_keeps_the_generic_description(self):
        agent = _agent_with_edge({"availability": "string", "attempts": "number"})

        props = _properties(agent)
        assert props["availability"] == {"type": "string", "description": "The availability provided by the user"}
        assert props["attempts"] == {"type": "number", "description": "The attempts provided by the user"}

    def test_definition_without_allowed_values_adds_no_enum(self):
        agent = _agent_with_edge({"years": {"type": "number", "description": "Total years of experience."}})

        assert _properties(agent)["years"] == {"type": "number", "description": "Total years of experience."}

    def test_mixed_forms_are_all_required(self):
        agent = _agent_with_edge({"a": "string", "b": {"type": "string", "description": "B."}})
        edge = agent.get_node_by_id("ask")["edges"][0]

        required = agent._build_transition_tools_for_edges([edge])[0]["function"]["parameters"]["required"]
        assert {"a", "b"} <= set(required)


class TestEdgeParameterValidation:
    def test_accepts_both_forms(self):
        edge = GraphEdge(
            to_node_id="done",
            parameters={"a": "string", "b": {"description": "B.", "allowed_values": ["x", "y"]}},
        )
        assert edge.parameters["a"] == "string"
        assert edge.parameters["b"].type == "string"
        assert edge.parameters["b"].allowed_values == ["x", "y"]

    def test_rejects_allowed_values_on_a_non_string_parameter_with_one_error(self):
        with pytest.raises(ValidationError, match="only supported on string parameters") as exc:
            GraphEdge(to_node_id="done", parameters={"years": {"type": "number", "allowed_values": ["1", "2"]}})
        assert exc.value.error_count() == 1

    def test_rejects_an_empty_allowed_values_list(self):
        with pytest.raises(ValidationError, match="at least one value"):
            GraphEdge(to_node_id="done", parameters={"a": {"allowed_values": []}})
