"""Graph agents: the language directive reaches the node prompt whenever the language is known,
not only when the node happens to carry an example for that language.

_get_prompt_with_example is the only language channel a graph agent has, since _build_messages strips
system messages from history, so the TaskManager-side pin never reaches it. Nodes may also carry
an explicit "examples": null, so the lookup has to tolerate None rather than crash generate().
"""

from unittest.mock import MagicMock

from bolna.agent_types.graph_agent import GraphAgent

BUILD = GraphAgent._get_prompt_with_example


def _prompt(node, lang):
    return BUILD(MagicMock(), node, lang)


def test_examples_null_does_not_crash_and_still_gets_the_directive():
    # Agent JSONs carry "examples": null explicitly, so the lookup must tolerate None.
    out = _prompt({"prompt": "P", "examples": None}, "en")
    assert "LANGUAGE GUIDELINES" in out and "English" in out


def test_examples_null_without_language_is_the_bare_prompt():
    assert _prompt({"prompt": "P", "examples": None}, None) == "P"


def test_directive_without_any_examples():
    # No examples on the node, language known: the directive must still appear.
    out = _prompt({"prompt": "आप सहायक हैं।"}, "en")
    assert "LANGUAGE GUIDELINES" in out
    assert "English" in out and "'en'" in out
    assert out.startswith("आप सहायक हैं।")


def test_directive_when_examples_lack_the_detected_language():
    out = _prompt({"prompt": "P", "examples": {"hi": "नमस्ते"}}, "en")
    assert "LANGUAGE GUIDELINES" in out and "English" in out
    assert "नमस्ते" not in out  # wrong-language example must not be attached


def test_directive_with_matching_example_keeps_the_example():
    out = _prompt({"prompt": "P", "examples": {"te": "నమస్తే"}}, "te")
    assert "LANGUAGE GUIDELINES" in out
    assert "Telugu" in out
    assert 'Example response: "నమస్తే"' in out


def test_no_language_keeps_the_all_examples_behavior():
    out = _prompt({"prompt": "P", "examples": {"hi": "नमस्ते", "en": "Hello"}}, None)
    assert "LANGUAGE GUIDELINES" not in out
    assert 'HI: "नमस्ते"' in out and 'EN: "Hello"' in out


def test_no_language_no_examples_is_the_bare_prompt():
    assert _prompt({"prompt": "P"}, None) == "P"


def test_unknown_code_falls_back_to_the_raw_code():
    # A label missing from LANGUAGE_NAMES must not crash or drop the directive.
    out = _prompt({"prompt": "P"}, "xx")
    assert "LANGUAGE GUIDELINES" in out and "'xx'" in out
