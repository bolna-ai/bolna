"""reasoning_effort is checked when the agent is configured, not once per turn.

An effort the model does not accept is rejected by the provider on every request, so the agent
answers nothing for the whole call. The check has to see through a provider-prefixed deployment
name, and must stay out of the way for models it knows nothing about.
"""

import pytest

from bolna.models import validate_reasoning_effort_for_model


def test_a_supported_effort_is_accepted():
    validate_reasoning_effort_for_model("gpt-5", "minimal")


def test_an_unsupported_effort_is_rejected():
    """ "minimal" is a gpt-5-family value that gpt-5.1 dropped."""
    with pytest.raises(ValueError, match="minimal"):
        validate_reasoning_effort_for_model("gpt-5.1", "minimal")


def test_a_provider_prefixed_deployment_is_still_checked():
    """Azure deployment names arrive prefixed; reading the raw name would skip the check."""
    with pytest.raises(ValueError):
        validate_reasoning_effort_for_model("azure/gpt-5.1", "minimal")


def test_a_non_gpt_model_is_left_alone():
    validate_reasoning_effort_for_model("claude-sonnet-4", "minimal")


def test_an_unknown_gpt_model_is_left_alone():
    """A model newer than this map must not be blocked by it."""
    validate_reasoning_effort_for_model("gpt-5-not-released-yet", "minimal")
