"""Pydantic validation for IntegrationConfig + ToolsConfig.integrations."""

import pytest
from pydantic import ValidationError

from bolna.models import IntegrationConfig, SlackIntegrationConfig, ToolsConfig


def test_slack_config_with_explicit_url():
    cfg = IntegrationConfig(
        provider="slack",
        provider_config={"webhook_url": "https://hooks.slack.com/services/AAA/BBB/CCC"},
    )
    assert cfg.provider == "slack"
    assert isinstance(cfg.provider_config, SlackIntegrationConfig)
    assert cfg.provider_config.webhook_url == "https://hooks.slack.com/services/AAA/BBB/CCC"


def test_slack_config_with_no_url_is_valid():
    # url can be omitted at parse time; runner falls back to env
    cfg = IntegrationConfig(provider="slack", provider_config={})
    assert cfg.provider_config.webhook_url is None


def test_unknown_provider_rejected():
    with pytest.raises(ValidationError):
        IntegrationConfig(provider="discord", provider_config={})


def test_tools_config_accepts_integrations_list():
    tc = ToolsConfig(
        integrations=[
            {
                "provider": "slack",
                "provider_config": {"webhook_url": "https://hooks.slack.com/services/x/y/z"},
            }
        ]
    )
    assert tc.integrations is not None
    assert len(tc.integrations) == 1
    assert tc.integrations[0].provider == "slack"


def test_tools_config_integrations_optional():
    tc = ToolsConfig()
    assert tc.integrations is None


def test_tools_config_empty_integrations_list():
    tc = ToolsConfig(integrations=[])
    assert tc.integrations == []
