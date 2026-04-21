"""SlackIntegration: from_config env fallback, block builder shape, execute path."""

from unittest.mock import AsyncMock, patch

import pytest

from bolna.integrations.base import PostCallContext
from bolna.integrations.slack import SlackIntegration, _build_slack_blocks, _format_duration
from bolna.models import IntegrationConfig


def _make_config(webhook_url=None):
    return IntegrationConfig(provider="slack", provider_config={"webhook_url": webhook_url})


def _ctx(**overrides):
    base = dict(
        agent_name="loan-collections",
        run_id="r-abc",
        call_sid="CA123",
        duration_seconds=92.5,
        hangup_reason="llm_prompted_hangup",
        summary="customer agreed to repay by friday.",
        extracted_data={"customer_name": "Ananya", "promise_to_pay": "2026-04-25"},
        recording_url="https://s3.example/rec.wav",
    )
    base.update(overrides)
    return PostCallContext(**base)


def test_from_config_uses_explicit_url():
    integ = SlackIntegration.from_config(_make_config("https://hooks.slack.com/services/x/y/z"))
    assert integ.webhook_url == "https://hooks.slack.com/services/x/y/z"


def test_from_config_falls_back_to_env(monkeypatch):
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.com/env-url")
    integ = SlackIntegration.from_config(_make_config())
    assert integ.webhook_url == "https://hooks.slack.com/env-url"


def test_from_config_raises_without_url(monkeypatch):
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
    with pytest.raises(ValueError):
        SlackIntegration.from_config(_make_config())


def test_format_duration():
    assert _format_duration(None) == "n/a"
    assert _format_duration(0) == "n/a"
    assert _format_duration(45) == "45s"
    assert _format_duration(92.5) == "1m 32s"
    assert _format_duration(3600) == "60m 0s"


def test_blocks_full_context():
    blocks = _build_slack_blocks(_ctx())
    types = [b["type"] for b in blocks]
    assert types[0] == "header"
    assert "loan-collections" in blocks[0]["text"]["text"]
    section = blocks[1]["text"]["text"]
    assert "CA123" in section
    assert "1m 32s" in section
    assert "llm_prompted_hangup" in section
    # summary + extracted_data each preceded by a divider
    assert types.count("divider") == 2
    assert any(b.get("type") == "section" and "summary" in b.get("text", {}).get("text", "") for b in blocks)
    assert any(b.get("type") == "actions" for b in blocks)


def test_blocks_minimal_context():
    blocks = _build_slack_blocks(PostCallContext(agent_name="a", run_id="r-1"))
    types = [b["type"] for b in blocks]
    assert types == ["header", "section"]


def test_blocks_omits_recording_when_absent():
    blocks = _build_slack_blocks(_ctx(recording_url=None))
    assert not any(b["type"] == "actions" for b in blocks)


def test_blocks_caps_extracted_fields_at_ten():
    big = {f"field_{i}": str(i) for i in range(20)}
    blocks = _build_slack_blocks(_ctx(extracted_data=big))
    fields_block = next(b for b in blocks if b.get("fields"))
    assert len(fields_block["fields"]) == 10


def test_blocks_truncate_long_summary():
    long_summary = "x" * 5000
    blocks = _build_slack_blocks(_ctx(summary=long_summary))
    summary_block = next(b for b in blocks if "summary" in b.get("text", {}).get("text", ""))
    assert len(summary_block["text"]["text"]) < 3000


@pytest.mark.asyncio
async def test_execute_posts_to_webhook():
    integ = SlackIntegration(webhook_url="https://hooks.slack.com/services/x/y/z")
    with patch("bolna.integrations.slack._post_with_retry", new=AsyncMock()) as mocked:
        await integ.execute(_ctx())
    mocked.assert_awaited_once()
    args, _ = mocked.call_args
    assert args[0] == "https://hooks.slack.com/services/x/y/z"
    assert "blocks" in args[1]
