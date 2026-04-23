"""AssistantManager post-call integration wiring: coerce raw dicts, dedup, strong ref."""

import asyncio
from unittest.mock import patch

import pytest

from bolna.agent_manager.assistant_manager import AssistantManager
from bolna.models import IntegrationConfig


def _make_manager(tasks):
    return AssistantManager(agent_config={"agent_name": "a", "tasks": tasks}, turn_based_conversation=True)


def test_collect_coerces_raw_dicts():
    tasks = [
        {
            "tools_config": {
                "integrations": [
                    {"provider": "slack", "provider_config": {"webhook_url": "https://x/1"}}
                ]
            }
        }
    ]
    mgr = _make_manager(tasks)
    configs = mgr._collect_integration_configs()
    assert len(configs) == 1
    assert isinstance(configs[0], IntegrationConfig)
    assert configs[0].provider == "slack"


def test_collect_dedupes_across_tasks():
    task = {
        "tools_config": {
            "integrations": [
                {"provider": "slack", "provider_config": {"webhook_url": "https://x/1"}}
            ]
        }
    }
    mgr = _make_manager([task, task])
    configs = mgr._collect_integration_configs()
    assert len(configs) == 1


def test_collect_handles_missing_tools_config():
    mgr = _make_manager([{}, {"tools_config": None}])
    assert mgr._collect_integration_configs() == []


def test_build_context_pulls_from_task_outputs():
    mgr = _make_manager([{}])
    outputs = [
        {
            "call_sid": "CA1",
            "conversation_time": 45.0,
            "hangup_detail": "inactivity_timeout",
            "recording_url": "https://r",
        },
        {"task_type": "summarization", "summary": "ok"},
        {"task_type": "extraction", "extracted_data": {"outcome": "resolved"}},
    ]
    ctx = mgr._build_post_call_context(outputs)
    assert ctx.call_sid == "CA1"
    assert ctx.duration_seconds == 45.0
    assert ctx.hangup_reason == "inactivity_timeout"
    assert ctx.summary == "ok"
    assert ctx.extracted_data == {"outcome": "resolved"}


@pytest.mark.asyncio
async def test_fire_post_call_integrations_holds_task_ref():
    tasks = [
        {
            "tools_config": {
                "integrations": [
                    {"provider": "slack", "provider_config": {"webhook_url": "https://x/1"}}
                ]
            }
        }
    ]
    mgr = _make_manager(tasks)
    fired = asyncio.Event()

    async def fake_runner(configs, ctx):
        fired.set()

    with patch("bolna.agent_manager.assistant_manager.run_post_call_integrations", new=fake_runner):
        mgr._fire_post_call_integrations([{}])
        assert len(mgr._background_tasks) == 1
        await asyncio.wait_for(fired.wait(), timeout=1.0)
    # done callback clears the set
    await asyncio.sleep(0)
    assert len(mgr._background_tasks) == 0


def test_no_fire_when_no_configs():
    mgr = _make_manager([{"tools_config": {}}])
    # should not raise, should not create any task
    mgr._fire_post_call_integrations([{}])
    assert len(mgr._background_tasks) == 0
