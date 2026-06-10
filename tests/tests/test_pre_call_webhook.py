"""Tests for the custom-tool pre_call_webhook feature.

Verifies that ``TaskManager.fire_pre_call_webhook`` POSTs the LLM args plus call
context to the configured URL, excludes internal bookkeeping keys, and never
raises (fire-and-forget) when the webhook endpoint fails.
"""

import asyncio
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from bolna.agent_manager.task_manager import TaskManager


def _make_self():
    """Minimal stand-in exposing only the attributes fire_pre_call_webhook reads."""
    inp = MagicMock()
    inp.io_provider = "plivo"
    inp.get_call_sid = MagicMock(return_value="call-abc")
    me = SimpleNamespace(
        run_id="exec-123",
        assistant_id="agent-1",
        call_sid="call-abc",
        stream_sid="stream-xyz",
        context_data={"recipient_data": {"from_number": "+15551112222", "to_number": "+15553334444"}},
        tools={"input": inp},
        background_tasks=set(),
        function_tool_api_call_details=[],
    )
    # Bind the real methods so fire_pre_call_webhook exercises them.
    me._build_call_context = types.MethodType(TaskManager._build_call_context, me)
    me._start_api_call_detail = types.MethodType(TaskManager._start_api_call_detail, me)
    me._finalize_api_call_detail = TaskManager._finalize_api_call_detail
    me._sanitize_api_call_headers = TaskManager._sanitize_api_call_headers
    return me


class _FakeResponse:
    status = 200
    headers = {"Content-Type": "application/json"}

    async def text(self):
        return "ok"

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


class _FakeSession:
    """Captures the (url, json=) of the POST for assertions."""

    last_post = {}

    def __init__(self, *a, **k):
        pass

    def post(self, url, json=None):
        _FakeSession.last_post = {"url": url, "json": json}
        return _FakeResponse()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


@pytest.mark.asyncio
async def test_pre_call_webhook_payload_and_target():
    me = _make_self()
    resp = {
        "reason": "customer asked for billing",
        "call_transfer_number": "+15559998888",
        # internal keys that must NOT be forwarded:
        "model_response": [{"x": 1}],
        "textual_response": "sure",
        "tool_call_id": "tc-1",
        "resp": "raw",
    }

    with (
        patch("bolna.agent_manager.task_manager.aiohttp.ClientSession", _FakeSession),
        patch("bolna.agent_manager.task_manager.convert_to_request_log"),
    ):
        TaskManager.fire_pre_call_webhook(
            me, "https://hook.example/notify", "custom_task_transfer", resp, {"turn_id": 1}
        )
        # let the background task run
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    sent = _FakeSession.last_post
    assert sent["url"] == "https://hook.example/notify"
    body = sent["json"]
    # common call-state fields
    assert body["execution_id"] == "exec-123"
    assert body["agent_id"] == "agent-1"
    assert body["call_sid"] == "call-abc"
    assert body["provider"] == "plivo"
    assert body["from_number"] == "+15551112222"
    assert body["to_number"] == "+15553334444"
    # transfer-specific fields must NOT be present
    assert "stream_sid" not in body
    assert "tool_name" not in body
    # LLM args (params) forwarded
    assert body["reason"] == "customer asked for billing"
    assert body["call_transfer_number"] == "+15559998888"
    # internal keys excluded
    for k in ("model_response", "textual_response", "tool_call_id", "resp"):
        assert k not in body


@pytest.mark.asyncio
async def test_pre_call_webhook_system_context_overrides_llm_args():
    """A model-fabricated call_sid/provider in the LLM args must NOT leak into the
    webhook — system context wins (regression guard for the payload-order review note)."""
    me = _make_self()
    resp = {
        "reason": "transfer me",
        "call_sid": "MODEL-MADE-UP-SID",  # LLM fabricated — must be overridden
        "provider": "bogus",
    }

    with (
        patch("bolna.agent_manager.task_manager.aiohttp.ClientSession", _FakeSession),
        patch("bolna.agent_manager.task_manager.convert_to_request_log"),
    ):
        TaskManager.fire_pre_call_webhook(me, "https://hook.example/notify", "custom_task_transfer", resp, {})
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    body = _FakeSession.last_post["json"]
    assert body["call_sid"] == "call-abc"  # system value wins, not the fabricated one
    assert body["provider"] == "plivo"
    assert body["reason"] == "transfer me"  # genuine LLM arg preserved


@pytest.mark.asyncio
async def test_pre_call_webhook_recorded_in_api_call_details():
    """The pre-call webhook must be appended to function_tool_api_call_details so it
    lands in the same per-call S3 record as the other API/tool calls."""
    me = _make_self()

    with (
        patch("bolna.agent_manager.task_manager.aiohttp.ClientSession", _FakeSession),
        patch("bolna.agent_manager.task_manager.convert_to_request_log"),
    ):
        TaskManager.fire_pre_call_webhook(
            me, "https://hook.example/notify", "custom_task_transfer", {"reason": "x"}, {"turn_id": 1}
        )
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    assert len(me.function_tool_api_call_details) == 1
    detail = me.function_tool_api_call_details[0]
    assert detail["tool_name"] == "custom_task_transfer:pre_call_webhook"
    assert detail["url"] == "https://hook.example/notify"
    assert detail["status"] == "completed"  # finalized after the POST
    assert detail["response_status_code"] == 200


@pytest.mark.asyncio
async def test_pre_call_webhook_keeps_strong_task_reference():
    """The fire-and-forget task must be retained (so the loop can't GC it mid-POST)
    and then discarded once it completes."""
    me = _make_self()

    with (
        patch("bolna.agent_manager.task_manager.aiohttp.ClientSession", _FakeSession),
        patch("bolna.agent_manager.task_manager.convert_to_request_log"),
    ):
        TaskManager.fire_pre_call_webhook(me, "https://hook.example/notify", "custom_task_x", {"reason": "x"}, {})
        # reference held while in flight
        assert len(me.background_tasks) == 1
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    # discarded after completion via done-callback
    assert len(me.background_tasks) == 0


@pytest.mark.asyncio
async def test_pre_call_webhook_swallows_errors():
    me = _make_self()

    class _BoomSession(_FakeSession):
        def post(self, url, json=None):
            raise RuntimeError("endpoint down")

    with (
        patch("bolna.agent_manager.task_manager.aiohttp.ClientSession", _BoomSession),
        patch("bolna.agent_manager.task_manager.convert_to_request_log"),
    ):
        # must not raise even though the POST blows up
        TaskManager.fire_pre_call_webhook(me, "https://hook.example/notify", "custom_task_x", {}, {})
        await asyncio.sleep(0)
        await asyncio.sleep(0)


def test_build_call_context_has_common_fields_only():
    """Call context carries the common call-state fields — and NO transfer-specific ones."""
    me = _make_self()
    ctx = TaskManager._build_call_context(me)
    assert ctx["execution_id"] == "exec-123"
    assert ctx["agent_id"] == "agent-1"
    assert ctx["call_sid"] == "call-abc"  # from get_call_sid() for non-default provider
    assert ctx["provider"] == "plivo"
    assert ctx["from_number"] == "+15551112222"
    assert ctx["to_number"] == "+15553334444"
    assert "stream_sid" not in ctx          # transfer-specific — removed
