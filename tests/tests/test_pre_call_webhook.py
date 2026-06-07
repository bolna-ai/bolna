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
        call_sid="call-abc",
        stream_sid="stream-xyz",
        context_data={"recipient_data": {"from_number": "+15551112222"}},
        tools={"input": inp},
    )
    # Bind the real _build_call_context so fire_pre_call_webhook exercises it.
    me._build_call_context = types.MethodType(TaskManager._build_call_context, me)
    return me


class _FakeResponse:
    status = 200

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
    # call context
    assert body["execution_id"] == "exec-123"
    assert body["call_sid"] == "call-abc"
    assert body["stream_sid"] == "stream-xyz"
    assert body["provider"] == "plivo"
    assert body["from_number"] == "+15551112222"
    assert body["tool_name"] == "custom_task_transfer"
    # LLM args forwarded
    assert body["reason"] == "customer asked for billing"
    assert body["call_transfer_number"] == "+15559998888"
    # internal keys excluded
    for k in ("model_response", "textual_response", "tool_call_id", "resp"):
        assert k not in body


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


def test_build_call_context_exposes_transfer_fields():
    """Call context must carry the fields /process_transfer needs, so a custom tool
    can drive the transfer itself via %(...)s substitution."""
    me = _make_self()
    ctx = TaskManager._build_call_context(me)
    assert ctx["call_sid"] == "call-abc"  # from get_call_sid() for non-default provider
    assert ctx["provider"] == "plivo"
    assert ctx["stream_sid"] == "stream-xyz"
    assert ctx["from_number"] == "+15551112222"
    assert ctx["execution_id"] == "exec-123"


def test_build_call_context_injected_into_resp_overrides():
    """resp.update(call_context) makes the fields available for param substitution,
    and system values win over any same-named LLM arg (the LLM can't know stream_sid)."""
    me = _make_self()
    resp = {"reason": "user asked", "stream_sid": "llm-guessed-wrong", "tool_call_id": "tc-1"}
    resp.update(TaskManager._build_call_context(me))
    assert resp["stream_sid"] == "stream-xyz"  # system value wins
    assert resp["call_sid"] == "call-abc"
    assert resp["reason"] == "user asked"  # LLM arg preserved
