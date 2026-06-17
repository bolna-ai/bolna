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


@pytest.fixture(autouse=True)
def _bypass_ssrf_validation():
    """The fallback pre-call webhook now SSRF-validates its target via a real DNS
    lookup. These tests exercise payload/bookkeeping logic with non-resolving
    placeholder hosts, so stub the validator out here; the validator's own
    behaviour is covered in test_ssrf_validation.py."""

    async def _noop(url):
        return None

    with patch("bolna.agent_manager.task_manager.validate_outbound_url", _noop):
        yield


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
async def test_pre_call_webhook_no_template_sends_common_only():
    """With no pre_call_webhook_param template, the webhook sends ONLY the common
    call-state fields — the LLM args are NOT dumped into it."""
    me = _make_self()
    resp = {
        "reason": "customer asked for billing",
        "call_transfer_number": "+15559998888",
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
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    sent = _FakeSession.last_post
    assert sent["url"] == "https://hook.example/notify"
    body = sent["json"]
    # common call-state fields present
    assert body["execution_id"] == "exec-123"
    assert body["agent_id"] == "agent-1"
    assert body["provider"] == "plivo"
    assert body["from_number"] == "+15551112222"
    assert body["to_number"] == "+15553334444"
    # NO LLM args / internal keys / excluded fields
    for k in (
        "reason",
        "call_transfer_number",
        "model_response",
        "textual_response",
        "tool_call_id",
        "resp",
        "stream_sid",
        "tool_name",
        "call_sid",
    ):
        assert k not in body


@pytest.mark.asyncio
async def test_pre_call_webhook_param_template_controls_body():
    """A pre_call_webhook_param template defines the webhook body independently of the
    function-call param: only its fields are sent (+ common call-state fields)."""
    me = _make_self()
    resp = {"reason": "billing", "customer_name": "Alice", "tool_call_id": "tc-1"}
    webhook_param = {"who": "%(customer_name)s", "channel": "voice"}  # template, not the function param

    with (
        patch("bolna.agent_manager.task_manager.aiohttp.ClientSession", _FakeSession),
        patch("bolna.agent_manager.task_manager.convert_to_request_log"),
    ):
        TaskManager.fire_pre_call_webhook(me, "https://hook.example/notify", "custom_task_x", resp, {}, webhook_param)
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    body = _FakeSession.last_post["json"]
    assert body["who"] == "Alice"  # templated from LLM arg
    assert body["channel"] == "voice"  # static value in template
    assert body["execution_id"] == "exec-123"  # common fields still added
    assert body["agent_id"] == "agent-1"
    assert "reason" not in body  # not in the template → not sent
    assert "customer_name" not in body


@pytest.mark.asyncio
async def test_pre_call_webhook_common_fields_win_over_template():
    """If the template sets a field that collides with a common call-state field,
    the system value wins (common fields are spread last)."""
    me = _make_self()
    resp = {"reason": "transfer me"}
    webhook_param = {"provider": "bogus", "agent_id": "fake", "note": "%(reason)s"}

    with (
        patch("bolna.agent_manager.task_manager.aiohttp.ClientSession", _FakeSession),
        patch("bolna.agent_manager.task_manager.convert_to_request_log"),
    ):
        TaskManager.fire_pre_call_webhook(
            me, "https://hook.example/notify", "custom_task_transfer", resp, {}, webhook_param
        )
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    body = _FakeSession.last_post["json"]
    assert body["provider"] == "plivo"  # common wins over template
    assert body["agent_id"] == "agent-1"  # common wins over template
    assert body["note"] == "transfer me"  # templated from LLM arg


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
async def test_pre_call_webhook_dispatch_mode(monkeypatch):
    """With PRE_CALL_WEBHOOK_DISPATCH_URL set, bolna POSTs {execution_id, webhook_url,
    params} to the dispatch endpoint (the backend enriches with the full record)."""
    monkeypatch.setenv("PRE_CALL_WEBHOOK_DISPATCH_URL", "https://ts.example/dispatch_pre_call_webhook")
    me = _make_self()
    resp = {"customer_name": "Alice"}
    webhook_param = {"who": "%(customer_name)s"}

    with (
        patch("bolna.agent_manager.task_manager.aiohttp.ClientSession", _FakeSession),
        patch("bolna.agent_manager.task_manager.convert_to_request_log"),
    ):
        TaskManager.fire_pre_call_webhook(me, "https://customer/notify", "custom_task_x", resp, {}, webhook_param)
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    sent = _FakeSession.last_post
    assert sent["url"] == "https://ts.example/dispatch_pre_call_webhook"  # → dispatch endpoint
    body = sent["json"]
    assert body["execution_id"] == "exec-123"
    assert body["webhook_url"] == "https://customer/notify"  # customer URL passed through
    assert body["params"] == {"who": "Alice"}  # substituted template
    assert "provider" not in body  # common fields added by backend, not here


@pytest.mark.asyncio
async def test_pre_call_webhook_lazy_inits_background_tasks():
    """fire_pre_call_webhook must work even if __init__ never set background_tasks
    (regression guard: prod hit 'TaskManager has no attribute background_tasks' after a
    merge dropped the init line)."""
    me = _make_self()
    delattr(me, "background_tasks")  # simulate __init__ not initializing it

    with (
        patch("bolna.agent_manager.task_manager.aiohttp.ClientSession", _FakeSession),
        patch("bolna.agent_manager.task_manager.convert_to_request_log"),
    ):
        # must NOT raise AttributeError
        TaskManager.fire_pre_call_webhook(me, "https://hook.example/notify", "custom_task_x", {"reason": "x"}, {})
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    assert hasattr(me, "background_tasks")


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
    assert ctx["provider"] == "plivo"
    assert ctx["from_number"] == "+15551112222"
    assert ctx["to_number"] == "+15553334444"
    assert "stream_sid" not in ctx  # transfer-specific — removed
    assert "call_sid" not in ctx  # excluded from customer call-event webhooks
