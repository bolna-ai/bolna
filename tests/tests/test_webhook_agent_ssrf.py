from unittest.mock import AsyncMock

import aiohttp
import pytest

import bolna.agent_types.webhook_agent as webhook_module
import bolna.helpers.function_calling_helpers as function_calling_helpers
from bolna.agent_types.webhook_agent import WEBHOOK_TIMEOUT_SECONDS, WebhookAgent
from bolna.helpers.function_calling_helpers import SSRFError


class _FakeResponse:
    def __init__(self, status=200, body=""):
        self.status = status
        self.body = body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def text(self):
        return self.body


class _FakeSession:
    def __init__(self, response, requests, timeout):
        self.response = response
        self.requests = requests
        self.timeout = timeout

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    def post(self, url, **kwargs):
        self.requests.append((url, kwargs, self.timeout))
        return self.response


def _install_fake_session(monkeypatch, response):
    requests = []

    def session_factory(*, timeout):
        return _FakeSession(response, requests, timeout)

    monkeypatch.setattr(webhook_module.aiohttp, "ClientSession", session_factory)
    return requests


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/internal",
        "http://10.0.0.5/internal",
        "http://169.254.169.254/latest/meta-data/",
        "http://[::1]/internal",
        "file:///etc/passwd",
    ],
)
async def test_webhook_agent_blocks_unsafe_targets_before_opening_session(monkeypatch, url):
    session_factory = AsyncMock()
    monkeypatch.setattr(webhook_module.aiohttp, "ClientSession", session_factory)

    result = await WebhookAgent(url).execute({"event": "test"})

    assert result is None
    session_factory.assert_not_called()


@pytest.mark.asyncio
async def test_webhook_agent_blocks_hostname_resolving_to_private_ip(monkeypatch):
    async def reject_private_dns(_url):
        raise SSRFError("blocked private DNS result")

    session_factory = AsyncMock()
    monkeypatch.setattr(webhook_module, "validate_outbound_url", reject_private_dns)
    monkeypatch.setattr(webhook_module.aiohttp, "ClientSession", session_factory)

    result = await WebhookAgent("https://public-looking.example/hook").execute({"event": "test"})

    assert result is None
    session_factory.assert_not_called()


@pytest.mark.asyncio
async def test_webhook_agent_disables_redirects_and_uses_bounded_timeout(monkeypatch):
    validate = AsyncMock()
    requests = _install_fake_session(monkeypatch, _FakeResponse(status=302, body="redirect"))
    monkeypatch.setattr(webhook_module, "validate_outbound_url", validate)

    result = await WebhookAgent("https://example.com/hook").execute({"event": "test"})

    assert result is None
    validate.assert_awaited_once_with("https://example.com/hook")
    assert len(requests) == 1
    url, kwargs, timeout = requests[0]
    assert url == "https://example.com/hook"
    assert kwargs == {"json": {"event": "test"}, "allow_redirects": False}
    assert isinstance(timeout, aiohttp.ClientTimeout)
    assert timeout.total == WEBHOOK_TIMEOUT_SECONDS


@pytest.mark.asyncio
async def test_webhook_agent_allows_explicitly_allowlisted_internal_host(monkeypatch):
    monkeypatch.setattr(function_calling_helpers, "_ALLOWLISTED_HOSTS", frozenset({"internal.example"}))
    requests = _install_fake_session(monkeypatch, _FakeResponse(status=200))

    result = await WebhookAgent("http://internal.example/hook").execute({"event": "test"})

    assert result is True
    assert len(requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(("status", "expected"), [(200, True), (500, None)])
async def test_webhook_agent_preserves_success_and_error_results(monkeypatch, status, expected):
    validate = AsyncMock()
    requests = _install_fake_session(monkeypatch, _FakeResponse(status=status, body="response"))
    monkeypatch.setattr(webhook_module, "validate_outbound_url", validate)

    result = await WebhookAgent("https://example.com/hook").execute({"event": "test"})

    assert result is expected
    assert len(requests) == 1
