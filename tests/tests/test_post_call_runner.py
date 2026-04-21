"""Runner: dedup, retry, swallow-on-failure, timeout."""

import asyncio
from unittest.mock import patch

import aiohttp
import pytest

from bolna.integrations.base import PostCallContext, PostCallIntegration
from bolna.integrations.runner import _post_with_retry, run_post_call_integrations


def _ctx():
    return PostCallContext(agent_name="agent", run_id="r-1")


class _Cfg:
    def __init__(self, provider):
        self.provider = provider


class _Recorder(PostCallIntegration):
    def __init__(self, fail_with=None, sleep=None):
        self.fail_with = fail_with
        self.sleep = sleep
        self.calls = 0

    @classmethod
    def from_config(cls, config):
        return cls()

    async def execute(self, ctx):
        self.calls += 1
        if self.sleep:
            await asyncio.sleep(self.sleep)
        if self.fail_with:
            raise self.fail_with


@pytest.mark.asyncio
async def test_run_with_no_configs_is_noop():
    await run_post_call_integrations([], _ctx())
    await run_post_call_integrations(None, _ctx())


@pytest.mark.asyncio
async def test_unknown_provider_is_logged_and_skipped():
    await run_post_call_integrations([_Cfg("nonexistent")], _ctx())


@pytest.mark.asyncio
async def test_known_provider_is_invoked():
    rec = _Recorder()
    with patch.dict(
        "bolna.providers.SUPPORTED_INTEGRATIONS",
        {"slack": type("S", (), {"from_config": classmethod(lambda c, cfg: rec)})},
    ):
        await run_post_call_integrations([_Cfg("slack")], _ctx())
    assert rec.calls == 1


@pytest.mark.asyncio
async def test_failing_integration_is_swallowed():
    rec = _Recorder(fail_with=RuntimeError("boom"))
    with patch.dict(
        "bolna.providers.SUPPORTED_INTEGRATIONS",
        {"slack": type("S", (), {"from_config": classmethod(lambda c, cfg: rec)})},
    ):
        await run_post_call_integrations([_Cfg("slack")], _ctx())
    assert rec.calls == 1


@pytest.mark.asyncio
async def test_integration_timeout_is_swallowed(monkeypatch):
    monkeypatch.setattr("bolna.integrations.runner._INTEGRATION_TIMEOUT", 0.05)
    rec = _Recorder(sleep=0.5)
    with patch.dict(
        "bolna.providers.SUPPORTED_INTEGRATIONS",
        {"slack": type("S", (), {"from_config": classmethod(lambda c, cfg: rec)})},
    ):
        await run_post_call_integrations([_Cfg("slack")], _ctx())


@pytest.mark.asyncio
async def test_from_config_failure_does_not_block_other_providers():
    good = _Recorder()

    class Bad:
        @classmethod
        def from_config(cls, config):
            raise ValueError("missing creds")

    with patch.dict(
        "bolna.providers.SUPPORTED_INTEGRATIONS",
        {"slack": Bad, "notion": type("N", (), {"from_config": classmethod(lambda c, cfg: good)})},
    ):
        await run_post_call_integrations([_Cfg("slack"), _Cfg("notion")], _ctx())
    assert good.calls == 1


# ---------- _post_with_retry ----------


class _MockResponse:
    def __init__(self, status, text="ok"):
        self.status = status
        self._text = text

    async def text(self):
        return self._text

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


class _MockSession:
    def __init__(self, statuses=None, exc=None):
        self._responses = [_MockResponse(s) for s in (statuses or [])]
        self._exc = exc
        self.posts = []

    def post(self, url, json=None, headers=None):
        self.posts.append((url, json, headers))
        if self._exc:
            raise self._exc
        return self._responses.pop(0)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


def _patch_session(session):
    return patch("aiohttp.ClientSession", lambda *a, **kw: session)


@pytest.mark.asyncio
async def test_post_retry_2xx_returns_immediately():
    session = _MockSession(statuses=[200])
    with _patch_session(session):
        await _post_with_retry("https://example", {"x": 1})
    assert len(session.posts) == 1


@pytest.mark.asyncio
async def test_post_retry_4xx_no_retry():
    session = _MockSession(statuses=[400])
    with _patch_session(session):
        await _post_with_retry("https://example", {"x": 1})
    assert len(session.posts) == 1


@pytest.mark.asyncio
async def test_post_retry_5xx_then_success(monkeypatch):
    monkeypatch.setattr("bolna.integrations.runner._RETRY_DELAYS", [0.0, 0.0])
    session = _MockSession(statuses=[503, 503, 200])
    with _patch_session(session):
        await _post_with_retry("https://example", {"x": 1})
    assert len(session.posts) == 3


@pytest.mark.asyncio
async def test_post_retry_5xx_exhausted_raises(monkeypatch):
    monkeypatch.setattr("bolna.integrations.runner._RETRY_DELAYS", [0.0, 0.0])
    session = _MockSession(statuses=[500, 500, 500])
    with _patch_session(session):
        with pytest.raises(RuntimeError):
            await _post_with_retry("https://example", {"x": 1})
    assert len(session.posts) == 3


@pytest.mark.asyncio
async def test_post_retry_connection_error_then_success(monkeypatch):
    monkeypatch.setattr("bolna.integrations.runner._RETRY_DELAYS", [0.0, 0.0])

    calls = {"n": 0}
    ok = _MockResponse(200)

    class _FlakySession:
        def __init__(self, *a, **kw):
            pass

        def post(self, url, json=None, headers=None):
            calls["n"] += 1
            if calls["n"] <= 2:
                raise aiohttp.ClientConnectionError("transient")
            return ok

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    with patch("aiohttp.ClientSession", _FlakySession):
        await _post_with_retry("https://example", {"x": 1})
    assert calls["n"] == 3
