"""Bedrock credentials reach litellm pre-resolved, and are resolved in a worker thread.

litellm signs Bedrock requests synchronously; left to refresh IRSA credentials itself, it makes a
blocking STS call on the event loop, freezing every call on the worker for as long as STS takes.
"""

import asyncio
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from botocore.credentials import ReadOnlyCredentials
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM

from bolna.llms import aws_credentials
from bolna.llms.aws_credentials import AwsCredentialCache
from bolna.llms.litellm import LiteLLM

FIRST = ReadOnlyCredentials("AKIA1", "secret1", "token1")
SECOND = ReadOnlyCredentials("AKIA2", "secret2", "token2")


class _Credentials:
    """Stands in for botocore's refreshable credentials; each get_frozen_credentials is one fetch."""

    def __init__(self, *results, block=None, delay=0.0):
        self.results = list(results)
        self.calls = 0
        self.threads = []
        self.block = block
        self.delay = delay

    def get_frozen_credentials(self):
        self.calls += 1
        self.threads.append(threading.current_thread())
        if self.block is not None:
            self.block.wait(5)
        time.sleep(self.delay)
        result = self.results[min(self.calls, len(self.results)) - 1]
        if isinstance(result, Exception):
            raise result
        return result


def _session(credentials):
    return patch.object(
        aws_credentials.botocore.session,
        "get_session",
        return_value=SimpleNamespace(get_credentials=lambda: credentials),
    )


async def test_resolves_in_a_worker_thread_while_the_loop_keeps_running():
    creds = _Credentials(FIRST, delay=0.2)
    ticks = 0

    async def heartbeat():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    hb = asyncio.create_task(heartbeat())
    with _session(creds):
        kwargs = await AwsCredentialCache().litellm_kwargs()
    hb.cancel()

    assert kwargs == {"aws_access_key_id": "AKIA1", "aws_secret_access_key": "secret1", "aws_session_token": "token1"}
    assert creds.threads[0] is not threading.main_thread()
    assert ticks > 5


async def test_fresh_credentials_are_served_from_cache():
    creds = _Credentials(FIRST, SECOND)
    cache = AwsCredentialCache()
    with _session(creds):
        await cache.litellm_kwargs()
        kwargs = await cache.litellm_kwargs()

    assert kwargs["aws_access_key_id"] == "AKIA1"
    assert creds.calls == 1


async def test_aging_credentials_are_served_while_a_background_refetch_runs():
    release = threading.Event()
    creds = _Credentials(FIRST, SECOND)
    cache = AwsCredentialCache(refetch_after_s=0.0, stale_after_s=3600.0)
    with _session(creds):
        await cache.litellm_kwargs()
        creds.block = release

        kwargs = await asyncio.wait_for(cache.litellm_kwargs(), timeout=1)
        assert kwargs["aws_access_key_id"] == "AKIA1"

        release.set()
        await cache._refresh_task
        assert (await cache.litellm_kwargs())["aws_access_key_id"] == "AKIA2"
        await cache._refresh_task


async def test_stale_credentials_wait_for_the_refetch():
    creds = _Credentials(FIRST, SECOND)
    cache = AwsCredentialCache(refetch_after_s=0.0, stale_after_s=0.0)
    with _session(creds):
        await cache.litellm_kwargs()
        kwargs = await cache.litellm_kwargs()

    assert kwargs["aws_access_key_id"] == "AKIA2"


async def test_concurrent_callers_share_one_fetch():
    creds = _Credentials(FIRST, delay=0.1)
    cache = AwsCredentialCache()
    with _session(creds):
        results = await asyncio.gather(*(cache.litellm_kwargs() for _ in range(5)))

    assert creds.calls == 1
    assert all(r["aws_access_key_id"] == "AKIA1" for r in results)


async def test_failed_fetch_with_nothing_cached_raises():
    cache = AwsCredentialCache()
    with _session(_Credentials(RuntimeError("sts unreachable"))):
        with pytest.raises(RuntimeError, match="sts unreachable"):
            await cache.litellm_kwargs()


async def test_failed_background_refetch_keeps_serving_the_old_credentials():
    creds = _Credentials(FIRST, RuntimeError("sts unreachable"))
    cache = AwsCredentialCache(refetch_after_s=0.0, stale_after_s=3600.0)
    with _session(creds):
        await cache.litellm_kwargs()
        await cache.litellm_kwargs()
        await asyncio.gather(cache._refresh_task, return_exceptions=True)
        kwargs = await cache.litellm_kwargs()
        await asyncio.gather(cache._refresh_task, return_exceptions=True)

    assert kwargs["aws_access_key_id"] == "AKIA1"


async def test_static_keys_send_no_session_token():
    with _session(_Credentials(ReadOnlyCredentials("AKIA1", "secret1", None))):
        kwargs = await AwsCredentialCache().litellm_kwargs()

    assert kwargs == {"aws_access_key_id": "AKIA1", "aws_secret_access_key": "secret1"}


async def test_no_aws_configuration_yields_nothing_and_is_not_retried_per_call():
    cache = AwsCredentialCache()
    with patch.object(
        aws_credentials.botocore.session,
        "get_session",
        return_value=SimpleNamespace(get_credentials=lambda: None),
    ) as get_session:
        assert await cache.litellm_kwargs() == {}
        assert await cache.litellm_kwargs() == {}

    assert get_session.call_count == 1


def _completion():
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))], usage=None)


async def test_bedrock_model_passes_resolved_credentials_to_litellm():
    llm = LiteLLM(model="bedrock/global.anthropic.claude-haiku-4-5", llm_key="", aws_region_name="us-east-1")
    resolved = {"aws_access_key_id": "AKIA1", "aws_secret_access_key": "secret1", "aws_session_token": "token1"}
    acompletion = AsyncMock(return_value=_completion())

    with (
        patch("bolna.llms.litellm.bedrock_credentials", AsyncMock(return_value=resolved)),
        patch("bolna.llms.litellm.acompletion", acompletion),
    ):
        await llm.generate([{"role": "user", "content": "hi"}])

    kwargs = acompletion.await_args.kwargs
    assert kwargs["aws_session_token"] == "token1"
    assert kwargs["aws_region_name"] == "us-east-1"
    assert "aws_session_token" not in llm.model_args


async def test_non_bedrock_model_does_not_touch_aws_credentials():
    llm = LiteLLM(model="anthropic/claude-haiku-4-5", llm_key="sk-test")
    bedrock_credentials = AsyncMock()
    acompletion = AsyncMock(return_value=_completion())

    with (
        patch("bolna.llms.litellm.bedrock_credentials", bedrock_credentials),
        patch("bolna.llms.litellm.acompletion", acompletion),
    ):
        await llm.generate([{"role": "user", "content": "hi"}])

    bedrock_credentials.assert_not_awaited()
    assert not any(k.startswith("aws_") for k in acompletion.await_args.kwargs)


def test_litellm_signs_with_passed_credentials_without_refreshing_its_own(monkeypatch):
    """The litellm behaviour this fix relies on: explicit keys build static credentials, not the default chain."""
    for var in ("AWS_ROLE_NAME", "AWS_PROFILE_NAME", "AWS_WEB_IDENTITY_TOKEN"):
        monkeypatch.delenv(var, raising=False)

    with patch.object(
        BaseAWSLLM, "_auth_with_env_vars", side_effect=AssertionError("litellm resolved its own credentials")
    ):
        credentials = BaseAWSLLM().get_credentials(
            aws_access_key_id="AKIA1",
            aws_secret_access_key="secret1",
            aws_session_token="token1",
            aws_region_name="us-east-1",
        )

    assert (credentials.access_key, credentials.secret_key, credentials.token) == ("AKIA1", "secret1", "token1")
    assert not hasattr(credentials, "refresh_needed")
