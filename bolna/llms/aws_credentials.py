"""AWS credentials for litellm's Bedrock calls, resolved in a worker thread.

litellm signs Bedrock requests synchronously, so credentials it refreshes itself (an STS call for
IRSA) block the event loop; handing it already-frozen credentials keeps that call off the loop.
"""

import asyncio
import time

import botocore.session

from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)

# botocore never hands out frozen credentials with under 10 min left, so refetching after 5 min
# keeps every served set valid; past 8 min a caller waits for the refetch instead.
REFETCH_AFTER_S = 300.0
STALE_AFTER_S = 480.0


class AwsCredentialCache:
    def __init__(self, refetch_after_s=REFETCH_AFTER_S, stale_after_s=STALE_AFTER_S):
        self.refetch_after_s = refetch_after_s
        self.stale_after_s = stale_after_s
        self._credentials = None
        self._frozen = None
        self._fetched_at = None
        self._refresh_task = None

    def _resolve(self):
        if self._credentials is None:
            self._credentials = botocore.session.get_session().get_credentials()
        if self._credentials is None:
            return None
        return self._credentials.get_frozen_credentials()

    async def _refresh(self):
        frozen = await asyncio.to_thread(self._resolve)
        self._frozen = frozen
        self._fetched_at = time.monotonic()

    def _start_refresh(self):
        task = self._refresh_task
        if task is None or task.done() or task.get_loop() is not asyncio.get_running_loop():
            task = asyncio.create_task(self._refresh())
            task.add_done_callback(self._log_refresh_failure)
            self._refresh_task = task
        return task

    @staticmethod
    def _log_refresh_failure(task):
        if not task.cancelled() and task.exception() is not None:
            logger.warning(f"AWS credential refresh failed: {task.exception()!r}")

    async def litellm_kwargs(self):
        """aws_* kwargs for litellm.acompletion; empty when no AWS credentials are configured."""
        age = None if self._fetched_at is None else time.monotonic() - self._fetched_at
        if age is None or age > self.stale_after_s:
            # Shielded: a cancelled caller must not cancel the refresh other callers share.
            await asyncio.shield(self._start_refresh())
        elif age > self.refetch_after_s:
            self._start_refresh()

        frozen = self._frozen
        if frozen is None:
            return {}
        kwargs = {"aws_access_key_id": frozen.access_key, "aws_secret_access_key": frozen.secret_key}
        if frozen.token:
            kwargs["aws_session_token"] = frozen.token
        return kwargs


_cache = AwsCredentialCache()


async def bedrock_credentials():
    return await _cache.litellm_kwargs()
