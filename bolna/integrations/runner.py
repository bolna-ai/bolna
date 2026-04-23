import asyncio
from typing import List, Optional

import aiohttp

from bolna.helpers.logger_config import configure_logger
from .base import PostCallContext, PostCallIntegration

logger = configure_logger(__name__)

_RETRY_DELAYS = [1.0, 2.0]
_REQUEST_TIMEOUT = 10.0
_INTEGRATION_TIMEOUT = 30.0


async def _post_with_retry(url: str, payload: dict, headers: Optional[dict] = None) -> None:
    last_exc: Optional[Exception] = None
    timeout = aiohttp.ClientTimeout(total=_REQUEST_TIMEOUT)
    attempts = len(_RETRY_DELAYS) + 1

    async with aiohttp.ClientSession(timeout=timeout) as session:
        for attempt in range(attempts):
            try:
                async with session.post(url, json=payload, headers=headers) as resp:
                    if resp.status < 400:
                        return
                    body = await resp.text()
                    if resp.status < 500:
                        logger.error(f"post to {url} failed with {resp.status}: {body[:200]}")
                        return
                    last_exc = RuntimeError(f"HTTP {resp.status}: {body[:200]}")
            except (aiohttp.ClientConnectionError, asyncio.TimeoutError) as e:
                last_exc = e

            if attempt < attempts - 1:
                await asyncio.sleep(_RETRY_DELAYS[attempt])

    if last_exc:
        raise last_exc


async def run_post_call_integrations(integration_configs: List, ctx: PostCallContext) -> None:
    if not integration_configs:
        return

    from bolna.providers import SUPPORTED_INTEGRATIONS

    for cfg in integration_configs:
        provider = cfg.provider
        cls = SUPPORTED_INTEGRATIONS.get(provider)
        if cls is None:
            logger.warning(f"unknown integration provider: {provider}")
            continue
        try:
            integration: PostCallIntegration = cls.from_config(cfg)
        except Exception as e:
            logger.error(f"could not build {provider} integration: {e}")
            continue

        try:
            await asyncio.wait_for(integration.execute(ctx), timeout=_INTEGRATION_TIMEOUT)
        except asyncio.TimeoutError:
            logger.error(f"{provider} integration timed out after {_INTEGRATION_TIMEOUT}s")
        except Exception as e:
            logger.error(f"{provider} integration failed: {e}")
