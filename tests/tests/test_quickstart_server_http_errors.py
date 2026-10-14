import pytest
import httpx
import json
from unittest.mock import AsyncMock, MagicMock, patch


mock_pool = MagicMock()
mock_redis = AsyncMock()


@pytest.fixture()
def app():
    with patch("redis.asyncio.ConnectionPool.from_url", return_value=mock_pool):
        with patch("redis.asyncio.Redis.from_pool", return_value=mock_redis):
            import importlib
            import local_setup.quickstart_server as qs
            importlib.reload(qs)
            qs.redis_client = mock_redis
            yield qs.app


@pytest.mark.asyncio
async def test_get_missing_agent_returns_404(app):
    mock_redis.get = AsyncMock(return_value=None)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        resp = await client.get("/agent/nonexistent-id")
    assert resp.status_code == 404, (
        f"Expected 404, got {resp.status_code} — bug #904: HTTPException(404) swallowed as 500"
    )
    assert resp.json()["detail"] == "Agent not found"


@pytest.mark.asyncio
async def test_get_agent_redis_error_returns_500(app):
    mock_redis.get = AsyncMock(side_effect=Exception("Redis down"))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        resp = await client.get("/agent/some-id")
    assert resp.status_code == 500


@pytest.mark.asyncio
async def test_delete_missing_agent_returns_404(app):
    mock_redis.exists = AsyncMock(return_value=0)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        resp = await client.delete("/agent/nonexistent-id")
    assert resp.status_code == 404, (
        f"Expected 404, got {resp.status_code} — bug #904: HTTPException(404) swallowed as 500"
    )
    assert resp.json()["detail"] == "Agent not found"


@pytest.mark.asyncio
async def test_delete_agent_redis_error_returns_500(app):
    mock_redis.exists = AsyncMock(side_effect=Exception("Redis down"))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        resp = await client.delete("/agent/some-id")
    assert resp.status_code == 500


@pytest.mark.asyncio
async def test_put_missing_agent_returns_404(app):
    mock_redis.get = AsyncMock(return_value=None)
    payload = {
        "agent_config": {"agent_name": "test", "agent_type": "other", "tasks": []},
        "agent_prompts": {}
    }
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        resp = await client.put("/agent/nonexistent-id", json=payload)
    assert resp.status_code == 404, (
        f"Expected 404, got {resp.status_code} — bug #904: HTTPException(404) swallowed as 500"
    )
    assert resp.json()["detail"] == "Agent not found"
