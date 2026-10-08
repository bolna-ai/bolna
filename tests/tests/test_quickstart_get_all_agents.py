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
async def test_all_agents_correct_pairing_when_one_fetch_fails(app):
    """
    If fetching agent B fails, A->data_A and C->data_C must still be returned.
    The old code would return A->data_A and B->data_C (shifted zip).
    """
    mock_redis.keys = AsyncMock(return_value=["agent-A", "agent-B", "agent-C"])

    data_A = json.dumps({"name": "alpha"})
    data_C = json.dumps({"name": "gamma"})

    async def fake_get(key):
        if key == "agent-A":
            return data_A
        if key == "agent-B":
            raise Exception("Redis timeout for B")
        if key == "agent-C":
            return data_C

    mock_redis.get = fake_get

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        resp = await client.get("/all")

    assert resp.status_code == 200
    agents = {a["agent_id"]: a["data"] for a in resp.json()["agents"]}

    # A must map to alpha, C must map to gamma
    assert "agent-A" in agents
    assert agents["agent-A"]["name"] == "alpha", "agent-A has wrong data"

    assert "agent-C" in agents
    assert agents["agent-C"]["name"] == "gamma", (
        "agent-C has wrong data — bug #905: zip shift caused data_C to be returned under agent-B"
    )

    # B failed so it should be absent
    assert "agent-B" not in agents


@pytest.mark.asyncio
async def test_all_agents_returns_empty_when_no_keys(app):
    mock_redis.keys = AsyncMock(return_value=[])
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        resp = await client.get("/all")
    assert resp.status_code == 200
    assert resp.json() == {"agents": []}


@pytest.mark.asyncio
async def test_all_agents_skips_keys_with_none_data(app):
    """Keys that return None (e.g. expired TTL) should be silently skipped."""
    mock_redis.keys = AsyncMock(return_value=["agent-X", "agent-Y"])
    mock_redis.get = AsyncMock(side_effect=[None, json.dumps({"name": "Y"})])
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        resp = await client.get("/all")
    agents = {a["agent_id"]: a["data"] for a in resp.json()["agents"]}
    assert "agent-X" not in agents
    assert "agent-Y" in agents
    assert agents["agent-Y"]["name"] == "Y"


@pytest.mark.asyncio
async def test_all_agents_returns_all_when_no_failures(app):
    """Happy path: all three agents returned with correct IDs."""
    mock_redis.keys = AsyncMock(return_value=["a1", "a2", "a3"])
    mock_redis.get = AsyncMock(side_effect=[
        json.dumps({"name": "one"}),
        json.dumps({"name": "two"}),
        json.dumps({"name": "three"}),
    ])
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        resp = await client.get("/all")
    agents = {a["agent_id"]: a["data"] for a in resp.json()["agents"]}
    assert agents["a1"]["name"] == "one"
    assert agents["a2"]["name"] == "two"
    assert agents["a3"]["name"] == "three"
