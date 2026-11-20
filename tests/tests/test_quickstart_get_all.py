import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest
import pytest_asyncio


@pytest_asyncio.fixture
async def quickstart(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379")
    module_name = "quickstart_server_get_all_under_test"
    server = sys.modules.get(module_name)
    if server is None:
        server_path = Path(__file__).resolve().parents[2] / "local_setup" / "quickstart_server.py"
        spec = importlib.util.spec_from_file_location(module_name, server_path)
        assert spec is not None and spec.loader is not None
        server = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = server
        spec.loader.exec_module(server)

    redis_client = AsyncMock()
    monkeypatch.setattr(server, "redis_client", redis_client)

    transport = httpx.ASGITransport(app=server.app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client, redis_client


def _redis_get(values, failed_key=None):
    async def get(key):
        if key == failed_key:
            raise RuntimeError("redis read failed")
        return values[key]

    return get


@pytest.mark.asyncio
async def test_all_agent_values_remain_attached_to_their_keys(quickstart):
    client, redis_client = quickstart
    redis_client.keys.return_value = ["agent-a", "agent-b", "agent-c"]
    redis_client.get.side_effect = _redis_get(
        {
            "agent-a": json.dumps({"name": "A"}),
            "agent-b": json.dumps({"name": "B"}),
            "agent-c": json.dumps({"name": "C"}),
        }
    )

    response = await client.get("/all")

    assert response.status_code == 200
    assert response.json() == {
        "agents": [
            {"agent_id": "agent-a", "data": {"name": "A"}},
            {"agent_id": "agent-b", "data": {"name": "B"}},
            {"agent_id": "agent-c", "data": {"name": "C"}},
        ]
    }


@pytest.mark.parametrize("failed_key", ["agent-a", "agent-b", "agent-c"])
@pytest.mark.asyncio
async def test_failed_read_never_shifts_another_agents_data(quickstart, failed_key):
    client, redis_client = quickstart
    keys = ["agent-a", "agent-b", "agent-c"]
    values = {
        "agent-a": json.dumps({"name": "A"}),
        "agent-b": json.dumps({"name": "B"}),
        "agent-c": json.dumps({"name": "C"}),
    }
    redis_client.keys.return_value = keys
    redis_client.get.side_effect = _redis_get(values, failed_key=failed_key)

    response = await client.get("/all")

    assert response.status_code == 200
    assert response.json() == {
        "agents": [{"agent_id": key, "data": json.loads(values[key])} for key in keys if key != failed_key],
        "failed_agent_ids": [failed_key],
    }


@pytest.mark.asyncio
async def test_missing_value_is_skipped_without_shifting_later_data(quickstart):
    client, redis_client = quickstart
    redis_client.keys.return_value = ["agent-a", "agent-b", "agent-c"]
    redis_client.get.side_effect = _redis_get(
        {
            "agent-a": json.dumps({"name": "A"}),
            "agent-b": None,
            "agent-c": json.dumps({"name": "C"}),
        }
    )

    response = await client.get("/all")

    assert response.status_code == 200
    assert response.json() == {
        "agents": [
            {"agent_id": "agent-a", "data": {"name": "A"}},
            {"agent_id": "agent-c", "data": {"name": "C"}},
        ]
    }


@pytest.mark.asyncio
async def test_invalid_json_is_reported_without_corrupting_other_mappings(quickstart):
    client, redis_client = quickstart
    redis_client.keys.return_value = ["agent-a", "agent-b", "agent-c"]
    redis_client.get.side_effect = _redis_get(
        {
            "agent-a": json.dumps({"name": "A"}),
            "agent-b": "not-json",
            "agent-c": json.dumps({"name": "C"}),
        }
    )

    response = await client.get("/all")

    assert response.status_code == 200
    assert response.json() == {
        "agents": [
            {"agent_id": "agent-a", "data": {"name": "A"}},
            {"agent_id": "agent-c", "data": {"name": "C"}},
        ],
        "failed_agent_ids": ["agent-b"],
    }
