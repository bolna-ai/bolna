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
    module_name = "quickstart_server_under_test"
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


def _agent_payload(tasks=None):
    return {
        "agent_config": {
            "agent_name": "test-agent",
            "tasks": tasks or [],
        },
        "agent_prompts": None,
    }


def _extraction_task():
    return {
        "task_type": "extraction",
        "tools_config": {
            "llm_agent": {
                "agent_flow_type": "streaming",
                "agent_type": "simple_llm_agent",
                "llm_config": {"extraction_details": "Extract the caller's name"},
            }
        },
        "toolchain": {"execution": "sequential", "pipelines": [["llm"]]},
        "task_config": {},
    }


@pytest.mark.asyncio
async def test_missing_agent_get_returns_404(quickstart):
    client, redis_client = quickstart
    redis_client.get.return_value = None

    response = await client.get("/agent/missing")

    assert response.status_code == 404
    assert response.json() == {"detail": "Agent not found"}


@pytest.mark.asyncio
async def test_missing_agent_edit_returns_404(quickstart):
    client, redis_client = quickstart
    redis_client.get.return_value = None

    response = await client.put("/agent/missing", json=_agent_payload())

    assert response.status_code == 404
    assert response.json() == {"detail": "Agent not found"}


@pytest.mark.asyncio
async def test_missing_agent_delete_returns_404(quickstart):
    client, redis_client = quickstart
    redis_client.exists.return_value = 0

    response = await client.delete("/agent/missing")

    assert response.status_code == 404
    assert response.json() == {"detail": "Agent not found"}


@pytest.mark.parametrize(
    ("method", "path", "redis_method", "request_kwargs"),
    [
        ("get", "/agent/test", "get", {}),
        ("put", "/agent/test", "get", {"json": _agent_payload()}),
        ("delete", "/agent/test", "exists", {}),
    ],
)
@pytest.mark.asyncio
async def test_redis_failures_return_500(quickstart, method, path, redis_method, request_kwargs):
    client, redis_client = quickstart
    getattr(redis_client, redis_method).side_effect = RuntimeError("redis unavailable")

    response = await getattr(client, method)(path, **request_kwargs)

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}


@pytest.mark.asyncio
async def test_configured_extraction_error_preserves_detail(quickstart, monkeypatch):
    client, redis_client = quickstart
    redis_client.get.return_value = json.dumps({"agent_name": "existing"})
    monkeypatch.delenv("EXTRACTION_PROMPT_GENERATION_MODEL", raising=False)

    response = await client.put("/agent/test", json=_agent_payload([_extraction_task()]))

    assert response.status_code == 500
    assert response.json() == {"detail": "Extraction model not configured"}
