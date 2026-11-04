import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import pytest_asyncio

from bolna.helpers import rag_service_client
from bolna.helpers.rag_service_client import RAGServiceClientSingleton


class RecordingClient:
    instances = []

    def __init__(self, base_url):
        time.sleep(0.01)
        self.base_url = base_url
        self.closed = False
        type(self).instances.append(self)

    async def close(self):
        self.closed = True


@pytest_asyncio.fixture(autouse=True)
async def reset_client_pool():
    await RAGServiceClientSingleton.close_all_clients()
    RecordingClient.instances = []
    yield
    await RAGServiceClientSingleton.close_all_clients()


@pytest.mark.asyncio
async def test_same_normalized_url_reuses_client():
    first = await RAGServiceClientSingleton.get_client("https://rag.example/")
    second = await RAGServiceClientSingleton.get_client("https://rag.example")

    assert second is first
    assert first.base_url == "https://rag.example"
    assert len(RAGServiceClientSingleton._clients) == 1


@pytest.mark.asyncio
async def test_different_urls_use_different_clients():
    first = await RAGServiceClientSingleton.get_client("https://rag-a.example")
    second = await RAGServiceClientSingleton.get_client("https://rag-b.example")

    assert second is not first
    assert first.base_url == "https://rag-a.example"
    assert second.base_url == "https://rag-b.example"


@pytest.mark.asyncio
async def test_close_all_clients_closes_every_client_and_empties_pool():
    first = await RAGServiceClientSingleton.get_client("https://rag-a.example")
    second = await RAGServiceClientSingleton.get_client("https://rag-b.example")
    await first._ensure_session()
    await second._ensure_session()
    first_session = first.session
    second_session = second.session

    await RAGServiceClientSingleton.close_all_clients()

    assert first_session.closed
    assert second_session.closed
    assert RAGServiceClientSingleton._clients == {}


@pytest.mark.asyncio
async def test_closed_client_is_replaced_when_reacquired():
    first = await RAGServiceClientSingleton.get_client("https://rag.example")
    await first._ensure_session()
    await RAGServiceClientSingleton.close_client()

    second = await RAGServiceClientSingleton.get_client("https://rag.example/")
    await second._ensure_session()

    assert first.session is None
    assert second is not first
    assert not second.session.closed


def test_concurrent_acquisition_creates_only_one_client(monkeypatch):
    monkeypatch.setattr(rag_service_client, "RAGServiceClient", RecordingClient)

    def acquire_client():
        return asyncio.run(RAGServiceClientSingleton.get_client("https://rag.example/"))

    with ThreadPoolExecutor(max_workers=8) as executor:
        clients = list(executor.map(lambda _: acquire_client(), range(16)))

    assert all(client is clients[0] for client in clients)
    assert len(RecordingClient.instances) == 1
