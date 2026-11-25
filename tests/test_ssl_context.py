import ssl

import aiohttp
import certifi
import pytest

from bolna.helpers import ssl_context


@pytest.fixture(autouse=True)
def _fresh_context(monkeypatch):
    monkeypatch.setattr(ssl_context, "_ssl_context", None)


def test_context_trusts_certifi_roots():
    context = ssl_context.get_ssl_context()
    assert isinstance(context, ssl.SSLContext)
    certifi_ca_count = ssl.create_default_context(cafile=certifi.where()).cert_store_stats()["x509_ca"]
    assert context.cert_store_stats()["x509_ca"] >= certifi_ca_count


def test_plain_http_url_gets_no_context():
    assert ssl_context.get_ssl_context("ws://example.com") is None


@pytest.mark.asyncio
async def test_tcp_connector_uses_shared_context():
    connector = ssl_context.tcp_connector()
    try:
        assert isinstance(connector, aiohttp.TCPConnector)
        assert connector._ssl is ssl_context.get_ssl_context()
    finally:
        await connector.close()


@pytest.mark.asyncio
async def test_no_bare_client_session_left_in_package():
    import pathlib
    import re

    root = pathlib.Path(ssl_context.__file__).resolve().parents[1]
    offenders = []
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r"aiohttp\.ClientSession\(([^)]*)", text):
            if "connector" not in match.group(1):
                offenders.append(f"{path.relative_to(root)}: {match.group(0)}")
    assert not offenders, offenders
