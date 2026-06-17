"""Tests for the SSRF guard on user-controlled outbound URLs.

Covers ``validate_outbound_url`` directly: scheme/host/port rejection, non-public
IP literals (incl. the cloud metadata endpoint), and the DNS-pointed-inward case
where a public-looking hostname resolves to an internal address.
"""

import asyncio
import socket

import pytest

from bolna.helpers.function_calling_helpers import SSRFError, validate_outbound_url


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "http://169.254.169.254/latest/meta-data/",  # AWS/GCP metadata (IMDS)
        "http://127.0.0.1/",  # loopback
        "http://10.0.0.5/internal",  # RFC-1918
        "http://192.168.1.1/",
        "http://172.16.0.1/",
        "http://[::1]/",  # IPv6 loopback
        "http://100.64.0.1/",  # CGNAT / shared address space
        "http://0.0.0.0/",  # unspecified
    ],
)
async def test_blocks_non_public_ip_literals(url):
    with pytest.raises(SSRFError):
        await validate_outbound_url(url)


@pytest.mark.asyncio
@pytest.mark.parametrize("url", ["ftp://host/x", "file:///etc/passwd", "gopher://h:70/"])
async def test_blocks_non_http_schemes(url):
    with pytest.raises(SSRFError):
        await validate_outbound_url(url)


@pytest.mark.asyncio
@pytest.mark.parametrize("url", ["", "   ", None, "http:///nopath", "https://"])
async def test_blocks_missing_or_invalid_url(url):
    with pytest.raises(SSRFError):
        await validate_outbound_url(url)


@pytest.mark.asyncio
async def test_blocks_invalid_port():
    # urlsplit raises ValueError on an out-of-range port; must surface as SSRFError
    # rather than silently passing validation and failing later in aiohttp.
    with pytest.raises(SSRFError):
        await validate_outbound_url("http://1.1.1.1:99999/")


@pytest.mark.asyncio
@pytest.mark.parametrize("url", ["http://8.8.8.8/", "https://1.1.1.1/dns"])
async def test_allows_public_ip_literals(url):
    # Public IP literals resolve locally (no network) and must pass unchanged so
    # domain-less clients on public IPs keep working.
    await validate_outbound_url(url)


@pytest.mark.asyncio
async def test_blocks_hostname_resolving_to_internal_ip(monkeypatch):
    """A public-looking hostname that resolves to an internal address is blocked.
    A string-only URL check would miss this (DNS pointed inward / rebinding)."""
    loop = asyncio.get_running_loop()

    async def fake_getaddrinfo(host, port, *args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("169.254.169.254", port or 80))]

    monkeypatch.setattr(loop, "getaddrinfo", fake_getaddrinfo)
    with pytest.raises(SSRFError):
        await validate_outbound_url("http://totally-public-looking.example/")


@pytest.mark.asyncio
async def test_allows_hostname_resolving_to_public_ip(monkeypatch):
    loop = asyncio.get_running_loop()

    async def fake_getaddrinfo(host, port, *args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port or 80))]

    monkeypatch.setattr(loop, "getaddrinfo", fake_getaddrinfo)
    await validate_outbound_url("http://example.com/")
