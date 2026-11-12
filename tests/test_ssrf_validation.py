"""Tests for the SSRF guard on user-controlled outbound URLs.

Covers ``validate_outbound_url`` directly: scheme/host/port rejection, non-public
IP literals (incl. the cloud metadata endpoint), and the DNS-pointed-inward case
where a public-looking hostname resolves to an internal address. Also covers the
connect-time checks (``outbound_connector`` for aiohttp, ``guarded_transport`` for httpx)
and the OUTBOUND_BLOCKED_CIDRS ranges.
"""

import asyncio
import errno
import ipaddress
import socket

import httpx
import openai
import pytest

from bolna.agent_types.webhook_agent import WebhookAgent
from bolna.helpers import function_calling_helpers as fch
from bolna.helpers.function_calling_helpers import (
    SSRFError,
    _guarded_socket,
    _parse_blocked_networks,
    guarded_transport,
    trigger_api,
    validate_outbound_url,
)
from bolna.llms.openai_llm import OpenAiLLM
from bolna.providers import SUPPORTED_SYNTHESIZER_MODELS


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


@pytest.mark.parametrize("url", ["ftp://host/x", "file:///etc/passwd", "gopher://h:70/"])
async def test_blocks_non_http_schemes(url):
    with pytest.raises(SSRFError):
        await validate_outbound_url(url)


@pytest.mark.parametrize("url", ["", "   ", None, "http:///nopath", "https://"])
async def test_blocks_missing_or_invalid_url(url):
    with pytest.raises(SSRFError):
        await validate_outbound_url(url)


async def test_blocks_invalid_port():
    # urlsplit raises ValueError on an out-of-range port; must surface as SSRFError
    # rather than silently passing validation and failing later in aiohttp.
    with pytest.raises(SSRFError):
        await validate_outbound_url("http://1.1.1.1:99999/")


@pytest.mark.parametrize("url", ["http://8.8.8.8/", "https://1.1.1.1/dns"])
async def test_allows_public_ip_literals(url):
    # Public IP literals resolve locally (no network) and must pass unchanged so
    # domain-less clients on public IPs keep working.
    await validate_outbound_url(url)


async def test_blocks_hostname_resolving_to_internal_ip(monkeypatch):
    """A public-looking hostname that resolves to an internal address is blocked.
    A string-only URL check would miss this (DNS pointed inward / rebinding)."""
    loop = asyncio.get_running_loop()

    async def fake_getaddrinfo(host, port, *args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("169.254.169.254", port or 80))]

    monkeypatch.setattr(loop, "getaddrinfo", fake_getaddrinfo)
    with pytest.raises(SSRFError):
        await validate_outbound_url("http://totally-public-looking.example/")


async def test_allows_hostname_resolving_to_public_ip(monkeypatch):
    loop = asyncio.get_running_loop()

    async def fake_getaddrinfo(host, port, *args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port or 80))]

    monkeypatch.setattr(loop, "getaddrinfo", fake_getaddrinfo)
    await validate_outbound_url("http://example.com/")


@pytest.fixture
def blocked_range(monkeypatch):
    """A public range configured as internal through OUTBOUND_BLOCKED_CIDRS."""
    monkeypatch.setattr(fch, "_BLOCKED_NETWORKS", (ipaddress.ip_network("44.10.0.0/16"),))


def test_parse_blocked_networks_skips_invalid_entries():
    assert _parse_blocked_networks(" 44.10.0.0/16, not-a-cidr,, 10.1.2.3/8") == (
        ipaddress.ip_network("44.10.0.0/16"),
        ipaddress.ip_network("10.0.0.0/8"),
    )


async def test_blocks_configured_range(blocked_range):
    with pytest.raises(SSRFError):
        await validate_outbound_url("http://44.10.1.1/")
    with pytest.raises(SSRFError):
        await validate_outbound_url("http://[::ffff:44.10.1.1]/")
    await validate_outbound_url("http://44.11.1.1/")


async def test_blocks_hostname_resolving_into_configured_range(blocked_range, monkeypatch):
    loop = asyncio.get_running_loop()

    async def fake_getaddrinfo(host, port, *args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("44.10.5.5", port or 80))]

    monkeypatch.setattr(loop, "getaddrinfo", fake_getaddrinfo)
    with pytest.raises(SSRFError):
        await validate_outbound_url("http://internal-looking.example/")


@pytest.mark.parametrize("ip", ["127.0.0.1", "169.254.169.254", "10.0.0.5", "44.10.0.7", "::1", "::ffff:44.10.0.7"])
def test_guarded_socket_refuses_disallowed_address(blocked_range, ip):
    family = socket.AF_INET6 if ":" in ip else socket.AF_INET
    sockaddr = (ip, 80, 0, 0) if family == socket.AF_INET6 else (ip, 80)
    with pytest.raises(OSError) as exc:
        _guarded_socket((family, socket.SOCK_STREAM, 6, "", sockaddr))
    assert exc.value.errno == errno.EACCES


def test_guarded_socket_opens_public_address(blocked_range):
    sock = _guarded_socket((socket.AF_INET, socket.SOCK_STREAM, 6, "", ("44.11.0.7", 80)))
    sock.close()


async def _call_tool(url):
    return await trigger_api(url, "GET", None, None, None, {}, None, return_response_metadata=True)


async def test_tool_call_refuses_internal_address_that_passed_preflight(monkeypatch, local_http_server):
    """A host that resolves public for the pre-flight check and internal at connect time is still refused."""
    port, paths = local_http_server

    async def preflight_passes(url):
        return None

    monkeypatch.setattr(fch, "validate_outbound_url", preflight_passes)
    monkeypatch.setattr(fch, "convert_to_request_log", lambda *a, **k: None)

    result = await _call_tool(f"http://localhost:{port}/")

    assert paths == []
    assert "blocked by outbound URL policy" in result["body"]
    assert "127.0.0.1" not in result["body"]


async def test_tool_call_reaches_allowlisted_internal_host(monkeypatch, local_http_server):
    port, paths = local_http_server
    monkeypatch.setattr(fch, "_ALLOWLISTED_HOSTS", frozenset({"localhost"}))
    monkeypatch.setattr(fch, "convert_to_request_log", lambda *a, **k: None)

    result = await _call_tool(f"http://localhost:{port}/")

    assert result["status_code"] == 200
    assert paths == ["/"]


async def test_guarded_transport_refuses_internal_address(local_http_server):
    port, paths = local_http_server
    async with httpx.AsyncClient(transport=guarded_transport()) as client:
        with pytest.raises(httpx.ConnectError, match="blocked by outbound URL policy"):
            await client.get(f"http://localhost:{port}/")
    assert paths == []


async def test_guarded_transport_dials_the_address_it_checked(monkeypatch, local_http_server):
    """The host is resolved once, and the connection goes to that checked address."""
    port, paths = local_http_server
    loop = asyncio.get_running_loop()
    lookups = []

    async def fake_getaddrinfo(host, p, *args, **kwargs):
        lookups.append(host)
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", p))]

    monkeypatch.setattr(loop, "getaddrinfo", fake_getaddrinfo)
    monkeypatch.setattr(fch, "_is_disallowed_ip", lambda ip: str(ip) != "127.0.0.1")
    async with httpx.AsyncClient(transport=guarded_transport()) as client:
        response = await client.get(f"http://pinned.example:{port}/")

    assert response.status_code == 200
    assert lookups == ["pinned.example"]
    assert paths == ["/"]


async def test_guarded_transport_reaches_allowlisted_host(monkeypatch, local_http_server):
    port, paths = local_http_server
    monkeypatch.setattr(fch, "_ALLOWLISTED_HOSTS", frozenset({"localhost"}))
    async with httpx.AsyncClient(transport=guarded_transport()) as client:
        response = await client.get(f"http://localhost:{port}/")
    assert response.status_code == 200
    assert paths == ["/"]


def _chat(llm):
    return llm.async_client.with_options(max_retries=0).chat.completions.create(
        model="m", messages=[{"role": "user", "content": "hi"}]
    )


async def test_custom_llm_client_refuses_internal_address(local_http_server):
    port, paths = local_http_server
    llm = OpenAiLLM(model="m", provider="custom", base_url=f"http://localhost:{port}/v1", llm_key="k")
    with pytest.raises(openai.APIConnectionError):
        await _chat(llm)
    assert paths == []


async def test_platform_llm_endpoint_is_not_guarded(local_http_server):
    """Only a customer endpoint is guarded; a platform endpoint may be internal."""
    port, paths = local_http_server
    llm = OpenAiLLM(model="m", base_url=f"http://localhost:{port}/v1", llm_key="k")
    await _chat(llm)
    assert paths == ["/v1/chat/completions"]


async def test_custom_tts_client_refuses_internal_address(local_http_server):
    port, paths = local_http_server
    synth = SUPPORTED_SYNTHESIZER_MODELS["custom"](
        voice="v", model="m", synthesizer_base_url=f"http://localhost:{port}/v1", synthesizer_key="k"
    )
    with pytest.raises(openai.APIConnectionError):
        await synth.async_client.with_options(max_retries=0).audio.speech.create(model="m", voice="v", input="hi")
    assert paths == []


async def test_webhook_agent_refuses_internal_address(monkeypatch, local_http_server):
    port, paths = local_http_server

    async def preflight_passes(url):
        return None

    monkeypatch.setattr("bolna.agent_types.webhook_agent.validate_outbound_url", preflight_passes)
    assert await WebhookAgent(webhook_url=f"http://localhost:{port}/hook").execute({"a": 1}) is None
    assert paths == []


async def test_guarded_transport_moves_past_an_unresponsive_address(monkeypatch, local_http_server):
    """A dead first address costs a short delay, not the whole connect timeout."""
    port, paths = local_http_server
    loop = asyncio.get_running_loop()

    async def fake_getaddrinfo(host, p, *args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, p)) for ip in ("192.0.2.1", "127.0.0.1")]

    monkeypatch.setattr(loop, "getaddrinfo", fake_getaddrinfo)
    monkeypatch.setattr(fch, "_is_disallowed_ip", lambda ip: False)
    started = loop.time()
    async with httpx.AsyncClient(transport=guarded_transport(), timeout=httpx.Timeout(5.0)) as client:
        response = await client.get(f"http://two-addresses.example:{port}/")

    assert response.status_code == 200
    assert paths == ["/"]
    assert loop.time() - started < 2
