"""check_outbound_url: the DNS-free half of the SSRF guard, shared by save-time validation and validate_outbound_url."""

import asyncio
import socket

import pytest

from bolna.helpers import function_calling_helpers as fch
from bolna.helpers.function_calling_helpers import SSRFError, check_outbound_url, validate_outbound_url


@pytest.mark.parametrize(
    "url",
    [
        "",
        "ftp://files.example.com/a",
        "http:///no-host",
        "https://api.example.com:99999/",
        "http://169.254.169.254/latest/meta-data/iam/security-credentials",
        "http://10.0.0.5/hook",
        "http://[::1]/",
        "http://127.1/",
        "http://2130706433/",
        "http://0x7f000001/",
        "https://YOUR-PUBLISHED-DOMAIN/api/public/send-payment-link",
        "http://redis:6379/",
        "http://printer.local/",
        "http://ip-10-0-0-1.ec2.internal/",
        "http://api.localhost/",
    ],
)
def test_check_outbound_url_rejects_without_dns(url, monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: pytest.fail("check_outbound_url must not resolve"))
    with pytest.raises(SSRFError):
        check_outbound_url(url)


def test_check_outbound_url_leaves_a_public_hostname_to_dns():
    assert check_outbound_url("https://API.example.com:8443/v1?x=1") == ("api.example.com", 8443)


def test_check_outbound_url_leaves_an_invalid_placeholder_host_to_dns():
    assert check_outbound_url("https://tools.invalid/api/voice/bolna/0") == ("tools.invalid", None)


@pytest.mark.parametrize("url", ["https://93.184.216.34/hook", "http://[2606:2800:220:1:248:1893:25c8:1946]/"])
def test_check_outbound_url_settles_a_public_ip_literal(url):
    assert check_outbound_url(url) is None


def test_check_outbound_url_skips_an_allowlisted_host(monkeypatch):
    monkeypatch.setattr(fch, "_ALLOWLISTED_HOSTS", frozenset({"ws-server"}))
    assert check_outbound_url("http://ws-server:5001/x") is None


async def test_validate_rejects_a_single_label_host_before_resolving(monkeypatch):
    async def fake_getaddrinfo(*args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 80))]

    monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", fake_getaddrinfo)
    with pytest.raises(SSRFError, match="not a public domain name"):
        await validate_outbound_url("http://api/v1")
