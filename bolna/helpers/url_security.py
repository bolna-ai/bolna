"""SSRF guardrails for outbound tool and webhook requests.

User-configured tool URLs (``tools_params[].url``, ``pre_call_webhook_url``) are
fetched server-side by the voice agent, so an unrestricted URL lets a caller reach
cloud metadata endpoints (IMDS) and internal services. These helpers reject such
targets and provide a DNS-rebinding-safe aiohttp connector that re-validates the
address actually being dialed at resolution time.

Policy:
  * Cloud metadata endpoints are always blocked. They are never a legitimate tool
    target and are the highest-value SSRF destination.
  * Other private / loopback / link-local / reserved ranges are blocked unless
    ``BOLNA_ALLOW_PRIVATE_TOOL_URLS`` is set, which self-hosted deployments can use
    to let their tools call internal services.
  * Only http and https schemes are allowed.

The blocked ranges are fixed IANA standards (RFC 1918, 169.254/16, etc.), so they
are defined here rather than configured per deployment. ``validate_outbound_url``
accepts an optional ``allow_private`` override so a per-tenant policy can be layered
on later without changing this module.
"""

import ipaddress
import os
import socket
from urllib.parse import urlparse

import aiohttp
from aiohttp.abc import AbstractResolver

from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)

ALLOWED_SCHEMES = ("http", "https")

# Cloud metadata services: blocked even when private networks are allowed.
_METADATA_IPS = frozenset(ipaddress.ip_address(ip) for ip in ("169.254.169.254", "fd00:ec2::254"))
_METADATA_HOSTNAMES = frozenset({"metadata.google.internal"})

# Carrier-grade NAT, not flagged by ipaddress.is_private on some Python versions.
_EXTRA_PRIVATE_NETS = (ipaddress.ip_network("100.64.0.0/10"),)

_TRUTHY = ("1", "true", "yes", "on")


class SSRFError(ValueError):
    """Raised when an outbound request targets a blocked (non-public) address."""


def allow_private_networks():
    return os.getenv("BOLNA_ALLOW_PRIVATE_TOOL_URLS", "false").strip().lower() in _TRUTHY


def _normalize_ip(ip):
    # Unwrap IPv4-mapped IPv6 (e.g. ::ffff:169.254.169.254) to its IPv4 form so the
    # same checks apply regardless of how the address is expressed.
    if ip.version == 6 and ip.ipv4_mapped is not None:
        return ip.ipv4_mapped
    return ip


def _is_metadata_ip(ip):
    return _normalize_ip(ip) in _METADATA_IPS


def _is_private_ip(ip):
    ip = _normalize_ip(ip)
    if any(ip in net for net in _EXTRA_PRIVATE_NETS):
        return True
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def is_blocked_ip(ip_str, allow_private=None):
    """Return True if an IP address string is a blocked target."""
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return False  # Not an IP literal; hostnames are checked after resolution.
    if _is_metadata_ip(ip):
        return True
    if allow_private is None:
        allow_private = allow_private_networks()
    return not allow_private and _is_private_ip(ip)


def _is_loopback_hostname(host):
    # 'localhost' (and the reserved *.localhost TLD) never resolve to a public
    # address; catch them by name up front rather than relying on resolution.
    return host == "localhost" or host.endswith(".localhost")


def validate_outbound_url(url, allow_private=None):
    """Pre-flight check on a user-supplied URL. Raises SSRFError if blocked.

    Catches the scheme, literal-IP and known-metadata/loopback-hostname cases up
    front. Other hostnames that resolve to blocked addresses are caught at connect
    time by ``guarded_connector`` (which is DNS-rebinding-safe).
    """
    if not url or not isinstance(url, str):
        raise SSRFError(f"Invalid URL: {url!r}")
    if allow_private is None:
        allow_private = allow_private_networks()
    parsed = urlparse(url.strip())
    if parsed.scheme.lower() not in ALLOWED_SCHEMES:
        raise SSRFError(f"Blocked URL scheme {parsed.scheme!r}; only http and https are allowed")
    host = parsed.hostname
    if not host:
        raise SSRFError(f"URL has no host: {url!r}")
    host = host.lower()
    if host in _METADATA_HOSTNAMES:
        raise SSRFError(f"Blocked request to cloud metadata host {host!r}")
    if not allow_private and _is_loopback_hostname(host):
        raise SSRFError(f"Blocked request to loopback host {host!r}")
    if is_blocked_ip(host, allow_private=allow_private):
        raise SSRFError(f"Blocked request to non-public address {host!r}")
    return True


class _GuardedResolver(AbstractResolver):
    """Wraps the default resolver and rejects blocked addresses at resolution time,
    closing the DNS-rebinding gap between URL validation and the actual connection."""

    def __init__(self, allow_private=None):
        self._resolver = aiohttp.ThreadedResolver()
        self._allow_private = allow_private_networks() if allow_private is None else allow_private

    async def resolve(self, host, port=0, family=socket.AF_INET):
        infos = await self._resolver.resolve(host, port, family)
        for info in infos:
            if is_blocked_ip(info["host"], allow_private=self._allow_private):
                raise SSRFError(f"Blocked request to non-public address {info['host']} (resolved from {host!r})")
        return infos

    async def close(self):
        await self._resolver.close()


def guarded_connector(allow_private=None):
    """An aiohttp connector that blocks SSRF targets at DNS-resolution time."""
    return aiohttp.TCPConnector(resolver=_GuardedResolver(allow_private=allow_private))
