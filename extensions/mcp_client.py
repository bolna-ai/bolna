"""
extensions/mcp_client.py -- MCP server connection pool.

Keeps persistent connections open to all MCP servers configured on an agent.
This eliminates the 1-3s cold-connect latency that occurs if we create a new
connection on every call start.

Design (from MIGRATION_PLAN.md Section 4):
  - Connections are PERSISTENT and POOLED (one per unique server URL).
  - NO Redis caching of MCP schemas -- always live from the connection.
  - Lazy creation on first request.
  - Transparent reconnect on failure.
  - Graceful degradation: if an MCP server is unreachable, log the error and
    continue without MCP tools.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal bookkeeping for a single live connection
# ---------------------------------------------------------------------------

@dataclass
class _MCPConnection:
    """Holds the runtime artefacts for one persistent MCP connection."""

    server_url: str
    session: ClientSession
    # We must keep references to the context-manager objects so we can
    # properly tear them down on close.
    _transport_cm: Any = field(repr=False, default=None)
    _session_cm: Any = field(repr=False, default=None)
    connected_at: float = field(default_factory=time.monotonic)
    last_used: float = field(default_factory=time.monotonic)

    def touch(self) -> None:
        self.last_used = time.monotonic()

    async def close(self) -> None:
        """Tear down the session and transport in the correct order."""
        try:
            if self._session_cm is not None:
                await self._session_cm.__aexit__(None, None, None)
        except Exception:
            logger.debug("Ignoring error closing MCP session for %s", self.server_url, exc_info=True)
        try:
            if self._transport_cm is not None:
                await self._transport_cm.__aexit__(None, None, None)
        except Exception:
            logger.debug("Ignoring error closing MCP transport for %s", self.server_url, exc_info=True)


# ---------------------------------------------------------------------------
# Connection pool
# ---------------------------------------------------------------------------

class MCPConnectionPool:
    """
    Application-level pool of live MCP server connections.

    Each entry is keyed by ``server_url`` so a single persistent connection is
    shared across all concurrent calls that target the same MCP server.  Auth
    headers are stored per-URL so reconnects can re-authenticate transparently.

    Usage::

        tools = await mcp_pool.list_tools("https://mcp.example.com/sse")
        result = await mcp_pool.execute_tool(
            "https://mcp.example.com/sse", "my_tool", {"arg": "value"}
        )
    """

    def __init__(self) -> None:
        # Keyed by server_url
        self._connections: dict[str, _MCPConnection] = {}
        # Per-URL lock so concurrent callers wait for one connect attempt
        self._locks: dict[str, asyncio.Lock] = {}
        # Global lock only for mutating the _locks / _connections dicts
        self._global_lock = asyncio.Lock()
        # Stored auth headers for reconnect
        self._auth_headers: dict[str, dict[str, str]] = {}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _get_lock(self, server_url: str) -> asyncio.Lock:
        """Return (or create) the per-URL lock."""
        async with self._global_lock:
            if server_url not in self._locks:
                self._locks[server_url] = asyncio.Lock()
            return self._locks[server_url]

    async def _connect(
        self,
        server_url: str,
        auth_headers: dict[str, str] | None = None,
    ) -> _MCPConnection:
        """
        Create a new persistent MCP session to *server_url*.

        The ``streamablehttp_client`` and ``ClientSession`` context managers
        are entered manually (via ``__aenter__``) so that the connection stays
        open beyond a single ``async with`` block.  We store the context-manager
        objects so ``_MCPConnection.close()`` can call ``__aexit__`` later.
        """
        headers = auth_headers or {}
        logger.info("MCP pool: connecting to %s", server_url)

        # 1. Enter the transport context manager
        transport_cm = streamablehttp_client(url=server_url, headers=headers)
        read_stream, write_stream, _ = await transport_cm.__aenter__()

        try:
            # 2. Enter the session context manager
            session_cm = ClientSession(read_stream, write_stream)
            session: ClientSession = await session_cm.__aenter__()
        except Exception:
            # If session creation fails, close the transport we already opened
            await transport_cm.__aexit__(None, None, None)
            raise

        try:
            # 3. Perform the MCP handshake
            await session.initialize()
        except Exception:
            # Handshake failed -- tear down cleanly
            await session_cm.__aexit__(None, None, None)
            await transport_cm.__aexit__(None, None, None)
            raise

        conn = _MCPConnection(
            server_url=server_url,
            session=session,
            _transport_cm=transport_cm,
            _session_cm=session_cm,
        )
        logger.info("MCP pool: connected to %s", server_url)
        return conn

    async def _get_connection(
        self,
        server_url: str,
        auth_headers: dict[str, str] | None = None,
    ) -> _MCPConnection:
        """
        Return a live connection, creating or reconnecting as needed.

        This is the main internal entry-point.  Per-URL locking ensures that
        even if many tasks request the same URL concurrently, only one connect
        attempt happens.
        """
        # Store headers for future reconnects
        if auth_headers is not None:
            self._auth_headers[server_url] = auth_headers

        lock = await self._get_lock(server_url)
        async with lock:
            conn = self._connections.get(server_url)
            if conn is not None:
                conn.touch()
                return conn

            # Need to create a new connection
            stored_headers = self._auth_headers.get(server_url) or auth_headers
            conn = await self._connect(server_url, stored_headers)
            self._connections[server_url] = conn
            return conn

    async def _get_connection_with_retry(
        self,
        server_url: str,
        auth_headers: dict[str, str] | None = None,
    ) -> _MCPConnection:
        """
        Like ``_get_connection`` but if the cached connection is stale (e.g.
        the MCP server restarted), transparently reconnect once.
        """
        try:
            return await self._get_connection(server_url, auth_headers)
        except Exception:
            # First attempt failed on a fresh connect -- propagate
            if server_url not in self._connections:
                raise
            # Evict the dead connection and retry once
            logger.warning(
                "MCP pool: connection to %s appears dead, reconnecting",
                server_url,
            )
            await self._evict(server_url)
            return await self._get_connection(server_url, auth_headers)

    async def _evict(self, server_url: str) -> None:
        """Close and remove a connection from the pool."""
        lock = await self._get_lock(server_url)
        async with lock:
            conn = self._connections.pop(server_url, None)
            if conn is not None:
                await conn.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def list_tools(
        self,
        server_url: str,
        auth_headers: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        List tools from a live MCP connection (~5-20 ms warm).

        Returns a list of OpenAI-function-calling-style tool dicts::

            [
                {
                    "type": "function",
                    "function": {
                        "name": "...",
                        "description": "...",
                        "parameters": { ... }
                    }
                },
                ...
            ]

        On failure, logs the error and returns an empty list (graceful
        degradation -- the agent starts without MCP tools).
        """
        try:
            conn = await self._get_connection_with_retry(server_url, auth_headers)
        except Exception:
            logger.error(
                "MCP pool: cannot connect to %s -- skipping MCP tools",
                server_url,
                exc_info=True,
            )
            return []

        try:
            tools_result = await conn.session.list_tools()
            tools = getattr(tools_result, "tools", [])
        except Exception:
            # Connection might have gone stale between _get_connection and now
            logger.warning(
                "MCP pool: list_tools failed for %s, reconnecting once",
                server_url,
            )
            await self._evict(server_url)
            try:
                conn = await self._get_connection(server_url, auth_headers)
                tools_result = await conn.session.list_tools()
                tools = getattr(tools_result, "tools", [])
            except Exception:
                logger.error(
                    "MCP pool: list_tools failed for %s after reconnect -- skipping",
                    server_url,
                    exc_info=True,
                )
                await self._evict(server_url)
                return []

        result: list[dict[str, Any]] = []
        for tool in tools:
            input_schema = getattr(tool, "inputSchema", {})
            # Fix malformed schemas: object type without properties
            if (
                isinstance(input_schema, dict)
                and input_schema.get("type") == "object"
                and "properties" not in input_schema
            ):
                input_schema["properties"] = {}

            result.append(
                {
                    "type": "function",
                    "function": {
                        "name": getattr(tool, "name", ""),
                        "description": getattr(tool, "description", ""),
                        "parameters": input_schema,
                    },
                }
            )

        logger.info(
            "MCP pool: listed %d tools from %s",
            len(result),
            server_url,
        )
        return result

    async def execute_tool(
        self,
        server_url: str,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        auth_headers: dict[str, str] | None = None,
    ) -> str:
        """
        Execute a tool on the MCP server using the pooled connection.

        Returns the concatenated text content from the tool result.
        Raises ``RuntimeError`` if the connection cannot be established or
        the tool call fails after a retry.
        """
        arguments = arguments or {}

        conn = await self._get_connection_with_retry(server_url, auth_headers)

        try:
            tool_result = await conn.session.call_tool(
                name=tool_name,
                arguments=arguments,
            )
        except Exception:
            logger.warning(
                "MCP pool: call_tool(%s) failed on %s, reconnecting once",
                tool_name,
                server_url,
            )
            await self._evict(server_url)
            conn = await self._get_connection(server_url, auth_headers)
            tool_result = await conn.session.call_tool(
                name=tool_name,
                arguments=arguments,
            )

        conn.touch()

        # Extract text from multipart content
        response_text = ""
        if hasattr(tool_result, "content") and tool_result.content:
            for part in tool_result.content:
                if hasattr(part, "text"):
                    response_text += part.text

        logger.info(
            "MCP pool: executed tool %s on %s (%d chars response)",
            tool_name,
            server_url,
            len(response_text),
        )
        return response_text

    async def close_connection(self, server_url: str) -> None:
        """Close and remove the connection for a specific server URL."""
        logger.info("MCP pool: closing connection to %s", server_url)
        await self._evict(server_url)

    async def close_all(self) -> None:
        """Close all pooled connections (call from app shutdown)."""
        logger.info("MCP pool: closing all connections")
        async with self._global_lock:
            urls = list(self._connections.keys())

        for url in urls:
            try:
                await self._evict(url)
            except Exception:
                logger.warning(
                    "MCP pool: error closing connection to %s",
                    url,
                    exc_info=True,
                )

        async with self._global_lock:
            self._connections.clear()
            self._locks.clear()
            self._auth_headers.clear()

        logger.info("MCP pool: all connections closed")

    async def health_check(self) -> dict[str, bool]:
        """
        Ping all connections; reconnect any that have failed.

        Returns a dict mapping ``server_url -> is_healthy``.  Intended to
        be called periodically (e.g. every 30-60 s) from a background task.
        """
        async with self._global_lock:
            urls = list(self._connections.keys())

        results: dict[str, bool] = {}

        for url in urls:
            lock = await self._get_lock(url)
            async with lock:
                conn = self._connections.get(url)
                if conn is None:
                    results[url] = False
                    continue

                try:
                    # Use list_tools as a lightweight ping -- it exercises the
                    # full round-trip over the live connection.
                    await conn.session.list_tools()
                    conn.touch()
                    results[url] = True
                    logger.debug("MCP pool: health OK for %s", url)
                except Exception:
                    logger.warning(
                        "MCP pool: health check failed for %s, reconnecting",
                        url,
                    )
                    results[url] = False
                    # Evict and attempt immediate reconnect
                    old_conn = self._connections.pop(url, None)
                    if old_conn is not None:
                        await old_conn.close()

                    stored_headers = self._auth_headers.get(url)
                    try:
                        new_conn = await self._connect(url, stored_headers)
                        self._connections[url] = new_conn
                        results[url] = True
                        logger.info("MCP pool: reconnected to %s", url)
                    except Exception:
                        logger.error(
                            "MCP pool: reconnect failed for %s",
                            url,
                            exc_info=True,
                        )

        return results

    @property
    def active_urls(self) -> list[str]:
        """Return the list of server URLs with active connections."""
        return list(self._connections.keys())

    def __len__(self) -> int:
        return len(self._connections)

    def __repr__(self) -> str:
        return f"<MCPConnectionPool connections={len(self._connections)}>"


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

mcp_pool = MCPConnectionPool()
