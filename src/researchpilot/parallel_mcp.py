"""Async client for the Parallel Search MCP server."""

import asyncio
from contextlib import AsyncExitStack
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Self
from uuid import uuid4

from mcp import Client
from mcp.types import CallToolResult

PARALLEL_SEARCH_ENDPOINT = "https://search.parallel.ai/mcp"
DEFAULT_TIMEOUT_SECONDS = 60.0


class ParallelMCPError(RuntimeError):
    """A connection, tool, timeout, or protocol error from the Parallel MCP client."""


@dataclass(frozen=True)
class ToolDescription:
    """Public metadata discovered from the MCP server."""

    name: str
    description: str | None
    input_schema: dict[str, Any]


class ParallelSearchMCPClient:
    """Discover and call Parallel Search MCP tools within an async context."""

    def __init__(
        self,
        endpoint: str = PARALLEL_SEARCH_ENDPOINT,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        if not endpoint.strip():
            raise ValueError("endpoint must not be blank")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        self._endpoint = endpoint
        self._timeout_seconds = timeout_seconds
        self._stack: AsyncExitStack | None = None
        self._client: Client | None = None
        self._tools: tuple[ToolDescription, ...] = ()
        self._session_id = uuid4().hex

    async def __aenter__(self) -> Self:
        if self._stack is not None:
            raise RuntimeError("ParallelSearchMCPClient is already connected.")
        self._stack = AsyncExitStack()
        try:
            async with asyncio.timeout(self._timeout_seconds):
                self._client = await self._stack.enter_async_context(Client(self._endpoint))
                listing = await self._client.list_tools()
            raw_tools = getattr(listing, "tools", None)
            self._tools = self._parse_tools(raw_tools)
        except TimeoutError as exc:
            await self._close()
            raise ParallelMCPError(
                "Timed out connecting to or discovering Parallel MCP tools."
            ) from exc
        except Exception as exc:
            await self._close()
            if isinstance(exc, ParallelMCPError):
                raise
            raise ParallelMCPError("Could not connect to or discover Parallel MCP tools.") from exc
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        await self._close()

    async def list_tools(self) -> tuple[ToolDescription, ...]:
        """Return tool names, descriptions, and input schemas discovered on connection."""
        self._require_connected()
        return self._tools

    async def search(
        self,
        objective: str,
        search_queries: list[str] | tuple[str, ...],
    ) -> CallToolResult:
        """Search the web with Parallel's discovered ``web_search`` tool."""
        tool = self._find_tool("web_search")
        arguments = self._arguments_for(
            tool,
            {
                "objective": objective,
                "search_queries": list(search_queries),
                "session_id": self._session_id,
            },
        )
        return await self._call_tool(tool.name, arguments)

    async def fetch(
        self,
        url: str,
        *,
        objective: str | None = None,
        search_queries: list[str] | tuple[str, ...] | None = None,
        full_content: bool = False,
    ) -> CallToolResult:
        """Fetch one URL with Parallel's discovered ``web_fetch`` tool."""
        tool = self._find_tool("web_fetch")
        values: dict[str, Any] = {
            "urls": [url],
            "session_id": self._session_id,
            "full_content": full_content,
        }
        if objective is not None:
            values["objective"] = objective
        if search_queries is not None:
            values["search_queries"] = list(search_queries)
        arguments = self._arguments_for(tool, values)
        return await self._call_tool(tool.name, arguments)

    def _parse_tools(self, raw_tools: object) -> tuple[ToolDescription, ...]:
        if not isinstance(raw_tools, list):
            raise ParallelMCPError("Parallel MCP returned a malformed tool listing.")
        parsed: list[ToolDescription] = []
        names: set[str] = set()
        for tool in raw_tools:
            name = getattr(tool, "name", None)
            description = getattr(tool, "description", None)
            schema = getattr(tool, "input_schema", None)
            if (
                not isinstance(name, str)
                or not name
                or name in names
                or (description is not None and not isinstance(description, str))
                or not isinstance(schema, dict)
                or schema.get("type") != "object"
                or not isinstance(schema.get("properties"), dict)
            ):
                raise ParallelMCPError("Parallel MCP returned a malformed tool definition.")
            required = schema.get("required", [])
            properties = schema["properties"]
            if not isinstance(required, list) or any(
                not isinstance(item, str) or item not in properties for item in required
            ):
                raise ParallelMCPError("Parallel MCP returned a malformed tool input schema.")
            names.add(name)
            parsed.append(
                ToolDescription(
                    name=name,
                    description=description,
                    input_schema=deepcopy(schema),
                )
            )
        return tuple(parsed)

    def _find_tool(self, name: str) -> ToolDescription:
        self._require_connected()
        tool = next((item for item in self._tools if item.name == name), None)
        if tool is None:
            raise ParallelMCPError(f"Parallel MCP does not expose the required {name!r} tool.")
        return tool

    @staticmethod
    def _arguments_for(tool: ToolDescription, values: dict[str, Any]) -> dict[str, Any]:
        schema = tool.input_schema
        properties = schema["properties"]
        arguments = {key: value for key, value in values.items() if key in properties}
        missing = set(schema.get("required", [])) - arguments.keys()
        if missing:
            fields = ", ".join(sorted(missing))
            raise ParallelMCPError(
                f"Discovered {tool.name!r} schema requires unsupported arguments: {fields}."
            )
        return arguments

    async def _call_tool(self, name: str, arguments: dict[str, Any]) -> CallToolResult:
        client = self._require_connected()
        try:
            async with asyncio.timeout(self._timeout_seconds):
                result = await client.call_tool(name, arguments)
        except TimeoutError as exc:
            raise ParallelMCPError(f"Parallel MCP tool {name!r} timed out.") from exc
        except Exception as exc:
            raise ParallelMCPError(f"Parallel MCP tool {name!r} call failed.") from exc
        if not isinstance(result, CallToolResult):
            raise ParallelMCPError(f"Parallel MCP tool {name!r} returned a malformed result.")
        if result.is_error:
            raise ParallelMCPError(f"Parallel MCP tool {name!r} reported an error.")
        if not isinstance(result.content, list):
            raise ParallelMCPError(f"Parallel MCP tool {name!r} returned malformed content.")
        has_text = any(
            isinstance(text := getattr(block, "text", None), str) and text.strip()
            for block in result.content
        )
        if not has_text and result.structured_content is None:
            raise ParallelMCPError(f"Parallel MCP tool {name!r} returned an empty result.")
        return result

    def _require_connected(self) -> Client:
        if self._stack is None or self._client is None:
            raise RuntimeError("Use ParallelSearchMCPClient inside an async with block.")
        return self._client

    async def _close(self) -> None:
        stack, self._stack = self._stack, None
        self._client = None
        self._tools = ()
        if stack is not None:
            await stack.aclose()
