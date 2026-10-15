"""Offline tests for the Parallel Search MCP client."""

import asyncio
from typing import Self

import pytest
from mcp.types import CallToolResult, ListToolsResult, TextContent, Tool

from researchpilot import parallel_mcp
from researchpilot.parallel_mcp import ParallelMCPError, ParallelSearchMCPClient

SEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "objective": {"type": "string"},
        "search_queries": {"type": "array", "items": {"type": "string"}},
        "session_id": {"type": "string"},
    },
    "required": ["objective", "search_queries"],
}
FETCH_SCHEMA = {
    "type": "object",
    "properties": {
        "urls": {"type": "array", "items": {"type": "string"}},
        "objective": {"type": "string"},
        "search_queries": {"type": "array", "items": {"type": "string"}},
        "full_content": {"type": "boolean"},
        "session_id": {"type": "string"},
    },
    "required": ["urls"],
}
DEFAULT_TOOLS = [
    Tool(name="web_search", description="Search the web.", input_schema=SEARCH_SCHEMA),
    Tool(name="web_fetch", description="Fetch page content.", input_schema=FETCH_SCHEMA),
]
TEXT_RESULT = CallToolResult(content=[TextContent(type="text", text="Parallel result")])


class FakeMCPClient:
    """In-memory stand-in for the SDK client and its Streamable HTTP transport."""

    def __init__(
        self,
        endpoint: str,
        *,
        tools: list[Tool] | None = None,
        result: object = TEXT_RESULT,
        list_error: Exception | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.tools = DEFAULT_TOOLS if tools is None else tools
        self.result = result
        self.list_error = list_error
        self.calls: list[tuple[str, dict]] = []
        self.closed = False

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        self.closed = True

    async def list_tools(self) -> ListToolsResult:
        if self.list_error is not None:
            raise self.list_error
        return ListToolsResult(tools=self.tools)

    async def call_tool(self, name: str, arguments: dict) -> object:
        self.calls.append((name, arguments))
        return self.result


@pytest.fixture
def fake_client(monkeypatch: pytest.MonkeyPatch) -> FakeMCPClient:
    """One fresh fake per test, returned for every SDK client the code under test builds."""
    client = FakeMCPClient(endpoint="")

    def factory(endpoint: str) -> FakeMCPClient:
        client.endpoint = endpoint
        return client

    monkeypatch.setattr(parallel_mcp, "Client", factory)
    return client


def test_list_tools_exposes_discovered_names_descriptions_and_schemas(
    fake_client: FakeMCPClient,
) -> None:
    async def run() -> None:
        async with ParallelSearchMCPClient() as client:
            tools = await client.list_tools()

        assert [tool.name for tool in tools] == ["web_search", "web_fetch"]
        assert tools[0].description == "Search the web."
        assert tools[0].input_schema == SEARCH_SCHEMA
        assert fake_client.endpoint == parallel_mcp.PARALLEL_SEARCH_ENDPOINT
        assert fake_client.closed

    asyncio.run(run())


def test_search_uses_discovered_tool_name_and_schema_arguments(
    fake_client: FakeMCPClient,
) -> None:
    async def run() -> None:
        async with ParallelSearchMCPClient() as client:
            result = await client.search(
                "Find current evidence about the topic", ["topic review", "topic results"]
            )

        assert result is TEXT_RESULT
        assert len(fake_client.calls) == 1
        name, arguments = fake_client.calls[0]
        assert name == "web_search"
        assert arguments["objective"] == "Find current evidence about the topic"
        assert arguments["search_queries"] == ["topic review", "topic results"]
        assert len(arguments["session_id"]) == 32

    asyncio.run(run())


def test_fetch_wraps_one_url_and_uses_only_supported_discovered_fields(
    fake_client: FakeMCPClient,
) -> None:
    async def run() -> None:
        async with ParallelSearchMCPClient() as client:
            await client.fetch(
                "https://example.com/article",
                objective="Extract the study findings",
                search_queries=["study findings"],
                full_content=True,
            )

        name, arguments = fake_client.calls[0]
        assert name == "web_fetch"
        assert arguments["urls"] == ["https://example.com/article"]
        assert arguments["objective"] == "Extract the study findings"
        assert arguments["search_queries"] == ["study findings"]
        assert arguments["full_content"] is True
        assert len(arguments["session_id"]) == 32

    asyncio.run(run())


def test_unknown_claude_exposed_name_is_not_assumed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeMCPClient(
        parallel_mcp.PARALLEL_SEARCH_ENDPOINT,
        tools=[
            Tool(
                name="mcp__parallel-search__web_search",
                description="Search the web.",
                input_schema=SEARCH_SCHEMA,
            )
        ],
    )
    monkeypatch.setattr(parallel_mcp, "Client", lambda _endpoint: client)

    async def run() -> None:
        async with ParallelSearchMCPClient() as connected:
            with pytest.raises(ParallelMCPError, match="does not expose.*web_search"):
                await connected.search("Find evidence", ["evidence search"])

    asyncio.run(run())
    assert client.closed


def test_unrecognized_required_schema_arguments_fail_explicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schema = {
        **SEARCH_SCHEMA,
        "properties": {**SEARCH_SCHEMA["properties"], "required_mode": {"type": "string"}},
        "required": ["objective", "search_queries", "required_mode"],
    }
    client = FakeMCPClient(
        parallel_mcp.PARALLEL_SEARCH_ENDPOINT,
        tools=[Tool(name="web_search", input_schema=schema)],
    )
    monkeypatch.setattr(parallel_mcp, "Client", lambda _endpoint: client)

    async def run() -> None:
        async with ParallelSearchMCPClient() as connected:
            with pytest.raises(ParallelMCPError, match="required_mode"):
                await connected.search("Find evidence", ["evidence search"])

    asyncio.run(run())
    assert client.calls == []


@pytest.mark.parametrize(
    "tool",
    [
        Tool(name="", input_schema=SEARCH_SCHEMA),
        Tool(name="web_search", input_schema={"type": "array"}),
        Tool(
            name="web_search",
            input_schema={
                "type": "object",
                "properties": {},
                "required": ["missing_property"],
            },
        ),
    ],
)
def test_malformed_tool_definitions_fail_and_close(
    monkeypatch: pytest.MonkeyPatch, tool: Tool
) -> None:
    client = FakeMCPClient(parallel_mcp.PARALLEL_SEARCH_ENDPOINT, tools=[tool])
    monkeypatch.setattr(parallel_mcp, "Client", lambda _endpoint: client)

    async def run() -> None:
        with pytest.raises(ParallelMCPError, match="malformed"):
            async with ParallelSearchMCPClient():
                pass

    asyncio.run(run())
    assert client.closed


def test_tool_error_is_reported_without_returning_error_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error_result = CallToolResult(
        content=[TextContent(type="text", text="sensitive server detail")],
        is_error=True,
    )
    client = FakeMCPClient(parallel_mcp.PARALLEL_SEARCH_ENDPOINT, result=error_result)
    monkeypatch.setattr(parallel_mcp, "Client", lambda _endpoint: client)

    async def run() -> None:
        async with ParallelSearchMCPClient() as connected:
            with pytest.raises(ParallelMCPError, match="reported an error") as error:
                await connected.search("Find evidence", ["evidence search"])
        assert "sensitive server detail" not in str(error.value)

    asyncio.run(run())


@pytest.mark.parametrize(
    "result",
    [
        {"content": "not an MCP result"},
        CallToolResult(content=[]),
        CallToolResult(content=[TextContent(type="text", text="")]),
        CallToolResult(content=[TextContent(type="text", text=" \n\t ")]),
    ],
)
def test_malformed_or_empty_tool_results_fail(
    monkeypatch: pytest.MonkeyPatch, result: object
) -> None:
    client = FakeMCPClient(parallel_mcp.PARALLEL_SEARCH_ENDPOINT, result=result)
    monkeypatch.setattr(parallel_mcp, "Client", lambda _endpoint: client)

    async def run() -> None:
        async with ParallelSearchMCPClient() as connected:
            with pytest.raises(ParallelMCPError, match="malformed|empty"):
                await connected.search("Find evidence", ["evidence search"])

    asyncio.run(run())


def test_tool_timeout_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeMCPClient(parallel_mcp.PARALLEL_SEARCH_ENDPOINT)

    async def slow_call(_name: str, _arguments: dict) -> object:
        await asyncio.sleep(1)
        return TEXT_RESULT

    client.call_tool = slow_call  # type: ignore[method-assign]
    monkeypatch.setattr(parallel_mcp, "Client", lambda _endpoint: client)

    async def run() -> None:
        async with ParallelSearchMCPClient(timeout_seconds=0.01) as connected:
            with pytest.raises(ParallelMCPError, match="timed out"):
                await connected.search("Find evidence", ["evidence search"])

    asyncio.run(run())
    assert client.closed


def test_discovery_failure_closes_the_client(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeMCPClient(
        parallel_mcp.PARALLEL_SEARCH_ENDPOINT,
        list_error=ConnectionError("private endpoint details"),
    )
    monkeypatch.setattr(parallel_mcp, "Client", lambda _endpoint: client)

    async def run() -> None:
        with pytest.raises(ParallelMCPError, match="Could not connect"):
            async with ParallelSearchMCPClient():
                pass

    asyncio.run(run())
    assert client.closed


# Shapes match the live server's discovered schemas: an analytics-only model_name, an optional
# allow_live_fetch, and nullable anyOf fields.
LIVE_SHAPED_SEARCH_SCHEMA = {
    **SEARCH_SCHEMA,
    "properties": {**SEARCH_SCHEMA["properties"], "model_name": {"type": "string"}},
}
LIVE_SHAPED_FETCH_SCHEMA = {
    "type": "object",
    "properties": {
        "urls": {"type": "array", "items": {"type": "string"}},
        "objective": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
        "search_queries": {
            "anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "null"}],
            "default": None,
        },
        "full_content": {"type": "boolean", "default": False},
        "allow_live_fetch": {"type": "boolean", "default": True},
        "session_id": {"type": "string"},
        "model_name": {"type": "string"},
    },
    "required": ["urls"],
}


def test_live_shaped_schemas_receive_only_supported_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeMCPClient(
        parallel_mcp.PARALLEL_SEARCH_ENDPOINT,
        tools=[
            Tool(name="web_search", input_schema=LIVE_SHAPED_SEARCH_SCHEMA),
            Tool(name="web_fetch", input_schema=LIVE_SHAPED_FETCH_SCHEMA),
        ],
    )
    monkeypatch.setattr(parallel_mcp, "Client", lambda _endpoint: client)

    async def run() -> None:
        async with ParallelSearchMCPClient() as connected:
            await connected.search("Find evidence", ["evidence search"])
            await connected.fetch("https://example.com/article")

    asyncio.run(run())
    (search_name, search_args), (fetch_name, fetch_args) = client.calls
    assert search_name == "web_search"
    assert set(search_args) == {"objective", "search_queries", "session_id"}
    assert fetch_name == "web_fetch"
    assert set(fetch_args) == {"urls", "session_id", "full_content"}
    assert search_args["session_id"] == fetch_args["session_id"]


def test_entering_an_already_connected_client_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeMCPClient(parallel_mcp.PARALLEL_SEARCH_ENDPOINT)
    monkeypatch.setattr(parallel_mcp, "Client", lambda _endpoint: fake)

    async def run() -> None:
        client = ParallelSearchMCPClient()
        async with client:
            with pytest.raises(RuntimeError, match="already connected"):
                await client.__aenter__()
            assert await client.list_tools()

    asyncio.run(run())
    assert fake.closed
