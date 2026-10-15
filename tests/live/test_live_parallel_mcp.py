"""Opt-in smoke test against the public Parallel Search MCP server.

Skipped unless RESEARCHPILOT_LIVE_MCP_TESTS=1 is set. It makes real network calls to
https://search.parallel.ai/mcp: one tool listing, one search, and one fetch. Search
results change over time, so it checks shape, not content.
"""

import asyncio
import os

import pytest

from researchpilot.parallel_mcp import ParallelSearchMCPClient

pytestmark = [
    pytest.mark.live_mcp,
    pytest.mark.skipif(
        os.environ.get("RESEARCHPILOT_LIVE_MCP_TESTS") != "1",
        reason="set RESEARCHPILOT_LIVE_MCP_TESTS=1 to call the Parallel Search MCP server",
    ),
]

FETCH_URL = "https://docs.langchain.com/oss/python/langgraph/graph-api"


def _text(result) -> str:
    return "\n".join(
        block.text for block in result.content if isinstance(getattr(block, "text", None), str)
    )


def test_live_discovery_search_and_fetch() -> None:
    async def run() -> None:
        async with ParallelSearchMCPClient() as client:
            tools = {tool.name: tool for tool in await client.list_tools()}
            assert {"web_search", "web_fetch"} <= tools.keys()
            assert {"objective", "search_queries"} <= set(
                tools["web_search"].input_schema["required"]
            )
            assert "urls" in tools["web_fetch"].input_schema["required"]

            search = await client.search(
                "Find the official LangGraph documentation on conditional edges",
                ["LangGraph conditional edges", "add_conditional_edges docs"],
            )
            assert "http" in _text(search) or search.structured_content

            fetched = await client.fetch(
                FETCH_URL, objective="How conditional edges route between nodes"
            )
            assert "conditional" in _text(fetched).lower() or fetched.structured_content

    asyncio.run(run())
