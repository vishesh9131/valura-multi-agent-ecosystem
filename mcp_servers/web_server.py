"""Web MCP server — stdio. Run: ``PYTHONPATH=. python -m mcp_servers.web_server``."""
from __future__ import annotations

import json

from mcp.server.fastmcp import FastMCP

from src.orchestration.toolkits import web_tools as wt

app = FastMCP("valura_web_mcp")


@app.tool()
def web_search(query: str, max_results: int = 5) -> str:
    """Lightweight HTML search (DuckDuckGo). Returns JSON list of title/url/snippet."""
    return json.dumps(wt.toolkit_web_search(query, max_results=max_results))


@app.tool()
def fetch_url(url: str, max_chars: int = 12000) -> str:
    """HTTPS GET; plain text excerpt. SSRF-filtered."""
    return json.dumps(wt.toolkit_fetch_url(url, max_chars=max_chars))


def main() -> None:
    app.run(transport="stdio")


if __name__ == "__main__":
    main()
