"""Market MCP server — stdio. Run: ``PYTHONPATH=. python -m mcp_servers.market_server``."""
from __future__ import annotations

import json

from mcp.server.fastmcp import FastMCP

from src.orchestration.toolkits import market_tools as mt

app = FastMCP("valura_market_mcp")


@app.tool()
def get_quote(ticker: str) -> str:
    """Latest quote snapshot (best-effort via yfinance when online)."""
    return json.dumps(mt.toolkit_get_quote(ticker))


@app.tool()
def get_recent_returns(ticker: str, days: int = 30) -> str:
    """Approx total return percent over ``days`` using daily closes."""
    return json.dumps(mt.toolkit_get_recent_returns(ticker, days=days))


@app.tool()
def sector_benchmark_stub(sector: str | None = None) -> str:
    """Stub benchmark mapping for demo / offline."""
    return json.dumps(mt.toolkit_sector_benchmark_stub(sector))


@app.tool()
def symbol_resolve(query: str) -> str:
    """Heuristic resolver — prefers uppercase ticker-shaped tokens."""
    return json.dumps(mt.toolkit_symbol_resolve(query))


@app.tool()
def market_hours_stub(exchange: str = "US") -> str:
    """Static session hours placeholder."""
    return json.dumps(mt.toolkit_market_hours_stub(exchange))


def main() -> None:
    app.run(transport="stdio")


if __name__ == "__main__":
    main()
