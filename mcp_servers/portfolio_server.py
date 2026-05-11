"""Portfolio MCP server — stdio. Run: ``PYTHONPATH=. python -m mcp_servers.portfolio_server``."""
from __future__ import annotations

import json
from typing import Any

from mcp.server.fastmcp import FastMCP

from src.orchestration.toolkits import portfolio_tools as pt

app = FastMCP("valura_portfolio_mcp")


@app.tool()
def list_positions(positions: list[dict[str, Any]]) -> str:
    """Return holdings rows derived from the provided positions payload."""
    return json.dumps(pt.list_positions({"positions": positions}))


@app.tool()
def summarize_allocation(
    positions: list[dict[str, Any]],
    risk_profile: str | None = None,
) -> str:
    """Human-ish allocation lines for the given positions."""
    ctx = {"positions": positions, "risk_profile": risk_profile}
    return json.dumps(pt.summarize_allocation(ctx))


@app.tool()
def flag_concentration(positions: list[dict[str, Any]]) -> str:
    """Rough concentration flag using share-count weights."""
    return json.dumps(pt.flag_concentration({"positions": positions}))


@app.tool()
def match_tickers_to_positions(
    positions: list[dict[str, Any]],
    tickers: list[str],
) -> str:
    """Which requested tickers appear in positions."""
    return json.dumps(pt.match_tickers_to_positions({"positions": positions}, tickers))


@app.tool()
def snapshot_risk_profile(
    user_id: str | None = None,
    risk_profile: str | None = None,
    base_currency: str | None = None,
) -> str:
    """Echo profile hints — MCP callers pass explicit fields."""
    ctx = {"user_id": user_id, "risk_profile": risk_profile, "base_currency": base_currency}
    return json.dumps(pt.snapshot_risk_profile(ctx))


def main() -> None:
    app.run(transport="stdio")


if __name__ == "__main__":
    main()
