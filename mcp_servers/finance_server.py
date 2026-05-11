"""Finance MCP — stdio. Run: ``PYTHONPATH=. python -m mcp_servers.finance_server``."""
from __future__ import annotations

import json

from mcp.server.fastmcp import FastMCP

from src.orchestration.toolkits import finance_math as fm

app = FastMCP("valura_finance_mcp")


@app.tool()
def estimate_gain_bundle(user_context_json: str, ticker: str) -> str:
    """JSON user_context plus ticker -> deterministic gain / holding-period bundle."""
    ctx = json.loads(user_context_json)
    return json.dumps(fm.toolkit_tax_gain_bundle(ctx, ticker=ticker))


@app.tool()
def holding_period_class(purchased_at: str | None = None) -> str:
    """ISO purchase date -> long_term | short_term | unknown."""
    return json.dumps({"holding_period_class": fm.holding_period_class(purchased_at)})


def main() -> None:
    app.run(transport="stdio")


if __name__ == "__main__":
    main()
