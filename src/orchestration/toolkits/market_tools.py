"""Market data tools — thin wrappers so MCP + supervisor share one implementation."""
from __future__ import annotations

from typing import Any

from ... import market_data


def toolkit_get_quote(ticker: str) -> dict[str, Any]:
    t = (ticker or "").strip().upper()
    if not t:
        return {"error": "empty_ticker"}
    q = market_data.get_quote(t)
    if q is None:
        return {"ticker": t, "price": None, "note": "quote_unavailable"}
    return {
        "ticker": q.ticker,
        "price": q.price,
        "currency": q.currency,
        "name": q.name,
        "sector": q.sector,
    }


def toolkit_get_recent_returns(ticker: str, days: int = 30) -> dict[str, Any]:
    t = (ticker or "").strip().upper()
    if not t:
        return {"error": "empty_ticker"}
    pts = market_data.get_history_returns(t, lookback_days=max(7, min(int(days), 365)))
    if len(pts) < 2:
        return {"ticker": t, "return_pct": None, "note": "insufficient_history"}
    first = pts[0].close
    last = pts[-1].close
    if first <= 0:
        return {"ticker": t, "return_pct": None, "note": "bad_anchor"}
    ret = (last / first - 1.0) * 100.0
    return {"ticker": t, "lookback_days": days, "return_pct": round(ret, 2)}


def toolkit_sector_benchmark_stub(sector: str | None) -> dict[str, Any]:
    s = (sector or "unknown").strip() or "unknown"
    # canned benchmark names — not live levels; interviewer demo safe
    return {
        "sector": s,
        "benchmark_proxy": "MSCI World sector ETF (stub)",
        "comment": "Benchmark levels not fetched in stub mode.",
    }


def toolkit_symbol_resolve(query: str) -> dict[str, Any]:
    q = (query or "").strip().upper()
    if not q:
        return {"candidates": []}
    if len(q) <= 5 and q.isalnum():
        return {"candidates": [q], "note": "ticker-shaped"}
    return {"candidates": [q.split()[0][:8]], "note": "heuristic"}


def toolkit_market_hours_stub(exchange: str = "US") -> dict[str, Any]:
    return {
        "exchange": exchange,
        "regular_session": "09:30-16:00 local (stub)",
        "status": "unknown_clock",
    }
