"""
Market data access.

A *thin* wrapper around yfinance with a TTL cache so the Portfolio Health
agent doesn't smash the network on every request, and so tests don't need
internet access. If yfinance fails (offline, rate-limit, bad ticker) we
return None and the agent degrades gracefully — the structured response
still ships, just with a "price unavailable" note.

We deliberately do NOT hardcode prices or sectors. The fixtures README is
explicit about that. This module is the only thing that talks to a market
data provider; swapping yfinance for an MCP server later is one file.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable

from cachetools import TTLCache

from .config import get_settings


logger = logging.getLogger(__name__)


@dataclass
class Quote:
    ticker: str
    price: float | None
    currency: str | None
    name: str | None = None
    sector: str | None = None


@dataclass
class HistoryPoint:
    ts: datetime
    close: float


# Indices we benchmark against. Mapping is intentional — the assignment
# pegs us to canonical names like "S&P 500" / "FTSE 100" / "NIKKEI 225".
BENCHMARK_TICKER: dict[str, str] = {
    "S&P 500": "^GSPC",
    "FTSE 100": "^FTSE",
    "NIKKEI 225": "^N225",
    "MSCI World": "URTH",  # iShares MSCI World ETF — proxy
    "QQQ": "QQQ",
}


_settings = None
_quote_cache: TTLCache | None = None
_hist_cache: TTLCache | None = None


def _ensure_caches() -> tuple[TTLCache, TTLCache]:
    global _settings, _quote_cache, _hist_cache
    if _settings is None:
        _settings = get_settings()
    if _quote_cache is None:
        _quote_cache = TTLCache(maxsize=512, ttl=_settings.market_data_cache_ttl_s)
    if _hist_cache is None:
        _hist_cache = TTLCache(maxsize=128, ttl=_settings.market_data_cache_ttl_s)
    return _quote_cache, _hist_cache


def reset_market_data_caches() -> None:
    global _quote_cache, _hist_cache
    _quote_cache = None
    _hist_cache = None


def get_quote(ticker: str) -> Quote | None:
    qc, _ = _ensure_caches()
    if ticker in qc:
        return qc[ticker]

    try:
        import yfinance as yf  # heavy import, deferred
    except Exception:
        logger.warning("yfinance unavailable; returning empty quote for %s", ticker)
        return None

    try:
        t = yf.Ticker(ticker)
        info = getattr(t, "fast_info", None)
        price = None
        currency = None
        if info is not None:
            price = getattr(info, "last_price", None) or getattr(info, "lastPrice", None)
            currency = getattr(info, "currency", None)

        if price is None:
            # fast_info can be empty for some tickers — fall back to a 1d hist
            hist = t.history(period="5d")
            if not hist.empty:
                price = float(hist["Close"].dropna().iloc[-1])

        # we don't always need the slow `info` payload
        name = None
        sector = None
        try:
            slow = t.info  # may make a network call
            name = slow.get("shortName") or slow.get("longName")
            sector = slow.get("sector")
            currency = currency or slow.get("currency")
        except Exception:
            pass

        quote = Quote(
            ticker=ticker,
            price=float(price) if price is not None else None,
            currency=currency,
            name=name,
            sector=sector,
        )
        qc[ticker] = quote
        return quote
    except Exception as exc:
        logger.warning("Failed to fetch quote for %s: %s", ticker, exc)
        # cache the negative result briefly to stop a thundering herd
        qc[ticker] = None
        return None


def get_history_returns(ticker: str, *, lookback_days: int = 365) -> list[HistoryPoint]:
    """Daily close points over `lookback_days`. Empty list if unavailable."""
    _, hc = _ensure_caches()
    cache_key = (ticker, lookback_days)
    if cache_key in hc:
        return hc[cache_key]

    try:
        import yfinance as yf
    except Exception:
        return []

    try:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=lookback_days)
        t = yf.Ticker(ticker)
        hist = t.history(start=start, end=end, auto_adjust=True)
        points = [
            HistoryPoint(ts=idx.to_pydatetime(), close=float(row["Close"]))
            for idx, row in hist.iterrows()
            if not row.isna().any()
        ]
        hc[cache_key] = points
        return points
    except Exception as exc:
        logger.warning("Failed to fetch history for %s: %s", ticker, exc)
        hc[cache_key] = []
        return []


def get_quotes_bulk(tickers: Iterable[str]) -> dict[str, Quote | None]:
    out: dict[str, Quote | None] = {}
    for t in tickers:
        out[t] = get_quote(t)
    return out


def benchmark_ticker(name: str | None) -> str | None:
    if not name:
        return None
    return BENCHMARK_TICKER.get(name)
