"""Illustrative factor / ETF overlap hints — not live fund holdings weights.
(stub map can drift vs real index weights — dont quote this as gospel)
"""
from __future__ import annotations

from typing import Any

from .portfolio_tools import list_positions

# Narrative overlap map: ETF -> equities often discussed together (stub only).
_ETF_WRAP_MEMBERS: dict[str, frozenset[str]] = {
    "QQQ": frozenset(
        {
            "NVDA",
            "TSLA",
            "MSFT",
            "AAPL",
            "META",
            "GOOGL",
            "AMZN",
            "AVGO",
            "AMD",
        }
    ),
    "SPY": frozenset({"NVDA", "TSLA", "AAPL", "MSFT", "GOOGL", "META", "AMZN"}),
    "VOO": frozenset({"AAPL", "MSFT", "NVDA", "AMZN"}),
}

_TECH_GROWTH_TICKERS = frozenset(
    {"NVDA", "TSLA", "AMD", "SMCI", "META", "QQQ", "SOXL", "ARKK"}
)


def toolkit_factor_overlap_stub(user_context: dict[str, Any]) -> dict[str, Any]:
    positions = list_positions(user_context)
    tickers = [str(p.get("ticker") or "").strip().upper() for p in positions if (p.get("ticker") or "").strip()]
    if len(tickers) < 2:
        return {
            "overlap_pairs": [],
            "cluster_theme": "n/a",
            "summary_public": "Need at least two lines on file before overlap storytelling matters.",
            "disclaimer": "Stub only.",
        }

    tick_set = set(tickers)
    held_etfs = sorted(tick_set & frozenset(_ETF_WRAP_MEMBERS.keys()))
    overlap_pairs: list[dict[str, str]] = []
    for etf in held_etfs:
        members = _ETF_WRAP_MEMBERS[etf]
        for eq in sorted(tick_set):
            if eq == etf:
                continue
            if eq in members:
                overlap_pairs.append(
                    {"etf": etf, "equity": eq, "relationship": "etf_wraps_single_name"}
                )

    tech_hits = sum(1 for t in tickers if t in _TECH_GROWTH_TICKERS)
    cluster = "high_beta_growth_tech_tilt" if tech_hits >= 2 else "mixed_or_unknown"

    lines: list[str] = []
    if overlap_pairs:
        uniq = []
        seen: set[tuple[str, str]] = set()
        for row in overlap_pairs:
            key = (row["etf"], row["equity"])
            if key in seen:
                continue
            seen.add(key)
            uniq.append(row)
        parts = [f"{r['equity']} is also a major narrative slice inside {r['etf']}" for r in uniq[:8]]
        lines.append(
            "Effective diversification is thinner than three separate tickers suggest — "
            + "; ".join(parts)
            + "."
        )
    if cluster == "high_beta_growth_tech_tilt":
        lines.append(
            "Theme read: multiple lines cluster around high-beta growth / Nasdaq tech — "
            "they often move together when rates or recession sentiment shifts."
        )

    summary = " ".join(lines) if lines else "No predefined ETF-single-name overlaps matched this stub map."

    return {
        "held_tickers": tickers,
        "overlap_pairs": overlap_pairs,
        "cluster_theme": cluster,
        "summary_public": summary,
        "disclaimer": "Illustrative overlap map only — not a live holdings breakdown.",
    }


def recession_query_hint(query: str) -> bool:
    q = (query or "").lower()
    needles = (
        "recession",
        "slowdown",
        "drawdown",
        "crash",
        "bear market",
        "safer",
        "de-risk",
        "defensive",
        "flight to quality",
    )
    return any(n in q for n in needles)
