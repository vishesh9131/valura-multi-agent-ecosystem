"""Heuristic multi-domain flags when the router forgets to spell things out."""
from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

from .classifier import Classification
from .orchestration.toolkits import portfolio_tools as port_tools

_TAXISH = re.compile(
    r"\b(tax|taxes|taxable|capital\s+gains?|ltcg|stcg|short[- ]term|long[- ]term|harvest)\b",
    re.I,
)
_TICKER_RE = re.compile(r"\b([A-Z]{1,5})\b")
_BOGUS = frozenset(
    {
        "I",
        "A",
        "AN",
        "AS",
        "AT",
        "BE",
        "BY",
        "DO",
        "GO",
        "IF",
        "IN",
        "IS",
        "IT",
        "ME",
        "MY",
        "NO",
        "OF",
        "ON",
        "OR",
        "SO",
        "TO",
        "UP",
        "WE",
        "OK",
    }
)


def _tickers_from_query(query: str, positions: list[dict[str, Any]]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for p in positions:
        t = str(p.get("ticker") or "").strip().upper()
        if t and t not in seen:
            seen.add(t)
            ordered.append(t)
    for raw in _TICKER_RE.findall(query or ""):
        t = raw.upper()
        if len(t) < 2 or t in _BOGUS:
            continue
        if t not in seen:
            seen.add(t)
            ordered.append(t)
    return ordered[:8]


def merge_task_decomposition(
    classification: Classification,
    query: str,
    user_context: dict[str, Any],
) -> Classification:
    td = dict(classification.task_decomposition or {})
    td.setdefault("primary_theme", None)
    td.setdefault("sub_tasks", list(td.get("sub_tasks") or []))
    td.setdefault("domains", [str(d).lower() for d in (td.get("domains") or [])])
    td.setdefault("requires_multi_agent", bool(td.get("requires_multi_agent")))

    if classification.agent == "investment_debate":
        td["primary_theme"] = td.get("primary_theme") or "investment_debate"
        td["requires_multi_agent"] = True
        subs = list(td["sub_tasks"])
        for label in ("bull_case_generation", "bear_case_generation", "long_horizon_analysis"):
            if label not in subs:
                subs.append(label)
        td["sub_tasks"] = subs
        doms = set(td["domains"])
        doms.update({"strategy", "fundamentals", "risk"})
        td["domains"] = sorted(doms)

    q = (query or "").strip()
    ql = q.lower()
    ents = classification.entities or {}
    has_tax = bool(_TAXISH.search(q))
    has_sell = ents.get("action") == "sell" or bool(re.search(r"\b(sell|selling)\b", ql))

    positions = port_tools.list_positions(user_context)
    tickers = [str(t).strip().upper() for t in (ents.get("tickers") or []) if str(t).strip()]
    if not tickers:
        tickers = _tickers_from_query(q, positions)

    # nvda / timing / capital gains style mashups
    if has_tax and has_sell and tickers:
        td["requires_multi_agent"] = True
        doms = set(td["domains"])
        doms.update({"tax", "market", "strategy"})
        td["domains"] = sorted(doms)
        if not td.get("primary_theme"):
            td["primary_theme"] = "tax_timing_tradeoff"
        subs = list(td["sub_tasks"])
        for label in ("capital_gains_context", "market_timing", "strategy_tradeoffs"):
            if label not in subs:
                subs.append(label)
        td["sub_tasks"] = subs
    elif td.get("requires_multi_agent"):
        doms = set(td["domains"])
        if "tax" in doms and "market" not in doms:
            doms.add("market")
        td["domains"] = sorted(doms)

    return replace(classification, task_decomposition=td)
