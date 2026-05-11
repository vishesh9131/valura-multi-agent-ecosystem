"""Portfolio-side tools (holdings, concentration, profile hints)."""
from __future__ import annotations

from typing import Any


def list_positions(user_context: dict[str, Any]) -> list[dict[str, Any]]:
    raw = user_context.get("positions") or []
    out: list[dict[str, Any]] = []
    for p in raw:
        if isinstance(p, dict):
            out.append(p)
    return out


def summarize_allocation(user_context: dict[str, Any]) -> dict[str, Any]:
    positions = list_positions(user_context)
    if not positions:
        return {"mode": "empty", "lines": ["No positions on file for this user."], "position_count": 0}
    total_qty = sum(float(p.get("quantity") or 0) for p in positions)
    lines = []
    for p in positions:
        t = str(p.get("ticker") or "?")
        q = float(p.get("quantity") or 0)
        share = (q / total_qty * 100.0) if total_qty > 0 else 0.0
        lines.append(f"{t}: qty={q:.4g}, ~{share:.1f}% of share count")
    return {"mode": "holdings", "lines": lines, "position_count": len(positions)}


def flag_concentration(user_context: dict[str, Any]) -> dict[str, Any]:
    positions = list_positions(user_context)
    if not positions:
        return {"flag": "n/a", "detail": "nothing held"}
    top = max(positions, key=lambda p: float(p.get("quantity") or 0))
    top_t = str(top.get("ticker") or "?")
    total_q = sum(float(p.get("quantity") or 0) for p in positions)
    top_q = float(top.get("quantity") or 0)
    pct = (top_q / total_q * 100.0) if total_q > 0 else 0.0
    flag = "warning" if pct >= 45 else "ok"
    return {"flag": flag, "top_holding": top_t, "top_share_pct": round(pct, 1)}


def match_tickers_to_positions(user_context: dict[str, Any], tickers: list[str]) -> dict[str, Any]:
    tickers_u = [str(t).strip().upper() for t in tickers if str(t).strip()]
    held = {str(p.get("ticker") or "").strip().upper() for p in list_positions(user_context)}
    matched = [t for t in tickers_u if t in held]
    missing = [t for t in tickers_u if t not in held]
    return {"matched_in_portfolio": matched, "not_held": missing}


def snapshot_risk_profile(user_context: dict[str, Any]) -> dict[str, Any]:
    return {
        "risk_profile": user_context.get("risk_profile"),
        "base_currency": user_context.get("base_currency"),
        "user_id": user_context.get("user_id"),
    }
