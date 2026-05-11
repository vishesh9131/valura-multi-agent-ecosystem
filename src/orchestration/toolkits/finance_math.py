"""Tiny deterministic gain helpers — not tax advice, no brackets."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ...market_data import get_quote


def estimate_unrealized_gain_per_share(
    avg_cost: float | None,
    last_price: float | None,
) -> dict[str, Any]:
    if last_price is None:
        return {"gain_per_share": None, "note": "price_unavailable"}
    try:
        px = float(last_price)
    except (TypeError, ValueError):
        return {"gain_per_share": None, "note": "price_unparseable"}
    try:
        basis = float(avg_cost) if avg_cost is not None else 0.0
    except (TypeError, ValueError):
        basis = 0.0
    if basis <= 0:
        return {"gain_per_share": px - basis, "note": "basis_missing_or_zero"}
    return {"gain_per_share": px - basis, "note": None}


def estimate_position_gain_notional(
    quantity: float,
    avg_cost: float | None,
    last_price: float | None,
) -> dict[str, Any]:
    gps = estimate_unrealized_gain_per_share(avg_cost, last_price)
    g = gps.get("gain_per_share")
    try:
        q = float(quantity)
    except (TypeError, ValueError):
        q = 0.0
    notional = None if g is None else round(float(g) * q, 4)
    return {
        "estimated_unrealized_gain_notional": notional,
        "gain_per_share": g,
        "per_share_note": gps.get("note"),
    }


def holding_period_class(purchased_at: str | None) -> str:
    raw = (purchased_at or "").strip()
    if not raw:
        return "unknown"
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return "unknown"
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    days = (datetime.now(timezone.utc) - dt).days
    if days >= 365:
        return "long_term"
    return "short_term"


def toolkit_tax_gain_bundle(user_context: dict[str, Any], *, ticker: str) -> dict[str, Any]:
    sym = (ticker or "").strip().upper()
    positions = user_context.get("positions") or []
    match: dict[str, Any] | None = None
    for p in positions:
        if not isinstance(p, dict):
            continue
        if str(p.get("ticker") or "").strip().upper() == sym:
            match = p
            break

    qty_raw = match.get("quantity") if match else None
    try:
        qty = float(qty_raw) if qty_raw is not None else 0.0
    except (TypeError, ValueError):
        qty = 0.0

    avg_raw = match.get("avg_cost") if match else None
    try:
        avg_cost = float(avg_raw) if avg_raw is not None else 0.0
    except (TypeError, ValueError):
        avg_cost = 0.0

    purchased_at = match.get("purchased_at") if match else None
    hp = holding_period_class(purchased_at if isinstance(purchased_at, str) else None)

    qte = get_quote(sym) if sym else None
    last_px = float(qte.price) if qte and qte.price is not None else None

    gn = estimate_position_gain_notional(qty, avg_cost, last_px)
    basis_note = gn.get("per_share_note") or "ok"

    disclaim = (
        "Educational estimate only — not tax advice; brackets, NIIT, wash sales, and lot choice not modeled."
    )

    summary_public = (
        f"{sym or '(no symbol)'}: stub quote ~{_fmt_px(last_px)}, qty {qty:g}, avg cost {avg_cost:g}. "
        f"Rough unrealized gain about {_fmt_money(gn.get('estimated_unrealized_gain_notional'))} "
        f"(per-share note: {basis_note}); holding bucket `{hp}` from purchase metadata when present. {disclaim}"
    )

    summary_internal = (
        f"tax_bundle ticker={sym} px={last_px} qty={qty} avg={avg_cost} hp={hp} "
        f"notional_gain={gn.get('estimated_unrealized_gain_notional')}"
    )

    return {
        "ticker": sym or None,
        "quantity": qty,
        "avg_cost": avg_cost,
        "last_price": last_px,
        "estimated_unrealized_gain_per_share": gn.get("gain_per_share"),
        "estimated_unrealized_gain_notional": gn.get("estimated_unrealized_gain_notional"),
        "holding_period_class": hp,
        "basis_note": basis_note,
        "disclaimer": disclaim,
        "summary_public": summary_public.strip(),
        "summary_internal": summary_internal.strip(),
    }


def _fmt_px(x: Any) -> str:
    try:
        return f"{float(x):,.2f}"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_money(x: Any) -> str:
    if x is None:
        return "n/a"
    try:
        return f"{float(x):,.2f}"
    except (TypeError, ValueError):
        return "n/a"
